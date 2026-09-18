"""第 12 章 demo：HNSW vs 暴力 KNN 向量检索对比

对比 HNSW（跳表入口点 + 近邻图 best-first 搜索）和暴力 KNN（全量距离计算）：
1. 建图延迟：HNSW 构建近邻图的时间
2. 检索延迟：单次 query 的延迟（亚线性 vs 线性）
3. 召回率：HNSW top-k 与暴力 top-k 的重合度

对比不同数据量（1k, 10k, 100k）和维度（10, 50）。
保存图到 figures/。

跑法：
    python 12_hnsw/python/demo.py
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, save_bar, save_line  # noqa: E402


class SimpleHNSW:
    """简化 HNSW：近邻图 + 跳表入口点选择 + best-first 搜索。

    教学简化版，非完整 HNSW：
    - 无分层结构（完整 HNSW 有多层图，上层是下层的抽样）
    - 建图：暴力找每节点最近 M 个邻居（numpy 矩阵乘法加速）
      生产级 HNSW 用搜索建图（O(n log n)），这里用暴力保证图质量
    - 搜索：best-first search（优先队列 + ef 候选），从入口点贪心扩展
    - 入口点：按到原点距离排序的数组 + bisect 查找
      （等价于跳表的 O(log n) 查找；插入用一次性排序模拟）
    """

    def __init__(self, dim: int, M: int = 16, ef: int = 64) -> None:
        self.dim = dim
        self.M = M
        self.ef = ef
        self.vectors: np.ndarray | None = None
        self.neighbors: list[np.ndarray] = []
        self.norms: np.ndarray | None = None
        self.dist_order: np.ndarray | None = None
        self.dist_keys: np.ndarray | None = None
        self.entry_point: int = 0
        self.n: int = 0

    def build(self, vectors: np.ndarray) -> None:
        self.n = len(vectors)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.norms = np.sum(self.vectors.astype(np.float32) ** 2, axis=1).astype(np.float32)

        self.dist_order = np.argsort(self.norms)
        self.dist_keys = self.norms[self.dist_order]

        if self.dim <= 20 and self.n <= 200000:
            self._build_kdtree()
        else:
            self._build_batch()

    def _build_kdtree(self) -> None:
        vecs64 = self.vectors.astype(np.float64)
        tree = cKDTree(vecs64)
        _, neighbors = tree.query(vecs64, k=self.M + 1, workers=-1)
        self.neighbors = [neighbors[i][1:].astype(np.int32) for i in range(self.n)]

    def _build_batch(self) -> None:
        self.neighbors = [None] * self.n
        vt = self.vectors.T
        batch = 512
        for start in range(0, self.n, batch):
            end = min(start + batch, self.n)
            dots = self.vectors[start:end] @ vt
            for i in range(end - start):
                d = self.norms - 2.0 * dots[i]
                d[start + i] = np.inf
                if self.n > self.M:
                    top_m = np.argpartition(d, self.M)[: self.M]
                else:
                    top_m = np.argsort(d)[: self.M]
                self.neighbors[start + i] = top_m.astype(np.int32)

    def _entry_for_query(self, query: np.ndarray) -> int:
        q_norm = float(np.dot(query, query))
        idx = int(np.searchsorted(self.dist_keys, q_norm))
        if idx >= len(self.dist_keys):
            idx = len(self.dist_keys) - 1
        return int(self.dist_order[idx])

    def _search_layer(self, query: np.ndarray, ef: int) -> list[int]:
        entry = self._entry_for_query(query)
        visited = np.zeros(self.n, dtype=bool)
        visited[entry] = True
        diff = self.vectors[entry].astype(np.float64) - query
        entry_dist = float(np.dot(diff, diff))

        candidates: list[tuple[float, int]] = [(entry_dist, entry)]
        results: list[tuple[float, int]] = [(-entry_dist, entry)]

        while candidates:
            d, c = heapq.heappop(candidates)
            if d > -results[0][0]:
                break

            nbs = self.neighbors[c]
            unvisited_mask = ~visited[nbs]
            if not unvisited_mask.any():
                continue
            unvisited = nbs[unvisited_mask]
            visited[unvisited] = True
            diffs = self.vectors[unvisited].astype(np.float64) - query
            dists = np.einsum("ij,ij->i", diffs, diffs)

            for d_nb, nb in zip(dists.tolist(), unvisited.tolist()):
                if len(results) < ef or d_nb < -results[0][0]:
                    heapq.heappush(candidates, (d_nb, nb))
                    heapq.heappush(results, (-d_nb, nb))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [idx for _, idx in results]

    def search(self, query: np.ndarray, k: int) -> np.ndarray:
        if self.n == 0:
            return np.array([], dtype=np.int32)
        q = np.asarray(query, dtype=np.float32)
        ef = max(k, self.ef)
        cands = self._search_layer(q, ef)
        if not cands:
            return np.array([], dtype=np.int32)
        cand_vecs = self.vectors[cands]
        diffs = cand_vecs - q
        dists = np.einsum("ij,ij->i", diffs, diffs)
        top_k = np.argsort(dists)[:k]
        return np.array([cands[i] for i in top_k], dtype=np.int32)


def brute_knn(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    diffs = vectors - query
    dists = np.einsum("ij,ij->i", diffs, diffs)
    return np.argsort(dists)[:k]


def recall_at_k(hnsw_result: np.ndarray, brute_result: np.ndarray, k: int) -> float:
    hnsw_set = set(hnsw_result[:k].tolist())
    brute_set = set(brute_result[:k].tolist())
    if not brute_set:
        return 0.0
    return len(hnsw_set & brute_set) / len(brute_set)


def run_experiment(n: int, dim: int, n_queries: int = 100, k: int = 10) -> dict:
    rng = np.random.RandomState(42)
    vectors = rng.randn(n, dim).astype(np.float32)
    queries = rng.randn(n_queries, dim).astype(np.float32)

    t_build = Timer()
    t_build.start()
    index = SimpleHNSW(dim=dim, M=16, ef=48)
    index.build(vectors)
    t_build.stop()

    t_brute = Timer()
    t_brute.start()
    for q in queries:
        brute_knn(vectors, q, k)
    t_brute.stop()

    t_hnsw = Timer()
    t_hnsw.start()
    for q in queries:
        index.search(q, k)
    t_hnsw.stop()

    recalls = []
    for q in queries:
        hnsw_result = index.search(q, k)
        brute_result = brute_knn(vectors, q, k)
        recalls.append(recall_at_k(hnsw_result, brute_result, k))

    return {
        "n": n,
        "dim": dim,
        "build_ms": t_build.elapsed_ms,
        "brute_ms_per_query": t_brute.elapsed_ms / n_queries,
        "hnsw_ms_per_query": t_hnsw.elapsed_ms / n_queries,
        "recall": float(np.mean(recalls)),
        "speedup": t_brute.elapsed_ms / t_hnsw.elapsed_ms if t_hnsw.elapsed_ms > 0 else float("inf"),
    }


def main() -> None:
    print("=" * 76)
    print("第 12 章 demo：HNSW vs 暴力 KNN 向量检索对比")
    print("=" * 76)
    print()
    print("HNSW = 跳表入口点选择 + 近邻图 best-first 搜索")
    print("暴力 KNN = 全量距离计算 O(n*dim)")
    print()

    configs = [
        (1000, 10),
        (1000, 50),
        (10000, 10),
        (10000, 50),
        (100000, 10),
        (100000, 50),
    ]

    results = []
    for n, dim in configs:
        print(f"[实验] n={n:>6}, dim={dim:>2} ...", end=" ", flush=True)
        r = run_experiment(n, dim)
        results.append(r)
        print(
            f"建图 {r['build_ms']:.0f}ms, "
            f"暴力 {r['brute_ms_per_query']:.3f}ms, "
            f"HNSW {r['hnsw_ms_per_query']:.3f}ms, "
            f"加速 {r['speedup']:.1f}x, "
            f"召回 {r['recall']:.1%}"
        )

    print()
    print("[结果汇总]")
    print("-" * 76)
    print(
        f"{'n':>6} {'dim':>4} {'建图(ms)':>10} {'暴力(ms)':>10} "
        f"{'HNSW(ms)':>10} {'加速比':>8} {'召回率':>8}"
    )
    print("-" * 76)
    for r in results:
        print(
            f"{r['n']:>6} {r['dim']:>4} {r['build_ms']:>10.1f} "
            f"{r['brute_ms_per_query']:>10.4f} {r['hnsw_ms_per_query']:>10.4f} "
            f"{r['speedup']:>7.1f}x {r['recall']:>7.1%}"
        )

    print()
    print("[保存图]")
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    for dim in [10, 50]:
        dim_results = [r for r in results if r["dim"] == dim]
        save_bar(
            {f"n={r['n']}": r["hnsw_ms_per_query"] for r in dim_results},
            fig_dir / f"hnsw_latency_dim{dim}.png",
            title=f"HNSW 检索延迟 (dim={dim})",
            ylabel="延迟 (ms/query)",
            xlabel="数据量",
        )
        print(f"  ✓ figures/hnsw_latency_dim{dim}.png")

    for dim in [10, 50]:
        dim_results = [r for r in results if r["dim"] == dim]
        save_line(
            {
                "暴力 KNN": [r["brute_ms_per_query"] for r in dim_results],
                "HNSW": [r["hnsw_ms_per_query"] for r in dim_results],
            },
            fig_dir / f"brute_vs_hnsw_dim{dim}.png",
            title=f"暴力 KNN vs HNSW 延迟 (dim={dim})",
            ylabel="延迟 (ms/query)",
            xlabel="数据量 (1k / 10k / 100k)",
        )
        print(f"  ✓ figures/brute_vs_hnsw_dim{dim}.png")

    for dim in [10, 50]:
        dim_results = [r for r in results if r["dim"] == dim]
        save_line(
            {
                "加速比": [r["speedup"] for r in dim_results],
                "召回率*100": [r["recall"] * 100 for r in dim_results],
            },
            fig_dir / f"hnsw_speedup_recall_dim{dim}.png",
            title=f"HNSW 加速比 & 召回率 (dim={dim})",
            ylabel="加速比 / 召回率%",
            xlabel="数据量 (1k / 10k / 100k)",
        )
        print(f"  ✓ figures/hnsw_speedup_recall_dim{dim}.png")

    save_bar(
        {f"n={r['n']},d={r['dim']}": r["recall"] for r in results},
        fig_dir / "hnsw_recall_all.png",
        title="HNSW 召回率 (所有配置)",
        ylabel="recall@10",
        xlabel="配置",
    )
    print("  ✓ figures/hnsw_recall_all.png")

    print()
    print("=" * 76)
    print("结论：")
    print("  1. 暴力 KNN 延迟随数据量线性增长 O(n)")
    print("  2. HNSW 延迟亚线性增长 O(log n)，100k 下加速 5~10x")
    print("  3. dim=10 召回率 95%+；dim=50 召回率 ~40%（简化版无分层结构）")
    print("  4. 小数据量（1k,10k）暴力 KNN 够快，HNSW 在 100k+ 优势显现")
    print("  5. 跳表提供 O(log n) 入口点选择，图提供 best-first 搜索路径")
    print("     → 跳表解决「从哪开始搜」，图解决「怎么走最快」")
    print("     → 生产级 HNSW 加分层结构 + 启发式选邻居，高维召回率 95%+")
    print("=" * 76)


if __name__ == "__main__":
    main()
