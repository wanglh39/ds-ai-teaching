"""第 10 章 demo：并查集 × AI 优化（层次聚类 HAC 的连通分量维护）

对比并查集 vs BFS 判连通在层次聚类中的性能：
1. 生成随机点集（多个高斯团），用距离阈值连边形成连通分量 = 聚类
2. cluster_dset(points, threshold)：并查集流式合并边，不存邻接表
3. cluster_bfs(points, threshold)：先建邻接表，再 BFS 找连通分量
4. 一次性计算对比：不同点数（1k, 5k, 10k, 50k）下的性能 + 内存
5. 增量合并对比（HAC 核心场景）：逐步增大阈值，并查集只 union 新边，
   BFS 每次重建——体现并查集近 O(1) 合并的增量优势
6. 验证两种方法聚类结果一致
7. 保存图到 figures/

跑法：
    python 10_disjoint_set/python/demo.py
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


class DisjointSet:
    """并查集：父指针 + 路径压缩 + 按秩合并。

    find 用迭代两趟实现路径压缩，避免递归栈溢出。
    union 按秩合并保证树高 O(log n)，配合路径压缩后近 O(1)。
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_sets = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_sets -= 1
        return True

    def labels(self) -> list[int]:
        return [self.find(i) for i in range(len(self.parent))]


def generate_clusters(n: int, n_clusters: int, seed: int = 42) -> np.ndarray:
    """生成 n 个 2D 点，分布在 n_clusters 个高斯团里。

    簇中心在 [0,10]² 内均匀随机，簇内点用 N(center, 0.3) 生成。
    同簇点距离 ~0.6，不同簇距离 ~10/sqrt(k)，阈值 1.0 能连通同簇不连异簇。
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 10, size=(n_clusters, 2))
    pts_per = n // n_clusters
    remainder = n - pts_per * n_clusters
    chunks = []
    for i in range(n_clusters):
        size = pts_per + (1 if i < remainder else 0)
        pts = centers[i] + rng.normal(0, 0.3, size=(size, 2))
        chunks.append(pts)
    return np.vstack(chunks)


def find_edges(points: np.ndarray, threshold: float) -> np.ndarray:
    """用 cKDTree 找所有距离 <= threshold 的点对 (i,j)，i<j。

    比暴力 O(n²) 快得多：平均 O(n log n + E)，E 是边数。
    返回的边集是两种聚类方法的共同输入，公平对比。
    """
    tree = cKDTree(points)
    pairs = tree.query_pairs(threshold, output_type="ndarray")
    return pairs


def cluster_dset(points: np.ndarray, threshold: float) -> tuple[list[int], int, int]:
    """并查集聚类：流式处理边，不存邻接表。

    每条边一次 union，O(E · α(n)) ≈ O(E)。
    内存：parent[] + rank[] = O(n)，不存邻接表。
    """
    n = len(points)
    ds = DisjointSet(n)
    edges = find_edges(points, threshold)
    for u, v in edges:
        ds.union(int(u), int(v))
    return ds.labels(), ds.n_sets, len(edges)


def cluster_bfs(points: np.ndarray, threshold: float) -> tuple[list[int], int, int]:
    """BFS 聚类：先建邻接表，再 BFS 找连通分量。

    建邻接表 O(E)，BFS O(V+E)，总 O(V+E)。
    内存：邻接表 O(V+E)，比并查集多 O(E)。
    """
    n = len(points)
    adj: list[list[int]] = [[] for _ in range(n)]
    edges = find_edges(points, threshold)
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)
    labels = [-1] * n
    n_clusters = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        labels[start] = n_clusters
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if labels[nb] == -1:
                    labels[nb] = n_clusters
                    queue.append(nb)
        n_clusters += 1
    return labels, n_clusters, len(edges)


def find_all_edges_sorted(
    points: np.ndarray, max_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """找所有距离 <= max_threshold 的边，按距离升序排序。

    返回 (edges, dists)：edges[i] = (u,v)，dists[i] = dist(p[u], p[v])。
    层次聚类按距离排序的边流就是 Kruskal 的加入顺序。
    """
    tree = cKDTree(points)
    pairs = tree.query_pairs(max_threshold, output_type="ndarray")
    if len(pairs) == 0:
        return pairs, np.zeros(0)
    diffs = points[pairs[:, 0]] - points[pairs[:, 1]]
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    order = np.argsort(dists)
    return pairs[order], dists[order]


def hac_incremental_dset(
    points: np.ndarray, thresholds: list[float]
) -> tuple[list[int], list[float]]:
    """并查集增量层次聚类：边按距离排序，每个阈值只 union 新增前缀。

    维护同一个 DSet 实例，每个新阈值只处理新增的边（前缀扩展）。
    每个批次 O(ΔE · α(n))，总 O(E · α(n))。
    """
    n = len(points)
    ds = DisjointSet(n)
    edges, dists = find_all_edges_sorted(points, max(thresholds))
    n_sets_history: list[int] = []
    next_edge = 0
    for thr in thresholds:
        idx = int(np.searchsorted(dists, thr, side="right"))
        for i in range(next_edge, idx):
            ds.union(int(edges[i, 0]), int(edges[i, 1]))
        next_edge = idx
        n_sets_history.append(ds.n_sets)
    return n_sets_history, []


def hac_incremental_bfs(
    points: np.ndarray, thresholds: list[float]
) -> tuple[list[int], list[float]]:
    """BFS 增量层次聚类：每个阈值重建邻接表 + BFS。

    无法增量——每次新阈值都要重建整个邻接表并重新 BFS。
    每个批次 O(V+E)，总 O(k·(V+E))，k 是阈值数。
    """
    n = len(points)
    edges, dists = find_all_edges_sorted(points, max(thresholds))
    n_sets_history: list[int] = []
    for thr in thresholds:
        idx = int(np.searchsorted(dists, thr, side="right"))
        adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(idx):
            u, v = int(edges[i, 0]), int(edges[i, 1])
            adj[u].append(v)
            adj[v].append(u)
        labels = [-1] * n
        n_clusters = 0
        for start in range(n):
            if labels[start] != -1:
                continue
            labels[start] = n_clusters
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if labels[nb] == -1:
                        labels[nb] = n_clusters
                        queue.append(nb)
            n_clusters += 1
        n_sets_history.append(n_clusters)
    return n_sets_history, []


def canonicalize_labels(labels: list[int]) -> tuple[int, ...]:
    """把簇标签规范化为可比的形式：按首次出现顺序重编号。"""
    remap: dict[int, int] = {}
    next_id = 0
    result = []
    for lb in labels:
        if lb not in remap:
            remap[lb] = next_id
            next_id += 1
        result.append(remap[lb])
    return tuple(result)


def main() -> None:
    print("=" * 72)
    print("第 10 章 demo：并查集 × AI 优化（层次聚类 HAC 的连通分量维护）")
    print("=" * 72)

    threshold = 1.0

    print(f"\n[1] 小规模正确性验证：n=2000, 20 个高斯团, 阈值={threshold}")
    print("-" * 72)
    pts = generate_clusters(2000, 20, seed=42)
    labels_dset, k_dset, n_edges = cluster_dset(pts, threshold)
    labels_bfs, k_bfs, _ = cluster_bfs(pts, threshold)
    canon_dset = canonicalize_labels(labels_dset)
    canon_bfs = canonicalize_labels(labels_bfs)
    print(f"  点数 = {len(pts)}, 边数 = {n_edges}, 平均度 = {2*n_edges/len(pts):.1f}")
    print(f"  并查集: {k_dset} 个簇")
    print(f"  BFS:    {k_bfs} 个簇")
    match = canon_dset == canon_bfs
    print(f"  两种方法聚类结果一致: {match}")
    assert match, "并查集 vs BFS 聚类结果不一致"
    print(f"  ✓ 结果一致（规范化标签后逐点对比）")

    print("\n[2] 一次性计算性能：不同点数下并查集 vs BFS")
    print("-" * 72)
    sizes = [1000, 5000, 10000, 50000]
    results: list[dict] = []
    for n in sizes:
        n_clust = max(10, n // 100)
        points = generate_clusters(n, n_clust, seed=42)
        with Timer() as t_dset:
            lb_d, k_d, ne = cluster_dset(points, threshold)
        with Timer() as t_bfs:
            lb_b, k_b, _ = cluster_bfs(points, threshold)
        match = canonicalize_labels(lb_d) == canonicalize_labels(lb_b)
        avg_deg = 2 * ne / n
        mem_dset = n * 8 * 2
        mem_bfs = n * 8 + ne * 2 * 8
        print(f"\n  n={n}, 簇数={n_clust}, 边数={ne}, 平均度={avg_deg:.1f}")
        print(f"    并查集: {t_dset.elapsed_ms:8.2f} ms, 内存≈{mem_dset/1024:.0f} KB, {k_d} 簇")
        print(f"    BFS:    {t_bfs.elapsed_ms:8.2f} ms, 内存≈{mem_bfs/1024:.0f} KB, {k_b} 簇")
        print(f"    内存比 (BFS/并查集) = {mem_bfs/mem_dset:.1f}x, 结果一致 = {match}")
        assert match, f"n={n} 时并查集 vs BFS 结果不一致"
        results.append({
            "n": n, "edges": ne, "avg_deg": avg_deg,
            "dset_ms": t_dset.elapsed_ms, "bfs_ms": t_bfs.elapsed_ms,
            "mem_dset": mem_dset, "mem_bfs": mem_bfs,
            "n_clusters": k_d,
        })

    print("\n  一次性计算解读：")
    print("  - 稠密图（平均度高）下两者时间接近，BFS 的 list 迭代效率好")
    print("  - 但 BFS 内存 O(V+E) 远大于并查集 O(V)——并查集不存邻接表")
    print("  - 并查集的真正优势在增量场景（下方 [3]）")

    print("\n[3] 增量合并性能（HAC 核心场景）：逐步增大阈值")
    print("-" * 72)
    n_inc = 10000
    n_clust_inc = 100
    points_inc = generate_clusters(n_inc, n_clust_inc, seed=42)
    thresholds = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    print(f"  n={n_inc}, 簇数={n_clust_inc}, 阈值序列={thresholds}")

    with Timer() as t_dset_inc:
        hist_dset, _ = hac_incremental_dset(points_inc, thresholds)
    with Timer() as t_bfs_inc:
        hist_bfs, _ = hac_incremental_bfs(points_inc, thresholds)

    print(f"\n  并查集增量: {t_dset_inc.elapsed_ms:.2f} ms")
    print(f"  BFS 重建:   {t_bfs_inc.elapsed_ms:.2f} ms")
    print(f"  加速比 = {t_bfs_inc.elapsed_ms/t_dset_inc.elapsed_ms:.2f}x")
    print(f"\n  各阈值下簇数变化:")
    print(f"    {'阈值':>6} {'并查集':>8} {'BFS':>8}")
    for thr, kd, kb in zip(thresholds, hist_dset, hist_bfs):
        assert kd == kb, f"阈值 {thr} 时簇数不一致: dset={kd}, bfs={kb}"
        print(f"    {thr:>6.2f} {kd:>8} {kb:>8}")
    print(f"  ✓ 增量场景下两种方法各阈值簇数一致")

    speedup_inc = t_bfs_inc.elapsed_ms / t_dset_inc.elapsed_ms

    print("\n  增量合并解读：")
    print("  - 并查集维护同一实例，每个新阈值只 union 新增边 O(ΔE·α)")
    print("  - BFS 每个阈值都要重建邻接表 + 重新 BFS O(V+E)")
    print(f"  - {len(thresholds)} 个阈值批次下，并查集快 {speedup_inc:.1f}x")
    print("  - 这正是层次聚类 HAC 的核心场景：合并频繁，近 O(1) 的 union 优势巨大")

    print("\n[4] 不同点数下增量合并的加速比")
    print("-" * 72)
    inc_sizes = [2000, 5000, 10000, 20000]
    inc_speedups: list[float] = []
    for n in inc_sizes:
        n_clust = max(10, n // 100)
        pts_i = generate_clusters(n, n_clust, seed=42)
        with Timer() as td:
            hac_incremental_dset(pts_i, thresholds)
        with Timer() as tb:
            hac_incremental_bfs(pts_i, thresholds)
        sp = tb.elapsed_ms / td.elapsed_ms
        inc_speedups.append(sp)
        print(f"  n={n:>5}: 并查集={td.elapsed_ms:8.2f} ms, BFS={tb.elapsed_ms:8.2f} ms, 加速比={sp:.2f}x")

    print("\n[5] 保存对比图到 figures/")
    print("-" * 72)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {
            "并查集 增量合并": t_dset_inc.elapsed_ms,
            "BFS 重建": t_bfs_inc.elapsed_ms,
        },
        fig_dir / "dset_vs_bfs_incremental_bar.png",
        title=f"增量合并：并查集 vs BFS（n={n_inc}, {len(thresholds)} 阈值批次）",
        ylabel="耗时 (ms)",
        baseline="BFS 重建",
    )
    print(f"  ✓ 保存 figures/dset_vs_bfs_incremental_bar.png")

    save_bar(
        {
            "并查集 内存": float(results[2]["mem_dset"]),
            "BFS 内存": float(results[2]["mem_bfs"]),
        },
        fig_dir / "dset_vs_bfs_memory_bar.png",
        title=f"内存对比：并查集 O(V) vs BFS O(V+E)（n={results[2]['n']}, 边={results[2]['edges']})",
        ylabel="内存 (字节)",
    )
    print(f"  ✓ 保存 figures/dset_vs_bfs_memory_bar.png")

    save_line(
        {
            "并查集 一次性": [r["dset_ms"] for r in results],
            "BFS 一次性": [r["bfs_ms"] for r in results],
        },
        fig_dir / "dset_vs_bfs_oneshot.png",
        title="一次性连通分量计算耗时 vs 点数",
        ylabel="耗时 (ms)",
        xlabel="点数 (1k / 5k / 10k / 50k)",
    )
    print(f"  ✓ 保存 figures/dset_vs_bfs_oneshot.png")

    save_line(
        {
            "增量合并加速比": inc_speedups,
        },
        fig_dir / "dset_incremental_speedup.png",
        title="增量合并加速比 vs 点数（10 阈值批次）",
        ylabel="加速比 (x)",
        xlabel="点数 (2k / 5k / 10k / 20k)",
    )
    print(f"  ✓ 保存 figures/dset_incremental_speedup.png")

    save_line(
        {
            "簇数": [float(h) for h in hist_dset],
        },
        fig_dir / "hac_clusters_by_threshold.png",
        title="HAC 簇数随阈值变化（n=10000）",
        ylabel="簇数",
        xlabel="阈值序号 (0.3 → 2.0)",
    )
    print(f"  ✓ 保存 figures/hac_clusters_by_threshold.png")

    print("\n" + "=" * 72)
    print("结论：层次聚类的连通分量维护，并查集在时间和内存上都优于 BFS。")
    print("一次性计算：并查集内存 O(V) vs BFS O(V+E)，不存邻接表。")
    print(f"增量合并（HAC 核心）：并查集快 {speedup_inc:.1f}x——只 union 新边 O(α)，BFS 要重建 O(V+E)。")
    print("核心原因：路径压缩 + 按秩合并让 union/find 近 O(1)（α(n)≈4），合并频繁时优势巨大。")
    print("=" * 72)


if __name__ == "__main__":
    main()
