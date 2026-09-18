"""第 08 章 demo：KD-Tree KNN vs 暴力 KNN，不同维度对比。

展示「KD-Tree」在 AI 里的核心应用：KNN（K-Nearest Neighbors）加速。
1. 纯 Python + numpy 实现通用 k 维 KD-Tree（按轴交替切分 + 递归剪枝搜索）
2. 两种 KNN 实现对比：
   - brute_knn：暴力算所有点距离取前 K，O(n)
   - kd_knn：KD-Tree 递归搜索 + 剪枝，低维 O(log n)，高维退化
3. 验证两种方法结果一致（返回的 K 个最近邻索引集合相同）
4. 对比不同维度（2, 10, 50, 100）和不同数据量（1k, 10k）下的性能
5. 展示「curse of dimensionality」：高维下 KD-Tree 剪枝失效，退化到接近暴力

跑法：
    python 08_kd_tree/python/demo.py
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line, save_scatter  # noqa: E402


class KDNode:
    __slots__ = ("point", "index", "axis", "left", "right")

    def __init__(self, point: np.ndarray, index: int, axis: int) -> None:
        self.point = point
        self.index = index
        self.axis = axis
        self.left: KDNode | None = None
        self.right: KDNode | None = None


class KDTree:
    """通用 k 维 KD-Tree，按轴交替切分，递归剪枝搜索 KNN。

    构建：每层选 depth % k 轴，按该轴排序取 median 切分，保证树高 O(log n)。
    搜索：先递归搜近侧子树，再用分裂超平面距离剪枝远侧。
    """

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.n, self.dim = self.points.shape
        indices = np.arange(self.n)
        self.root = self._build(indices, 0)

    def _build(self, indices: np.ndarray, depth: int) -> KDNode | None:
        if indices.size == 0:
            return None
        axis = depth % self.dim
        order = np.argsort(self.points[indices, axis], kind="quicksort")
        indices = indices[order]
        mid = indices.size // 2
        node = KDNode(self.points[indices[mid]], int(indices[mid]), axis)
        node.left = self._build(indices[:mid], depth + 1)
        node.right = self._build(indices[mid + 1:], depth + 1)
        return node

    def knn(self, query: np.ndarray, k: int) -> list[tuple[float, int]]:
        """返回前 k 个最近邻 [(dist_sq, index), ...]，按距离升序。"""
        heap: list[tuple[float, int]] = []
        self._knn_rec(self.root, query, k, heap)
        return sorted((-nd, idx) for nd, idx in heap)

    def _knn_rec(self, node: KDNode | None, query: np.ndarray,
                 k: int, heap: list[tuple[float, int]]) -> None:
        if node is None:
            return
        diff = node.point - query
        d = float(diff @ diff)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.index))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.index))
        axis = node.axis
        delta = float(query[axis] - node.point[axis])
        if delta < 0.0:
            near, far = node.left, node.right
        else:
            near, far = node.right, node.left
        self._knn_rec(near, query, k, heap)
        worst = -heap[0][0] if len(heap) == k else float("inf")
        if delta * delta < worst:
            self._knn_rec(far, query, k, heap)


def brute_knn(points: np.ndarray, query: np.ndarray, k: int) -> list[tuple[float, int]]:
    """暴力 KNN：算所有点距离，取前 k 个。O(n) 时间。"""
    diff = points - query
    dists = np.einsum("ij,ij->i", diff, diff)
    kk = min(k, points.shape[0])
    idx = np.argpartition(dists, kk - 1)[:kk]
    order = np.argsort(dists[idx])
    return [(float(dists[i]), int(i)) for i in idx[order]]


def kd_knn(tree: KDTree, query: np.ndarray, k: int) -> list[tuple[float, int]]:
    return tree.knn(query, k)


def knn_indices_match(a: list[tuple[float, int]],
                      b: list[tuple[float, int]]) -> bool:
    """验证两种 KNN 结果一致：比较索引集合（距离可能有浮点误差）。"""
    return set(idx for _, idx in a) == set(idx for _, idx in b)


def bench_knn(dim: int, n: int, k: int = 5, n_queries: int = 200,
              repeat: int = 5) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(42)
    points = rng.standard_normal((n, dim))
    queries = rng.standard_normal((n_queries, dim))
    tree = KDTree(points)

    ok = True
    for q in queries[:20]:
        r1 = kd_knn(tree, q, k)
        r2 = brute_knn(points, q, k)
        if not knn_indices_match(r1, r2):
            ok = False
            break
    if not ok:
        print(f"  [警告] dim={dim} n={n} KNN 结果不一致！")

    def run_kd() -> None:
        for q in queries:
            kd_knn(tree, q, k)

    def run_brute() -> None:
        for q in queries:
            brute_knn(points, q, k)

    return compare({"KD-Tree": run_kd, "brute": run_brute}, repeat=repeat, warmup=1)


def main() -> None:
    print("=" * 72)
    print("第 08 章 demo：KD-Tree KNN vs 暴力 KNN，不同维度对比")
    print("=" * 72)

    print("\n[1] 正确性验证：KD-Tree KNN 与暴力 KNN 结果一致")
    print("-" * 72)
    rng = np.random.default_rng(0)
    for dim in [2, 10, 50, 100]:
        pts = rng.standard_normal((500, dim))
        tree = KDTree(pts)
        q = rng.standard_normal(dim)
        r_kd = kd_knn(tree, q, 5)
        r_br = brute_knn(pts, q, 5)
        match = knn_indices_match(r_kd, r_br)
        print(f"  dim={dim:>3}: KD-Tree 索引={sorted(i for _, i in r_kd)}, "
              f"brute 索引={sorted(i for _, i in r_br)}, 一致={match}")
        assert match, f"dim={dim} KNN 不一致"

    print("\n[2] 2D 可视化：数据点 + 查询点 + 5 个最近邻")
    print("-" * 72)
    rng2 = np.random.default_rng(7)
    pts2 = rng2.standard_normal((300, 2))
    tree2 = KDTree(pts2)
    q2 = rng2.standard_normal(2)
    nn5 = kd_knn(tree2, q2, 5)
    print(f"  查询点 = ({q2[0]:.3f}, {q2[1]:.3f})")
    print(f"  5 最近邻索引 = {[i for _, i in nn5]}")
    print(f"  5 最近邻距离 = {[round(d, 3) for d, _ in nn5]}")
    fig_dir = Path(__file__).resolve().parent.parent / "figures"
    scatter_points = [(float(p[0]), float(p[1])) for p in pts2]
    extra = [(float(q2[0]), float(q2[1]), "query")]
    for rank, (d, idx) in enumerate(nn5, 1):
        extra.append((float(pts2[idx, 0]), float(pts2[idx, 1]), f"NN{rank}"))
    save_scatter(
        scatter_points,
        fig_dir / "kd_2d_knn_scatter.png",
        title="2D KD-Tree KNN：数据点 + 查询点 + 5 最近邻",
        xlabel="x", ylabel="y",
        extra=extra,
    )
    print(f"  ✓ 保存 figures/kd_2d_knn_scatter.png")

    print("\n[3] 性能对比：不同维度 × 不同数据量，K=5")
    print("-" * 72)
    dims = [2, 10, 50, 100]
    n_points_list = [1000, 10000]
    all_results: list[tuple[int, int, dict[str, dict[str, float]]]] = []
    for n in n_points_list:
        for dim in dims:
            print(f"\n  n={n}, dim={dim}, K=5, queries=200")
            res = bench_knn(dim, n, k=5, n_queries=200, repeat=5)
            all_results.append((dim, n, res))
            print(format_table(res, baseline="brute"))
            sp = res["brute"]["mean_ms"] / res["KD-Tree"]["mean_ms"]
            tag = "KD-Tree 加速" if sp > 1 else "KD-Tree 反而更慢（高维退化）"
            print(f"    → brute/KD-Tree = {sp:.2f}x  ({tag})")

    print("\n[4] 关键数字：n=10000，维度从 2 到 100")
    print("-" * 72)
    print(f"  {'dim':>4} {'KD-Tree(ms)':>14} {'brute(ms)':>14} {'加速比':>10} {'剪枝有效?':>12}")
    print("  " + "-" * 58)
    speedups_by_dim: dict[str, list[float]] = {"KD-Tree": [], "brute": []}
    dim_labels: list[str] = []
    for dim, n, res in all_results:
        if n != 10000:
            continue
        t_kd = res["KD-Tree"]["mean_ms"]
        t_br = res["brute"]["mean_ms"]
        sp = t_br / t_kd
        effective = "是" if sp > 1.5 else ("接近" if sp > 0.8 else "否（退化）")
        print(f"  {dim:>4} {t_kd:>14.4f} {t_br:>14.4f} {sp:>9.2f}x {effective:>12}")
        speedups_by_dim["KD-Tree"].append(t_kd)
        speedups_by_dim["brute"].append(t_br)
        dim_labels.append(str(dim))

    print("\n  解读：")
    print("  - dim=2：KD-Tree 剪枝极有效，比暴力快几十倍（O(log n) vs O(n)）")
    print("  - dim=10：剪枝仍有效但优势缩小")
    print("  - dim=50/100：剪枝几乎失效，KD-Tree 退化到接近暴力甚至更慢")
    print("    （递归 + 剪枝判断的常数开销 > 暴力 numpy 向量化的优势）")
    print("  - 这就是「curse of dimensionality」：高维下所有点距离趋近，")
    print("    分裂超平面剪枝条件几乎总成立，必须搜远侧，退化成全遍历")

    print("\n[5] 保存性能对比图")
    print("-" * 72)
    save_line(
        speedups_by_dim,
        fig_dir / "kd_vs_brute_by_dim.png",
        title="KD-Tree vs 暴力 KNN 耗时 vs 维度（n=10000, K=5）",
        ylabel="耗时 (ms, 200 queries)",
        xlabel="维度 (2 / 10 / 50 / 100)",
    )
    print(f"  ✓ 保存 figures/kd_vs_brute_by_dim.png")

    res_2d = next(r for d, n, r in all_results if d == 2 and n == 10000)
    res_100d = next(r for d, n, r in all_results if d == 100 and n == 10000)
    save_bar(
        {
            "KD-Tree 2D": res_2d["KD-Tree"]["mean_ms"],
            "brute 2D": res_2d["brute"]["mean_ms"],
            "KD-Tree 100D": res_100d["KD-Tree"]["mean_ms"],
            "brute 100D": res_100d["brute"]["mean_ms"],
        },
        fig_dir / "kd_2d_vs_100d_bar.png",
        title="2D vs 100D：KD-Tree 低维加速、高维退化（n=10000, K=5）",
        ylabel="耗时 (ms, 200 queries)",
    )
    print(f"  ✓ 保存 figures/kd_2d_vs_100d_bar.png")

    print("\n[6] 访问节点数对比：直观看剪枝效果")
    print("-" * 72)
    rng = np.random.default_rng(99)
    for dim in [2, 10, 50, 100]:
        pts = rng.standard_normal((10000, dim))
        tree = KDTree(pts)
        q = rng.standard_normal(dim)
        visited = [0]
        _count_visits(tree.root, q, 5, [], visited)
        print(f"  dim={dim:>3}: 访问 {visited[0]:>6} / 10000 节点 "
              f"({visited[0] / 100:.1f}%)")

    print("\n" + "=" * 72)
    print("结论：KD-Tree 是「二分搜索的 k 维推广」，低维高效、高维退化。")
    print("生产级高维 KNN 用 HNSW（第 12 章）或 FAISS，而非 KD-Tree。")
    print("=" * 72)


def _count_visits(node: KDNode | None, query: np.ndarray, k: int,
                  heap: list[tuple[float, int]], visited: list[int]) -> None:
    if node is None:
        return
    visited[0] += 1
    diff = node.point - query
    d = float(diff @ diff)
    if len(heap) < k:
        heapq.heappush(heap, (-d, node.index))
    elif d < -heap[0][0]:
        heapq.heapreplace(heap, (-d, node.index))
    axis = node.axis
    delta = float(query[axis] - node.point[axis])
    if delta < 0.0:
        near, far = node.left, node.right
    else:
        near, far = node.right, node.left
    _count_visits(near, query, k, heap, visited)
    worst = -heap[0][0] if len(heap) == k else float("inf")
    if delta * delta < worst:
        _count_visits(far, query, k, heap, visited)


if __name__ == "__main__":
    main()