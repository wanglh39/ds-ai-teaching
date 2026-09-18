"""第 09 章 demo：图数据结构 × AI 优化（PageRank + GNN 消息传递）

对比邻接表 vs 邻接矩阵在稀疏图上的内存和性能：
1. 构建随机稀疏图（n=1000, 边数~5000）
2. PageRank 两种实现：
   - pagerank_adj：邻接表版，只遍历实际存在的边 O(V+E)
   - pagerank_mat：邻接矩阵版，矩阵-向量乘 O(V²)
3. GNN 消息传递两种实现：
   - gnn_message_passing_adj：邻接表版，按邻居聚合
   - gnn_message_passing_mat：邻接矩阵版，矩阵乘
4. 三层对比：邻接表（dict）vs 邻接矩阵（dense）vs 稀疏矩阵（CSR）
5. 内存占用 + 性能 + 不同稀疏度趋势
6. 保存对比图到 figures/

跑法：
    python 09_graph/python/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


def build_sparse_graph(
    n: int, n_edges: int, seed: int = 42
) -> tuple[dict[int, np.ndarray], np.ndarray, sp.csr_matrix]:
    """构建随机稀疏图，返回 (邻接表, 邻接矩阵, CSR 稀疏矩阵)。

    无向图，避免自环和重边。三种表示同一张图，用于对比。
    """
    rng = np.random.default_rng(seed)
    adj_list: dict[int, list[int]] = {i: [] for i in range(n)}
    adj_mat = np.zeros((n, n), dtype=np.float64)
    added = 0
    while added < n_edges:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u != v and adj_mat[u, v] == 0.0:
            adj_list[u].append(v)
            adj_list[v].append(u)
            adj_mat[u, v] = 1.0
            adj_mat[v, u] = 1.0
            added += 1
    adj_array = {i: np.array(adj_list[i], dtype=np.int64) for i in range(n)}
    adj_csr = sp.csr_matrix(adj_mat)
    return adj_array, adj_mat, adj_csr


def pagerank_adj(
    adj_array: dict[int, np.ndarray], n: int, iters: int = 100,
    damping: float = 0.85,
) -> np.ndarray:
    """邻接表版 PageRank。沿实际边推送分数，每轮 O(V+E)。

    PR(v) = (1-d)/n + d * sum(PR(u)/deg(u) for u in neighbors(v))

    实现用 push 模型：对每个顶点 u，把 d*PR(u)/deg(u) 推给所有邻居。
    内层 new_pr[nb] += share 用 numpy fancy index 向量化。
    """
    pr = np.full(n, 1.0 / n)
    out_degree = np.array([adj_array[i].size for i in range(n)], dtype=np.float64)
    teleport = (1.0 - damping) / n
    for _ in range(iters):
        new_pr = np.full(n, teleport)
        for u in range(n):
            deg = out_degree[u]
            if deg > 0:
                new_pr[adj_array[u]] += damping * pr[u] / deg
        pr = new_pr
    return pr


def pagerank_mat(
    adj_mat: np.ndarray, n: int, iters: int = 100,
    damping: float = 0.85,
) -> np.ndarray:
    """邻接矩阵版 PageRank。矩阵-向量乘，每轮 O(V²)。

    M = D^-1 * A（行归一化），PR = (1-d)/n * 1 + d * M^T * PR
    用 BLAS gemv，向量化极快但扫整个矩阵（包括 0）。
    """
    pr = np.full(n, 1.0 / n)
    out_degree = adj_mat.sum(axis=1)
    inv_deg = np.zeros_like(out_degree)
    mask = out_degree > 0
    inv_deg[mask] = 1.0 / out_degree[mask]
    M = adj_mat * inv_deg[:, None]
    MT = M.T
    teleport = (1.0 - damping) / n
    for _ in range(iters):
        pr = teleport + damping * (MT @ pr)
    return pr


def pagerank_csr(
    adj_csr: sp.csr_matrix, n: int, iters: int = 100,
    damping: float = 0.85,
) -> np.ndarray:
    """CSR 稀疏矩阵版 PageRank。稀疏矩阵-向量乘，每轮 O(V+E)。

    生产级做法：scipy.sparse 用 CSR 格式 = 邻接表的紧凑数组存储，
    稀疏矩阵乘用 C 实现，既省内存又快。
    """
    pr = np.full(n, 1.0 / n)
    out_degree = np.asarray(adj_csr.sum(axis=1)).ravel()
    inv_deg = np.zeros_like(out_degree)
    mask = out_degree > 0
    inv_deg[mask] = 1.0 / out_degree[mask]
    M = adj_csr.multiply(inv_deg[:, None]).tocsr()
    MT = M.T.tocsr()
    teleport = (1.0 - damping) / n
    for _ in range(iters):
        pr = teleport + damping * (MT @ pr)
    return pr


def gnn_message_passing_adj(
    adj_array: dict[int, np.ndarray], features: np.ndarray, n: int,
) -> np.ndarray:
    """邻接表版 GNN 消息传递：聚合邻居特征。

    h_v' = h_v + sum(h_u for u in neighbors(v))

    沿边遍历，只碰实际存在的边。内层 features[nb].sum(axis=0) 向量化。
    """
    new_features = features.copy()
    for v in range(n):
        nb = adj_array[v]
        if nb.size:
            new_features[v] += features[nb].sum(axis=0)
    return new_features


def gnn_message_passing_mat(adj_mat: np.ndarray, features: np.ndarray) -> np.ndarray:
    """邻接矩阵版 GNN 消息传递：矩阵乘。

    h' = h + A @ h

    BLAS gemm，向量化极快但扫整个矩阵（包括 0）。
    """
    return features + adj_mat @ features


def gnn_message_passing_csr(adj_csr: sp.csr_matrix, features: np.ndarray) -> np.ndarray:
    """CSR 稀疏矩阵版 GNN 消息传递：稀疏矩阵-矩阵乘。

    h' = h + A @ h，但 A 是 CSR 稀疏格式，只碰非零元。
    """
    return features + adj_csr @ features


def memory_usage_adj(adj_array: dict[int, np.ndarray], n: int) -> int:
    """邻接表内存占用估算（字节）。

    dict overhead + 每顶点一个 ndarray（header + 数据）。
    每条无向边存两次（u 在 v 的表里，v 在 u 的表里）。
    """
    dict_overhead = sys.getsizeof(adj_array) + n * 64
    edges_total = sum(arr.size for arr in adj_array.values())
    ndarray_overhead = n * 112
    edge_storage = edges_total * 8
    return dict_overhead + ndarray_overhead + edge_storage


def memory_usage_mat(n: int) -> int:
    """邻接矩阵内存占用（字节）：n*n 个 float64。"""
    return n * n * 8


def memory_usage_csr(adj_csr: sp.csr_matrix) -> int:
    """CSR 稀疏矩阵内存占用（字节）：data + indices + indptr。"""
    return (
        adj_csr.data.nbytes + adj_csr.indices.nbytes + adj_csr.indptr.nbytes
    )


def main() -> None:
    print("=" * 72)
    print("第 09 章 demo：图数据结构 × AI 优化（PageRank + GNN 消息传递）")
    print("=" * 72)

    n = 1000
    n_edges = 5000
    in_dim = 32

    print(f"\n[1] 构建随机稀疏图：n={n}, 边数={n_edges}, 特征维度={in_dim}")
    print("-" * 72)
    adj_array, adj_mat, adj_csr = build_sparse_graph(n, n_edges, seed=42)
    actual_edges = int(adj_mat.sum() / 2)
    avg_degree = 2.0 * actual_edges / n
    density = 2.0 * actual_edges / (n * (n - 1))
    max_edges = n * (n - 1) // 2
    print(f"  实际边数 = {actual_edges}")
    print(f"  平均度 = {avg_degree:.2f}")
    print(f"  图密度 = {density:.6f}（稀疏：远小于 1）")
    print(f"  理论边数上限 = {max_edges}（完全图）")
    print(f"  边数/上限 = {actual_edges/max_edges:.6f}（仅存 {actual_edges/max_edges*100:.3f}%）")

    print("\n[2] 内存对比：邻接表 vs 邻接矩阵 vs CSR 稀疏矩阵")
    print("-" * 72)
    mem_adj = memory_usage_adj(adj_array, n)
    mem_mat = memory_usage_mat(n)
    mem_csr = memory_usage_csr(adj_csr)
    print(f"  邻接表内存     ≈ {mem_adj:>10} 字节 ({mem_adj/1024:.1f} KB)")
    print(f"  邻接矩阵内存   = {mem_mat:>10} 字节 ({mem_mat/1024:.1f} KB)")
    print(f"  CSR 稀疏矩阵   = {mem_csr:>10} 字节 ({mem_csr/1024:.1f} KB)")
    print(f"  比值（矩阵/表） = {mem_mat/mem_adj:.2f}x")
    print(f"  比值（矩阵/CSR）= {mem_mat/mem_csr:.2f}x")
    print(f"  → 稀疏图（密度={density:.6f}）下邻接表/CSR 省 {mem_mat/mem_adj:.1f}x 内存")

    print("\n[3] PageRank 正确性验证：三种实现结果一致")
    print("-" * 72)
    pr_adj = pagerank_adj(adj_array, n, iters=100)
    pr_mat = pagerank_mat(adj_mat, n, iters=100)
    pr_csr = pagerank_csr(adj_csr, n, iters=100)
    diff_am = np.abs(pr_adj - pr_mat).max()
    diff_ac = np.abs(pr_adj - pr_csr).max()
    print(f"  PR(邻接表) sum = {pr_adj.sum():.6f}")
    print(f"  PR(矩阵)   sum = {pr_mat.sum():.6f}")
    print(f"  PR(CSR)    sum = {pr_csr.sum():.6f}")
    print(f"  max|PR_adj - PR_mat| = {diff_am:.2e}")
    print(f"  max|PR_adj - PR_csr| = {diff_ac:.2e}")
    assert diff_am < 1e-10, f"PageRank 邻接表 vs 矩阵不一致，diff={diff_am}"
    assert diff_ac < 1e-10, f"PageRank 邻接表 vs CSR 不一致，diff={diff_ac}"
    top5_adj = np.argsort(pr_adj)[-5:][::-1]
    print(f"  Top-5 节点（邻接表）: {top5_adj.tolist()}")
    print(f"  Top-5 PR 值: {[round(float(pr_adj[i]), 6) for i in top5_adj]}")
    print(f"  ✓ 三种实现结果一致")

    print("\n[4] PageRank 性能对比：邻接表 vs 邻接矩阵 vs CSR")
    print("-" * 72)
    iters = 50
    res_pr = compare(
        {
            "邻接表 PageRank": lambda: pagerank_adj(adj_array, n, iters=iters),
            "邻接矩阵 PageRank": lambda: pagerank_mat(adj_mat, n, iters=iters),
            "CSR PageRank": lambda: pagerank_csr(adj_csr, n, iters=iters),
        },
        repeat=5, warmup=1,
    )
    print(format_table(res_pr, baseline="邻接矩阵 PageRank"))
    sp_pr_adj = res_pr["邻接矩阵 PageRank"]["mean_ms"] / res_pr["邻接表 PageRank"]["mean_ms"]
    sp_pr_csr = res_pr["邻接矩阵 PageRank"]["mean_ms"] / res_pr["CSR PageRank"]["mean_ms"]
    print(f"  → 邻接表 vs 矩阵: {sp_pr_adj:.2f}x {'(邻接表快)' if sp_pr_adj > 1 else '(矩阵快，BLAS 向量化优势)'}")
    print(f"  → CSR vs 矩阵: {sp_pr_csr:.2f}x {'(CSR 快)' if sp_pr_csr > 1 else '(矩阵快)'}")

    print("\n[5] GNN 消息传递正确性验证")
    print("-" * 72)
    rng = np.random.default_rng(7)
    features = rng.standard_normal((n, in_dim))
    h_adj = gnn_message_passing_adj(adj_array, features, n)
    h_mat = gnn_message_passing_mat(adj_mat, features)
    h_csr = gnn_message_passing_csr(adj_csr, features)
    diff_gnn_am = np.abs(h_adj - h_mat).max()
    diff_gnn_ac = np.abs(h_adj - h_csr).max()
    print(f"  特征矩阵 shape = {features.shape}")
    print(f"  max|h_adj - h_mat| = {diff_gnn_am:.2e}")
    print(f"  max|h_adj - h_csr| = {diff_gnn_ac:.2e}")
    assert diff_gnn_am < 1e-10, f"GNN 邻接表 vs 矩阵不一致，diff={diff_gnn_am}"
    assert diff_gnn_ac < 1e-10, f"GNN 邻接表 vs CSR 不一致，diff={diff_gnn_ac}"
    print(f"  ✓ 三种实现结果一致")

    print("\n[6] GNN 消息传递性能对比")
    print("-" * 72)
    res_gnn = compare(
        {
            "邻接表 GNN": lambda: gnn_message_passing_adj(adj_array, features, n),
            "邻接矩阵 GNN": lambda: gnn_message_passing_mat(adj_mat, features),
            "CSR GNN": lambda: gnn_message_passing_csr(adj_csr, features),
        },
        repeat=5, warmup=1,
    )
    print(format_table(res_gnn, baseline="邻接矩阵 GNN"))
    sp_gnn_adj = res_gnn["邻接矩阵 GNN"]["mean_ms"] / res_gnn["邻接表 GNN"]["mean_ms"]
    sp_gnn_csr = res_gnn["邻接矩阵 GNN"]["mean_ms"] / res_gnn["CSR GNN"]["mean_ms"]
    print(f"  → 邻接表 vs 矩阵: {sp_gnn_adj:.2f}x")
    print(f"  → CSR vs 矩阵: {sp_gnn_csr:.2f}x")

    print("\n[7] 不同稀疏度下的性能趋势（n=1000, 边数从 1k 到 50k）")
    print("-" * 72)
    edge_counts = [1000, 5000, 20000, 50000, 100000]
    pr_adj_times: list[float] = []
    pr_mat_times: list[float] = []
    pr_csr_times: list[float] = []
    gnn_adj_times: list[float] = []
    gnn_mat_times: list[float] = []
    gnn_csr_times: list[float] = []
    densities: list[float] = []
    for ne in edge_counts:
        al, am, ac = build_sparse_graph(n, ne, seed=42)
        d = 2.0 * ne / (n * (n - 1))
        densities.append(d)
        r_pr = compare(
            {
                "adj": lambda: pagerank_adj(al, n, iters=20),
                "mat": lambda: pagerank_mat(am, n, iters=20),
                "csr": lambda: pagerank_csr(ac, n, iters=20),
            },
            repeat=3, warmup=1,
        )
        r_gnn = compare(
            {
                "adj": lambda: gnn_message_passing_adj(al, features, n),
                "mat": lambda: gnn_message_passing_mat(am, features),
                "csr": lambda: gnn_message_passing_csr(ac, features),
            },
            repeat=3, warmup=1,
        )
        pr_adj_times.append(r_pr["adj"]["mean_ms"])
        pr_mat_times.append(r_pr["mat"]["mean_ms"])
        pr_csr_times.append(r_pr["csr"]["mean_ms"])
        gnn_adj_times.append(r_gnn["adj"]["mean_ms"])
        gnn_mat_times.append(r_gnn["mat"]["mean_ms"])
        gnn_csr_times.append(r_gnn["csr"]["mean_ms"])
        print(f"  边={ne:>6}, 密度={d:.5f}: "
              f"PR adj={r_pr['adj']['mean_ms']:6.2f} mat={r_pr['mat']['mean_ms']:6.2f} csr={r_pr['csr']['mean_ms']:6.2f} | "
              f"GNN adj={r_gnn['adj']['mean_ms']:6.2f} mat={r_gnn['mat']['mean_ms']:6.2f} csr={r_gnn['csr']['mean_ms']:6.2f}")

    print("\n  解读：")
    print("  - 邻接表/CSR 耗时随边数线性增长 O(V+E)，稀疏时极快")
    print("  - 邻接矩阵耗时几乎不随边数变 O(V²)，因为不管有没有边都扫整个矩阵")
    print("  - 图越稀疏，邻接表/CSR 优势越大；图越稠密，矩阵（BLAS 向量化）反超")
    print("  - CSR = 邻接表的紧凑数组存储 + C 层稀疏运算，生产级首选")

    print("\n[8] 保存对比图到 figures/")
    print("-" * 72)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {
            "邻接表 PageRank": res_pr["邻接表 PageRank"]["mean_ms"],
            "邻接矩阵 PageRank": res_pr["邻接矩阵 PageRank"]["mean_ms"],
            "CSR PageRank": res_pr["CSR PageRank"]["mean_ms"],
        },
        fig_dir / "pagerank_adj_vs_mat_bar.png",
        title=f"PageRank 三种实现（n={n}, 边={n_edges}, iters={iters}）",
        ylabel="耗时 (ms)",
        baseline="邻接矩阵 PageRank",
    )
    print(f"  ✓ 保存 figures/pagerank_adj_vs_mat_bar.png")

    save_bar(
        {
            "邻接表 GNN": res_gnn["邻接表 GNN"]["mean_ms"],
            "邻接矩阵 GNN": res_gnn["邻接矩阵 GNN"]["mean_ms"],
            "CSR GNN": res_gnn["CSR GNN"]["mean_ms"],
        },
        fig_dir / "gnn_adj_vs_mat_bar.png",
        title=f"GNN 消息传递三种实现（n={n}, 边={n_edges}, dim={in_dim}）",
        ylabel="耗时 (ms)",
        baseline="邻接矩阵 GNN",
    )
    print(f"  ✓ 保存 figures/gnn_adj_vs_mat_bar.png")

    save_line(
        {
            "PageRank 邻接表": pr_adj_times,
            "PageRank 邻接矩阵": pr_mat_times,
            "PageRank CSR": pr_csr_times,
        },
        fig_dir / "pagerank_by_sparsity.png",
        title="PageRank 耗时 vs 边数（n=1000, 20 iters）",
        ylabel="耗时 (ms)",
        xlabel="边数 (1k / 5k / 20k / 50k / 100k)",
    )
    print(f"  ✓ 保存 figures/pagerank_by_sparsity.png")

    save_line(
        {
            "GNN 邻接表": gnn_adj_times,
            "GNN 邻接矩阵": gnn_mat_times,
            "GNN CSR": gnn_csr_times,
        },
        fig_dir / "gnn_by_sparsity.png",
        title="GNN 消息传递耗时 vs 边数（n=1000, dim=32）",
        ylabel="耗时 (ms)",
        xlabel="边数 (1k / 5k / 20k / 50k / 100k)",
    )
    print(f"  ✓ 保存 figures/gnn_by_sparsity.png")

    save_bar(
        {
            "邻接表": float(mem_adj),
            "邻接矩阵": float(mem_mat),
            "CSR 稀疏矩阵": float(mem_csr),
        },
        fig_dir / "memory_adj_vs_mat_bar.png",
        title=f"内存占用三种表示（n={n}, 边={n_edges}）",
        ylabel="内存 (字节)",
    )
    print(f"  ✓ 保存 figures/memory_adj_vs_mat_bar.png")

    print("\n" + "=" * 72)
    print("结论：稀疏图必须用邻接表——内存省 O(V²/E) 倍，性能只遍历实际边 O(V+E)。")
    print("PageRank 和 GNN 消息传递本质都是「沿边聚合」，邻接表让聚合只碰真实存在的边。")
    print("生产级 GNN 框架（PyG、DGL）底层都用 CSR 稀疏矩阵 = 邻接表的紧凑数组存储。")
    print("=" * 72)


if __name__ == "__main__":
    main()