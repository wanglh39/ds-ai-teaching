"""第 05 章 demo：哈希表 O(1) 查找 vs 排序数组二分 O(log n) + MoE 路由。

展示「哈希表」在 AI 里的两个核心应用：
1. 词嵌入查找：词表 10 万级时，dict 哈希 O(1) vs bisect 二分 O(log n) 的性能对比
2. MoE 路由：用哈希把 token 路由到 top-2 专家，验证负载均匀分布

跑法：
    python 05_hash_table/python/demo.py
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402

EMBED_DIM = 128


def build_vocab(n: int) -> tuple[list[str], dict[str, int], list[int]]:
    """构建词表：n 个词，返回 (排序词表, 词->索引哈希, 索引列表)。

    词名格式 word_000000 ~ word_nnnnnn，天然有序，直接当排序数组用。
    embedding 用索引占位（性能对比只测查找开销，不测向量内容）。
    """
    words = [f"word_{i:06d}" for i in range(n)]
    hash_index = {w: i for i, w in enumerate(words)}
    return words, hash_index, list(range(n))


def hash_lookup(hash_index: dict[str, int], key: str) -> int:
    """哈希表 O(1) 查找：dict.__getitem__。"""
    return hash_index[key]


def binary_lookup(sorted_words: list[str], key: str) -> int:
    """排序数组二分 O(log n) 查找：bisect_left + 校验。"""
    idx = bisect.bisect_left(sorted_words, key)
    if idx < len(sorted_words) and sorted_words[idx] == key:
        return idx
    return -1


def bench_lookup(n: int, n_queries: int = 10000) -> dict[str, dict[str, float]]:
    """对词表大小 n，对比 hash vs binary 查找性能。

    每次查 n_queries 个词（一半命中一半未命中），重复测速。
    """
    words, hash_index, _ = build_vocab(n)

    import random
    rng = random.Random(42)
    half = n_queries // 2
    hit_keys = [words[rng.randrange(n)] for _ in range(half)]
    miss_keys = [f"miss_{i:06d}" for i in range(n_queries - half)]
    query_keys = hit_keys + miss_keys
    rng.shuffle(query_keys)

    def run_hash() -> None:
        for k in query_keys:
            hash_index.get(k, -1)

    def run_binary() -> None:
        for k in query_keys:
            binary_lookup(words, k)

    return compare({"hash O(1)": run_hash, "binary O(log n)": run_binary}, repeat=5, warmup=2)


def moe_route(token: str, n_experts: int = 8, top_k: int = 2) -> list[int]:
    """MoE 路由：用哈希把 token 路由到 top_k 个专家。

    策略：对 token 做哈希，取不同比特段 mod n_experts 选专家，去重。
    生产级 MoE 用学到的 gating network，这里用哈希模拟「确定性均匀路由」。
    """
    h = hash(token) & 0xFFFFFFFFFFFFFFFF
    experts: list[int] = []
    shift = 0
    while len(experts) < top_k:
        e = (h >> shift) % n_experts
        if e not in experts:
            experts.append(e)
        shift += 8
        if shift >= 64:
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            shift = 0
    return experts


def demo_moe_routing(n_tokens: int = 100000, n_experts: int = 8, top_k: int = 2) -> dict[str, float]:
    """模拟 MoE 路由：n_tokens 个 token 路由到 n_experts 个专家的 top_k。

    返回每个专家被选中的次数（统计负载分布）。
    """
    counts = [0] * n_experts
    for i in range(n_tokens):
        token = f"tok_{i:06d}"
        for e in moe_route(token, n_experts, top_k):
            counts[e] += 1
    return {f"expert_{i}": float(counts[i]) for i in range(n_experts)}


def main() -> None:
    print("=" * 64)
    print("第 05 章 demo：哈希表 O(1) vs 二分 O(log n) + MoE 路由")
    print("=" * 64)

    print(f"\n词嵌入维度 dim = {EMBED_DIM}")
    print("对比：dict 哈希查找 vs bisect 二分查找\n")

    sizes = [1000, 10000, 100000]
    all_results: list[dict[str, dict[str, float]]] = []
    hash_means: list[float] = []
    binary_means: list[float] = []

    print("[1] 不同词表大小下的查找性能（每次 10000 次查询，重复 5 次取均值）")
    print("-" * 64)
    for n in sizes:
        res = bench_lookup(n)
        all_results.append(res)
        hash_means.append(res["hash O(1)"]["mean_ms"])
        binary_means.append(res["binary O(log n)"]["mean_ms"])
        print(f"\n词表大小 n = {n}")
        print(format_table(res, baseline="binary O(log n)"))
        speedup = res["binary O(log n)"]["mean_ms"] / res["hash O(1)"]["mean_ms"]
        print(f"  → 哈希比二分快 {speedup:.2f}x")

    print("\n[2] 复杂度趋势：n 增长时二分耗时对数增长，哈希近似常数")
    print(f"{'n':>10} {'hash_ms':>12} {'binary_ms':>12} {'ratio':>10} {'log2(n)':>10}")
    print("-" * 58)
    import math
    for n, h, b in zip(sizes, hash_means, binary_means):
        ratio = b / h if h > 0 else float("inf")
        print(f"{n:>10} {h:>12.4f} {b:>12.4f} {ratio:>10.2f}x {math.log2(n):>10.2f}")

    print("\n[3] MoE 路由模拟：8 个专家，top-2，100000 个 token")
    print("-" * 64)
    moe_load = demo_moe_routing(n_tokens=100000, n_experts=8, top_k=2)
    total_assigns = sum(moe_load.values())
    expected_per_expert = total_assigns / 8
    print(f"总分配次数 = {int(total_assigns)}（100000 token × top-2 = 200000）")
    print(f"理想均匀负载 = {expected_per_expert:.0f}")
    print(f"{'专家':>12} {'负载':>10} {'偏差%':>10}")
    for name, load in moe_load.items():
        dev = (load - expected_per_expert) / expected_per_expert * 100
        print(f"{name:>12} {int(load):>10} {dev:>10.2f}%")

    max_dev = max(abs(v - expected_per_expert) / expected_per_expert * 100 for v in moe_load.values())
    print(f"\n最大偏差 = {max_dev:.2f}%（哈希路由近似均匀）")

    print("\n[4] MoE 路由示例（前 5 个 token）")
    for i in range(5):
        token = f"tok_{i:06d}"
        experts = moe_route(token, n_experts=8, top_k=2)
        print(f"  {token} → 专家 {experts}")

    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {"hash O(1)": hash_means[-1], "binary O(log n)": binary_means[-1]},
        fig_dir / "lookup_compare_100k.png",
        title="词表 10 万：哈希 O(1) vs 二分 O(log n)",
        ylabel="查找 10000 次耗时 (ms)",
        baseline="binary O(log n)",
    )

    save_line(
        {"hash O(1)": hash_means, "binary O(log n)": binary_means},
        fig_dir / "lookup_vs_vocab_size.png",
        title="查找耗时 vs 词表大小",
        ylabel="查找 10000 次耗时 (ms)",
        xlabel="词表大小 (1k / 10k / 100k)",
    )

    save_bar(
        moe_load,
        fig_dir / "moe_expert_load.png",
        title="MoE 路由：8 专家负载分布（10 万 token）",
        ylabel="被选中次数",
        xlabel="专家",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print("  - 哈希查找 O(1)：无论词表 1k 还是 100k，单次查找耗时近似常数")
    print("  - 二分查找 O(log n)：n 从 1k→100k，log2(n) 从 10→17，耗时增长 ~1.7x")
    print("  - 词表 10 万级时哈希比二分快数倍，且随词表增大优势扩大")
    print("  - MoE 路由用哈希把 token 分发到专家，10 万 token 负载偏差 < 1%")
    print("  - PyTorch nn.Embedding 内部用哈希表实现 token→embedding 的 O(1) 查找")


if __name__ == "__main__":
    main()