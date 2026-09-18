"""第 06 章 demo：Top-K 采样（堆 vs 排序 vs numpy）+ Beam Search（堆 vs 排序）。

展示「二叉堆 / 优先队列」在 AI 里的两个核心应用：
1. Top-K 采样：从 LLM logits（n=1e6 量级）选最大 K 个，小顶堆 O(n log k) vs 全排序 O(n log n)
2. Beam Search：每步从 beam×vocab 个候选里保留 top-beam，用堆剪枝 vs 全排序

跑法：
    python 06_heap/python/demo.py
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


def topk_heap(logits: list[float], k: int) -> list[float]:
    """小顶堆维护 K 大元素，O(n log k)。

    策略：堆大小始终 <= k，堆顶是当前 K 个里最小的。
    - 堆未满：直接 push
    - 堆满且新元素 > 堆顶：弹出最小（堆顶），push 新元素
    最后堆里就是全局 top-k，再排序输出（降序）。
    """
    if k <= 0:
        return []
    heap: list[float] = []
    for x in logits:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)


def topk_sort(logits: list[float], k: int) -> list[float]:
    """全排序取前 K，O(n log n)。朴素基线。"""
    if k <= 0:
        return []
    return sorted(logits, reverse=True)[:k]


def topk_numpy(logits: np.ndarray, k: int) -> np.ndarray:
    """np.argpartition 选 top-k，O(n) 平均。工业基准。

    argpartition 做一次划分把第 k 大元素放到正确位置，左边都更小，
    不保证有序，但 top-k 集合正确。再排序输出。
    """
    if k <= 0:
        return np.array([])
    if k >= len(logits):
        return np.sort(logits)[::-1]
    idx = np.argpartition(logits, -k)[-k:]
    return np.sort(logits[idx])[::-1]


def topk_heap_numpy(logits: np.ndarray, k: int) -> np.ndarray:
    """在 numpy 数组上跑小顶堆 top-k，纯 Python 循环，O(n log k)。

    用来隔离「堆算法」和「numpy 向量化」的差异——同样在 numpy 数组上循环，
    差异只来自算法（堆 vs 排序），不来自数据结构（list vs ndarray）。
    """
    if k <= 0:
        return np.array([])
    heap: list[float] = []
    push = heapq.heappush
    replace = heapq.heapreplace
    for x in logits:
        if len(heap) < k:
            push(heap, x)
        elif x > heap[0]:
            replace(heap, x)
    return np.sort(heap)[::-1]


def bench_topk(n: int, k: int, repeat: int = 5) -> dict[str, dict[str, float]]:
    """对比四种 top-k 实现，在 n 个随机 logits 上选 k 个最大值。"""
    rng = np.random.default_rng(42)
    logits_np = rng.standard_normal(n)
    logits_list = logits_np.tolist()

    def run_heap_list() -> None:
        topk_heap(logits_list, k)

    def run_sort_list() -> None:
        topk_sort(logits_list, k)

    def run_heap_np() -> None:
        topk_heap_numpy(logits_np, k)

    def run_numpy_partition() -> None:
        topk_numpy(logits_np, k)

    return compare(
        {
            "heap O(n log k)": run_heap_list,
            "sort O(n log n)": run_sort_list,
            "heap on ndarray": run_heap_np,
            "numpy argpartition": run_numpy_partition,
        },
        repeat=repeat,
        warmup=2,
    )


def beam_search_step_heap(
    beam_scores: list[float],
    vocab_logits: list[float],
    beam_width: int,
) -> list[float]:
    """Beam Search 一步扩展（堆版）。

    每个 beam 扩展 vocab_size 个候选，新分数 = beam_score + logit。
    从 beam_size * vocab_size 个候选里选 top beam_width，用小顶堆维护。
    """
    k = beam_width
    heap: list[float] = []
    push = heapq.heappush
    replace = heapq.heapreplace
    for bs in beam_scores:
        for logit in vocab_logits:
            score = bs + logit
            if len(heap) < k:
                push(heap, score)
            elif score > heap[0]:
                replace(heap, score)
    return sorted(heap, reverse=True)


def beam_search_step_sort(
    beam_scores: list[float],
    vocab_logits: list[float],
    beam_width: int,
) -> list[float]:
    """Beam Search 一步扩展（排序版）。

    生成所有候选后全排序取前 beam_width。朴素基线。
    """
    candidates = [bs + logit for bs in beam_scores for logit in vocab_logits]
    return sorted(candidates, reverse=True)[:beam_width]


def beam_search_step_numpy(
    beam_scores: np.ndarray,
    vocab_logits: np.ndarray,
    beam_width: int,
) -> np.ndarray:
    """Beam Search 一步扩展（numpy 版）。

    向量化生成所有候选（外积 + 展平），argpartition 选 top-beam。
    生产级 Beam Search 用类似策略但带索引回溯。
    """
    scores = (beam_scores[:, None] + vocab_logits[None, :]).ravel()
    k = beam_width
    idx = np.argpartition(scores, -k)[-k:]
    return np.sort(scores[idx])[::-1]


def beam_search_simulate(
    n_steps: int,
    beam_width: int,
    vocab_size: int,
    use_heap: bool,
) -> list[float]:
    """模拟 n_steps 步 Beam Search，返回最终 beam 分数。

    每步：当前 beam_width 个候选，每个扩展 vocab_size 个后续，
    从 beam_width * vocab_size 里选 top beam_width 进入下一步。
    """
    rng = np.random.default_rng(7)
    beam_scores = [0.0]
    for _ in range(n_steps):
        vocab_logits = rng.standard_normal(vocab_size).tolist()
        if use_heap:
            beam_scores = beam_search_step_heap(beam_scores, vocab_logits, beam_width)
        else:
            beam_scores = beam_search_step_sort(beam_scores, vocab_logits, beam_width)
    return beam_scores


def bench_beam_search(
    n_steps: int,
    beam_width: int,
    vocab_size: int,
    repeat: int = 3,
) -> dict[str, dict[str, float]]:
    """对比 Beam Search 单步扩展：堆 vs 排序 vs numpy。"""
    rng = np.random.default_rng(42)
    beam_scores_list = rng.standard_normal(beam_width).tolist()
    vocab_logits_list = rng.standard_normal(vocab_size).tolist()
    beam_scores_np = np.array(beam_scores_list)
    vocab_logits_np = np.array(vocab_logits_list)

    def run_heap() -> None:
        beam_search_step_heap(beam_scores_list, vocab_logits_list, beam_width)

    def run_sort() -> None:
        beam_search_step_sort(beam_scores_list, vocab_logits_list, beam_width)

    def run_numpy() -> None:
        beam_search_step_numpy(beam_scores_np, vocab_logits_np, beam_width)

    return compare(
        {
            "heap O(mv log beam)": run_heap,
            "sort O(mv log mv)": run_sort,
            "numpy argpartition": run_numpy,
        },
        repeat=repeat,
        warmup=1,
    )


def main() -> None:
    print("=" * 72)
    print("第 06 章 demo：Top-K 采样（堆 vs 排序）+ Beam Search（堆 vs 排序）")
    print("=" * 72)

    print("\n[1] Top-K 正确性验证：n=20, k=5")
    print("-" * 72)
    demo_logits = [3.1, -1.2, 5.7, 0.4, 2.8, 9.1, -4.0, 6.3, 1.5, 7.8,
                   2.2, 8.4, -0.7, 4.6, 5.0, 3.9, 6.7, 1.1, 0.0, 4.4]
    k = 5
    r_heap = topk_heap(demo_logits, k)
    r_sort = topk_sort(demo_logits, k)
    r_np = topk_numpy(np.array(demo_logits), k)
    print(f"  logits    = {demo_logits}")
    print(f"  topk_heap = {r_heap}")
    print(f"  topk_sort = {r_sort}")
    print(f"  topk_numpy= {list(r_np)}")
    assert r_heap == r_sort == list(r_np), "三种实现结果不一致"
    print(f"  ✓ 三种实现结果一致：top-{k} = {r_heap}")

    print("\n[2] Top-K 性能对比：n=1e6, k=10/50/100")
    print("-" * 72)
    n = 1_000_000
    ks = [10, 50, 100]
    all_topk: list[dict[str, dict[str, float]]] = []
    heap_means: list[float] = []
    sort_means: list[float] = []
    np_means: list[float] = []
    for k in ks:
        print(f"\n  n = {n:,}, k = {k}")
        res = bench_topk(n, k, repeat=5)
        all_topk.append(res)
        heap_means.append(res["heap O(n log k)"]["mean_ms"])
        sort_means.append(res["sort O(n log n)"]["mean_ms"])
        np_means.append(res["numpy argpartition"]["mean_ms"])
        print(format_table(res, baseline="sort O(n log n)"))
        speedup = res["sort O(n log n)"]["mean_ms"] / res["heap O(n log k)"]["mean_ms"]
        print(f"    → 堆比排序快 {speedup:.2f}x（k={k} 越小优势越大）")

    print("\n[3] 复杂度趋势：k 增大时堆优势缩小（log k → log n）")
    print(f"  {'k':>6} {'heap_ms':>12} {'sort_ms':>12} {'numpy_ms':>12} "
          f"{'sort/heap':>10} {'log2(n)/log2(k)':>16}")
    print("  " + "-" * 70)
    import math
    for k, hm, sm, nm in zip(ks, heap_means, sort_means, np_means):
        ratio = sm / hm if hm > 0 else float("inf")
        log_ratio = math.log2(n) / math.log2(k)
        print(f"  {k:>6} {hm:>12.3f} {sm:>12.3f} {nm:>12.3f} "
              f"{ratio:>10.2f}x {log_ratio:>16.2f}")

    print("\n[4] Beam Search 单步扩展性能：beam=5, vocab=不同大小")
    print("-" * 72)
    beam_width = 5
    vocab_sizes = [1000, 5000, 10000]
    beam_heap_means: list[float] = []
    beam_sort_means: list[float] = []
    beam_np_means: list[float] = []
    for vs in vocab_sizes:
        print(f"\n  beam_width = {beam_width}, vocab_size = {vs}, "
              f"候选数 = {beam_width * vs}")
        res = bench_beam_search(n_steps=1, beam_width=beam_width, vocab_size=vs, repeat=3)
        beam_heap_means.append(res["heap O(mv log beam)"]["mean_ms"])
        beam_sort_means.append(res["sort O(mv log mv)"]["mean_ms"])
        beam_np_means.append(res["numpy argpartition"]["mean_ms"])
        print(format_table(res, baseline="sort O(mv log mv)"))
        speedup = res["sort O(mv log mv)"]["mean_ms"] / res["heap O(mv log beam)"]["mean_ms"]
        print(f"    → 堆比排序快 {speedup:.2f}x")

    print("\n[5] Beam Search 完整模拟：10 步，beam=5，vocab=1000")
    print("-" * 72)
    final_heap = beam_search_simulate(n_steps=10, beam_width=5, vocab_size=1000, use_heap=True)
    final_sort = beam_search_simulate(n_steps=10, beam_width=5, vocab_size=1000, use_heap=False)
    print(f"  堆版最终 beam 分数 = {[f'{x:.4f}' for x in final_heap]}")
    print(f"  排序版最终 beam 分数 = {[f'{x:.4f}' for x in final_sort]}")
    max_diff = max(abs(a - b) for a, b in zip(sorted(final_heap, reverse=True),
                                              sorted(final_sort, reverse=True)))
    print(f"  最大差异 = {max_diff:.2e}（浮点误差范围内，两版结果一致）")

    print("\n[6] 关键数字：n=1e6, k=10 时堆 vs 排序")
    print("-" * 72)
    res_key = all_topk[0]
    t_heap = res_key["heap O(n log k)"]["mean_ms"]
    t_sort = res_key["sort O(n log n)"]["mean_ms"]
    t_np = res_key["numpy argpartition"]["mean_ms"]
    print(f"  小顶堆 top-10    : {t_heap:.3f} ms  (O(n log k) = O(1e6 * log 10)  ≈ 3.3e6)")
    print(f"  全排序 top-10    : {t_sort:.3f} ms  (O(n log n) = O(1e6 * log 1e6) ≈ 2.0e7)")
    print(f"  numpy argpartition: {t_np:.3f} ms  (O(n) 平均，C 向量化)")
    print(f"  排序/堆 = {t_sort/t_heap:.1f}x  |  排序/numpy = {t_sort/t_np:.1f}x  |  堆/numpy = {t_heap/t_np:.1f}x")
    print(f"  理论比值 log(n)/log(k) = {math.log2(1e6)/math.log2(10):.2f}x（堆应比排序快约这么多）")

    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {
            "小顶堆 O(n log k)": t_heap,
            "全排序 O(n log n)": t_sort,
            "numpy O(n)": t_np,
        },
        fig_dir / "topk_compare_n1e6_k10.png",
        title="Top-K 采样：n=1e6, k=10（堆 vs 排序 vs numpy）",
        ylabel="耗时 (ms)",
        baseline="全排序 O(n log n)",
    )

    save_line(
        {
            "小顶堆 O(n log k)": heap_means,
            "全排序 O(n log n)": sort_means,
            "numpy argpartition": np_means,
        },
        fig_dir / "topk_vs_k.png",
        title="Top-K 耗时 vs k（n=1e6）",
        ylabel="耗时 (ms)",
        xlabel="k (10 / 50 / 100)",
    )

    save_bar(
        {
            "小顶堆": beam_heap_means[-1],
            "全排序": beam_sort_means[-1],
            "numpy": beam_np_means[-1],
        },
        fig_dir / "beam_step_compare.png",
        title=f"Beam Search 单步：beam={beam_width}, vocab={vocab_sizes[-1]}",
        ylabel="耗时 (ms)",
        baseline="全排序",
    )

    save_line(
        {
            "小顶堆": beam_heap_means,
            "全排序": beam_sort_means,
            "numpy": beam_np_means,
        },
        fig_dir / "beam_step_vs_vocab.png",
        title=f"Beam Search 单步耗时 vs vocab_size（beam={beam_width}）",
        ylabel="耗时 (ms)",
        xlabel="vocab_size (1k / 5k / 10k)",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print("  - Top-K 用小顶堆 O(n log k)，k≪n 时远快于全排序 O(n log n)")
    print(f"  - n=1e6, k=10 时堆比排序快约 {t_sort/t_heap:.1f}x（理论 {math.log2(1e6)/math.log2(10):.1f}x）")
    print("  - k 越小堆优势越大；k→n 时堆退化为排序")
    print("  - numpy.argpartition 用 C 向量化 + introselect，比纯 Python 堆快一个量级")
    print("  - Beam Search 每步从 beam×vocab 候选选 top-beam，堆剪枝比全排序快")
    print("  - 生产级 LLM 推理用 numpy/CUDA argpartition，但原理仍是堆的 top-k 思想")


if __name__ == "__main__":
    main()