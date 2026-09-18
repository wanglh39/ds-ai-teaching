"""第 11 章 demo：PagedAttention × AI 优化（KV Cache 显存管理对比）

对比三种 KV Cache 分配方案的显存利用率：
1. naive_prealloc：朴素预分配——每个序列预留 max_seq_len 连续空间（早期框架做法）
2. naive_dynamic：朴素动态连续分配——首次适应，释放后产生外部碎片
3. paged：分页分配——固定块大小，按需分配块，块可跨序列复用（PagedAttention 核心）

模拟多 batch 推理工作负载：随机生成长度不同的序列，分配/增长/释放/再分配。
统计三种方案的：显存利用率、碎片化率、能同时容纳的序列数。
对比不同序列长度分布（均匀/长尾/固定）下的表现。
保存图到 figures/。

跑法：
    python 11_paged_attention/python/demo.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, save_bar, save_line  # noqa: E402


@dataclass
class CacheSnapshot:
    step: int
    usage_ratio: float
    fragmentation: float
    n_active_seqs: int
    total_used_slots: int


@dataclass
class SimResult:
    name: str
    snapshots: list[CacheSnapshot] = field(default_factory=list)
    final_usage: float = 0.0
    final_frag: float = 0.0
    max_concurrent_seqs: int = 0
    n_rejected: int = 0
    mean_usage: float = 0.0
    mean_frag: float = 0.0


class NaivePreallocCache:
    """朴素预分配：每个序列预留 max_seq_len 的连续空间。

    早期 LLM 推理框架的做法：为每个请求预留最大长度的缓冲区。
    能容纳的序列数 = total_slots // max_seq_len。
    利用率 = 实际 token / total_slots（极低，因为大部分序列远短于 max_seq_len）。
    """

    def __init__(self, total_slots: int, max_seq_len: int) -> None:
        self.total_slots = total_slots
        self.max_seq_len = max_seq_len
        self.max_seqs = total_slots // max_seq_len
        self.allocated = [False] * self.max_seqs
        self.lengths = [0] * self.max_seqs
        self.total_used = 0

    def allocate(self, length: int) -> int:
        if length > self.max_seq_len:
            return -1
        for i in range(self.max_seqs):
            if not self.allocated[i]:
                self.allocated[i] = True
                self.lengths[i] = length
                self.total_used += length
                return i
        return -1

    def free(self, sid: int) -> bool:
        if sid < 0 or sid >= self.max_seqs or not self.allocated[sid]:
            return False
        self.allocated[sid] = False
        self.total_used -= self.lengths[sid]
        self.lengths[sid] = 0
        return True

    def usage_ratio(self) -> float:
        return self.total_used / self.total_slots

    def fragmentation(self) -> float:
        reserved = sum(
            self.max_seq_len for i in range(self.max_seqs) if self.allocated[i]
        )
        if reserved == 0:
            return 0.0
        return (reserved - self.total_used) / reserved

    def n_active(self) -> int:
        return sum(self.allocated)


class NaiveDynamicCache:
    """朴素动态连续分配：首次适应，类似 malloc/free。

    每个序列分配一段连续 slot，释放后产生外部碎片（空闲区间不连续）。
    新序列必须找到一段足够长的连续空闲区间才能分配，否则被拒绝。
    """

    def __init__(self, total_slots: int) -> None:
        self.total_slots = total_slots
        self.free_list: list[tuple[int, int]] = [(0, total_slots)]
        self.allocated: dict[int, tuple[int, int]] = {}
        self.total_used = 0

    def allocate(self, seq_id: int, length: int) -> bool:
        if length <= 0:
            return True
        for i, (start, flen) in enumerate(self.free_list):
            if flen >= length:
                self.allocated[seq_id] = (start, length)
                if flen == length:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (start + length, flen - length)
                self.total_used += length
                return True
        return False

    def free(self, seq_id: int) -> bool:
        if seq_id not in self.allocated:
            return False
        start, length = self.allocated.pop(seq_id)
        self.total_used -= length
        self.free_list.append((start, length))
        self.free_list.sort()
        merged: list[tuple[int, int]] = []
        for s, l in self.free_list:
            if merged and merged[-1][0] + merged[-1][1] == s:
                merged[-1] = (merged[-1][0], merged[-1][1] + l)
            else:
                merged.append((s, l))
        self.free_list = merged
        return True

    def usage_ratio(self) -> float:
        return self.total_used / self.total_slots

    def fragmentation(self) -> float:
        free_slots = self.total_slots - self.total_used
        if free_slots == 0:
            return 0.0
        max_free = max((l for _, l in self.free_list), default=0)
        return 1.0 - max_free / free_slots

    def n_active(self) -> int:
        return len(self.allocated)


class PagedCache:
    """分页 KV Cache：固定块大小，按需分配块（PagedAttention 核心思想）。

    显存被分成等大的块（block_size 个 slot/块），序列按需申请块，
    块来自全局空闲块池。序列释放后块归还池，可立即被其他序列复用。
    页表：逻辑页号 -> 物理块号，让序列看到连续的逻辑地址。
    碎片只有"内部碎片"——每序列最后一块可能未满，最多浪费 block_size-1 slot。
    """

    def __init__(self, total_slots: int, block_size: int) -> None:
        self.total_slots = total_slots
        self.block_size = block_size
        self.n_blocks = total_slots // block_size
        self.free_blocks: list[int] = list(range(self.n_blocks))
        self.page_tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}
        self.block_used: dict[int, int] = {}
        self.total_used = 0

    def allocate(self, seq_id: int, length: int) -> bool:
        if length <= 0:
            self.page_tables[seq_id] = []
            self.lengths[seq_id] = 0
            return True
        n_needed = (length + self.block_size - 1) // self.block_size
        if n_needed > len(self.free_blocks):
            return False
        blocks = [self.free_blocks.pop() for _ in range(n_needed)]
        self.page_tables[seq_id] = blocks
        self.lengths[seq_id] = length
        full = length // self.block_size
        rem = length % self.block_size
        for i in range(full):
            self.block_used[blocks[i]] = self.block_size
        if rem > 0:
            self.block_used[blocks[full]] = rem
        self.total_used += length
        return True

    def free(self, seq_id: int) -> bool:
        if seq_id not in self.page_tables:
            return False
        for b in self.page_tables[seq_id]:
            self.free_blocks.append(b)
            del self.block_used[b]
        self.total_used -= self.lengths[seq_id]
        del self.page_tables[seq_id]
        del self.lengths[seq_id]
        return True

    def usage_ratio(self) -> float:
        return self.total_used / self.total_slots

    def fragmentation(self) -> float:
        allocated_slots = len(self.block_used) * self.block_size
        if allocated_slots == 0:
            return 0.0
        return (allocated_slots - self.total_used) / allocated_slots

    def n_active(self) -> int:
        return len(self.page_tables)


def sample_uniform(rng: np.random.Generator) -> int:
    return int(rng.integers(32, 256))


def sample_longtail(rng: np.random.Generator) -> int:
    return int(rng.exponential(64)) + 16


def sample_fixed(rng: np.random.Generator) -> int:
    return 128


def simulate_prealloc(
    total_slots: int,
    max_seq_len: int,
    n_steps: int,
    sampler,
    rng: np.random.Generator,
) -> SimResult:
    cache = NaivePreallocCache(total_slots, max_seq_len)
    res = SimResult(name="朴素预分配")
    active: dict[int, int] = {}
    next_sid = 0
    for step in range(n_steps):
        if rng.random() < 0.65:
            length = min(sampler(rng), max_seq_len)
            sid = cache.allocate(length)
            if sid >= 0:
                active[sid] = length
            else:
                res.n_rejected += 1
        if active and rng.random() < 0.45:
            sid = rng.choice(list(active.keys()))
            cache.free(sid)
            del active[sid]
        snap = CacheSnapshot(
            step=step,
            usage_ratio=cache.usage_ratio(),
            fragmentation=cache.fragmentation(),
            n_active_seqs=cache.n_active(),
            total_used_slots=cache.total_used,
        )
        res.snapshots.append(snap)
        res.max_concurrent_seqs = max(res.max_concurrent_seqs, snap.n_active_seqs)
    res.final_usage = cache.usage_ratio()
    res.final_frag = cache.fragmentation()
    res.mean_usage = float(np.mean([s.usage_ratio for s in res.snapshots]))
    res.mean_frag = float(np.mean([s.fragmentation for s in res.snapshots]))
    return res


def simulate_dynamic(
    total_slots: int,
    n_steps: int,
    sampler,
    rng: np.random.Generator,
) -> SimResult:
    cache = NaiveDynamicCache(total_slots)
    res = SimResult(name="朴素动态分配")
    active: dict[int, int] = {}
    next_sid = 0
    for step in range(n_steps):
        if rng.random() < 0.65:
            length = sampler(rng)
            sid = next_sid
            next_sid += 1
            if cache.allocate(sid, length):
                active[sid] = length
            else:
                res.n_rejected += 1
        if active and rng.random() < 0.45:
            sid = rng.choice(list(active.keys()))
            cache.free(sid)
            del active[sid]
        snap = CacheSnapshot(
            step=step,
            usage_ratio=cache.usage_ratio(),
            fragmentation=cache.fragmentation(),
            n_active_seqs=cache.n_active(),
            total_used_slots=cache.total_used,
        )
        res.snapshots.append(snap)
        res.max_concurrent_seqs = max(res.max_concurrent_seqs, snap.n_active_seqs)
    res.final_usage = cache.usage_ratio()
    res.final_frag = cache.fragmentation()
    res.mean_usage = float(np.mean([s.usage_ratio for s in res.snapshots]))
    res.mean_frag = float(np.mean([s.fragmentation for s in res.snapshots]))
    return res


def simulate_paged(
    total_slots: int,
    block_size: int,
    n_steps: int,
    sampler,
    rng: np.random.Generator,
) -> SimResult:
    cache = PagedCache(total_slots, block_size)
    res = SimResult(name="PagedAttention")
    active: dict[int, int] = {}
    next_sid = 0
    for step in range(n_steps):
        if rng.random() < 0.65:
            length = sampler(rng)
            sid = next_sid
            next_sid += 1
            if cache.allocate(sid, length):
                active[sid] = length
            else:
                res.n_rejected += 1
        if active and rng.random() < 0.45:
            sid = rng.choice(list(active.keys()))
            cache.free(sid)
            del active[sid]
        snap = CacheSnapshot(
            step=step,
            usage_ratio=cache.usage_ratio(),
            fragmentation=cache.fragmentation(),
            n_active_seqs=cache.n_active(),
            total_used_slots=cache.total_used,
        )
        res.snapshots.append(snap)
        res.max_concurrent_seqs = max(res.max_concurrent_seqs, snap.n_active_seqs)
    res.final_usage = cache.usage_ratio()
    res.final_frag = cache.fragmentation()
    res.mean_usage = float(np.mean([s.usage_ratio for s in res.snapshots]))
    res.mean_frag = float(np.mean([s.fragmentation for s in res.snapshots]))
    return res


def run_one_scenario(
    total_slots: int,
    max_seq_len: int,
    block_size: int,
    n_steps: int,
    sampler,
    label: str,
    seed: int = 42,
) -> dict[str, SimResult]:
    rng_p = np.random.default_rng(seed)
    rng_d = np.random.default_rng(seed)
    rng_a = np.random.default_rng(seed)
    r_pre = simulate_prealloc(total_slots, max_seq_len, n_steps, sampler, rng_p)
    r_dyn = simulate_dynamic(total_slots, n_steps, sampler, rng_d)
    r_pag = simulate_paged(total_slots, block_size, n_steps, sampler, rng_a)
    return {label: {"prealloc": r_pre, "dynamic": r_dyn, "paged": r_pag}}


def main() -> None:
    print("=" * 76)
    print("第 11 章 demo：PagedAttention × AI 优化（KV Cache 显存管理对比）")
    print("=" * 76)

    total_slots = 4096
    max_seq_len = 512
    block_size = 16
    n_steps = 2000

    print(f"\n[配置] 总显存 = {total_slots} slot, max_seq_len = {max_seq_len}, block_size = {block_size}")
    print(f"        朴素预分配最多容纳 {total_slots // max_seq_len} 个序列")
    print(f"        分页方案共有 {total_slots // block_size} 个块")

    scenarios = {
        "均匀分布 [32,256)": sample_uniform,
        "长尾分布 (指数)": sample_longtail,
        "固定长度 128": sample_fixed,
    }

    all_results: dict[str, dict[str, SimResult]] = {}

    for label, sampler in scenarios.items():
        print(f"\n[场景] 序列长度分布：{label}")
        print("-" * 76)
        res = run_one_scenario(
            total_slots, max_seq_len, block_size, n_steps, sampler, label
        )
        all_results.update(res)
        r_pre = res[label]["prealloc"]
        r_dyn = res[label]["dynamic"]
        r_pag = res[label]["paged"]

        print(f"  {'方案':<18} {'平均利用率':>10} {'平均碎片化':>10} {'最大并发数':>10} {'被拒次数':>10}")
        for name, r in [("朴素预分配", r_pre), ("朴素动态分配", r_dyn), ("PagedAttention", r_pag)]:
            print(f"  {name:<18} {r.mean_usage:>10.2%} {r.mean_frag:>10.2%} {r.max_concurrent_seqs:>10d} {r.n_rejected:>10d}")

    print("\n[解读] 三种方案的核心差异")
    print("-" * 76)
    print("  朴素预分配：每序列预留 max_seq_len，利用率极低（实际长度 << max_seq_len）")
    print("              优点：无碎片、O(1) 分配；缺点：浪费严重，并发数受限于 total/max_seq_len")
    print("  朴素动态分配：连续分配，利用率较高，但释放后产生外部碎片")
    print("              长序列可能因找不到连续空闲区间而被拒（即使总空闲足够）")
    print("  PagedAttention：分页 + 按需分配，利用率接近 100%，碎片只有块内尾部（< block_size）")
    print("                  块可跨序列复用，并发数高，是 vLLM 的核心创新")

    print("\n[2] 显存利用率随时间变化（均匀分布场景）")
    print("-" * 76)
    uni = all_results["均匀分布 [32,256)"]
    r_pre = uni["prealloc"]
    r_dyn = uni["dynamic"]
    r_pag = uni["paged"]
    checkpoints = [0, 500, 1000, 1500, 1999]
    print(f"  {'step':>6} {'朴素预分配':>14} {'朴素动态分配':>14} {'PagedAttention':>14}")
    for cp in checkpoints:
        print(f"  {cp:>6} {r_pre.snapshots[cp].usage_ratio:>14.2%} {r_dyn.snapshots[cp].usage_ratio:>14.2%} {r_pag.snapshots[cp].usage_ratio:>14.2%}")

    print("\n[3] 不同块大小对 PagedAttention 的影响（均匀分布）")
    print("-" * 76)
    block_sizes = [4, 8, 16, 32, 64, 128]
    block_results: list[tuple[int, SimResult]] = []
    for bs in block_sizes:
        rng = np.random.default_rng(42)
        r = simulate_paged(total_slots, bs, n_steps, sample_uniform, rng)
        block_results.append((bs, r))
        print(f"  block_size={bs:>3}: 平均利用率={r.mean_usage:.2%}, 平均碎片化={r.mean_frag:.2%}, 最大并发={r.max_concurrent_seqs}, 被拒={r.n_rejected}")
    print("  解读：块越小 → 碎片越少、利用率越高，但页表越大、管理开销增加")
    print("        块越大 → 碎片越多（块内浪费），但页表越小、分配次数少")
    print("        vLLM 默认 block_size=16，是利用率和管理开销的平衡点")

    print("\n[4] 性能对比：相同工作负载下三种方案的吞吐")
    print("-" * 76)
    rng_bench = np.random.default_rng(123)
    n_bench_steps = 5000

    with Timer() as t_pre:
        simulate_prealloc(total_slots, max_seq_len, n_bench_steps, sample_uniform, np.random.default_rng(123))
    with Timer() as t_dyn:
        simulate_dynamic(total_slots, n_bench_steps, sample_uniform, np.random.default_rng(123))
    with Timer() as t_pag:
        simulate_paged(total_slots, block_size, n_bench_steps, sample_uniform, np.random.default_rng(123))

    print(f"  朴素预分配:     {t_pre.elapsed_ms:8.2f} ms ({n_bench_steps} 步)")
    print(f"  朴素动态分配:   {t_dyn.elapsed_ms:8.2f} ms ({n_bench_steps} 步)")
    print(f"  PagedAttention: {t_pag.elapsed_ms:8.2f} ms ({n_bench_steps} 步)")
    print(f"  注：分页方案每次分配/释放只操作块池（O(1) pop/append），")
    print(f"      动态分配需要扫描空闲区间（O(n_free)），预分配只需找空槽（O(max_seqs)）")

    print("\n[5] 保存对比图到 figures/")
    print("-" * 76)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {
            "朴素预分配": r_pre.mean_usage,
            "朴素动态分配": r_dyn.mean_usage,
            "PagedAttention": r_pag.mean_usage,
        },
        fig_dir / "kv_cache_usage_bar.png",
        title=f"显存利用率对比（{total_slots} slot, 均匀分布, {n_steps} 步）",
        ylabel="平均显存利用率",
    )
    print(f"  ✓ 保存 figures/kv_cache_usage_bar.png")

    save_bar(
        {
            "朴素预分配": r_pre.mean_frag,
            "朴素动态分配": r_dyn.mean_frag,
            "PagedAttention": r_pag.mean_frag,
        },
        fig_dir / "kv_cache_fragmentation_bar.png",
        title="碎片化率对比（均匀分布）",
        ylabel="平均碎片化率",
    )
    print(f"  ✓ 保存 figures/kv_cache_fragmentation_bar.png")

    save_bar(
        {
            "朴素预分配": float(r_pre.max_concurrent_seqs),
            "朴素动态分配": float(r_dyn.max_concurrent_seqs),
            "PagedAttention": float(r_pag.max_concurrent_seqs),
        },
        fig_dir / "kv_cache_concurrency_bar.png",
        title="最大并发序列数对比（均匀分布）",
        ylabel="最大并发序列数",
    )
    print(f"  ✓ 保存 figures/kv_cache_concurrency_bar.png")

    step_axis = list(range(0, n_steps, 20))
    save_line(
        {
            "朴素预分配": [r_pre.snapshots[s].usage_ratio for s in step_axis],
            "朴素动态分配": [r_dyn.snapshots[s].usage_ratio for s in step_axis],
            "PagedAttention": [r_pag.snapshots[s].usage_ratio for s in step_axis],
        },
        fig_dir / "kv_cache_usage_over_time.png",
        title="显存利用率随时间变化（均匀分布）",
        ylabel="显存利用率",
        xlabel="模拟步数",
    )
    print(f"  ✓ 保存 figures/kv_cache_usage_over_time.png")

    save_line(
        {
            "朴素预分配": [r_pre.snapshots[s].n_active_seqs for s in step_axis],
            "朴素动态分配": [r_dyn.snapshots[s].n_active_seqs for s in step_axis],
            "PagedAttention": [r_pag.snapshots[s].n_active_seqs for s in step_axis],
        },
        fig_dir / "kv_cache_concurrency_over_time.png",
        title="并发序列数随时间变化（均匀分布）",
        ylabel="并发序列数",
        xlabel="模拟步数",
    )
    print(f"  ✓ 保存 figures/kv_cache_concurrency_over_time.png")

    save_line(
        {
            "平均利用率": [r.mean_usage for _, r in block_results],
            "1 - 平均碎片化": [1.0 - r.mean_frag for _, r in block_results],
        },
        fig_dir / "paged_block_size_sweep.png",
        title="PagedAttention 利用率/碎片化 vs 块大小",
        ylabel="比率",
        xlabel="块大小 (4 / 8 / 16 / 32 / 64 / 128)",
    )
    print(f"  ✓ 保存 figures/paged_block_size_sweep.png")

    dist_names = list(all_results.keys())
    save_bar(
        {name: all_results[name]["paged"].mean_usage for name in dist_names},
        fig_dir / "paged_usage_by_distribution.png",
        title="PagedAttention 在不同序列长度分布下的利用率",
        ylabel="平均显存利用率",
    )
    print(f"  ✓ 保存 figures/paged_usage_by_distribution.png")

    print("\n" + "=" * 76)
    print("结论：PagedAttention 通过分页 + 按需分配 + 块复用，把 KV Cache 显存利用率")
    print(f"      从朴素方案的 {r_pre.mean_usage:.0%}~{r_dyn.mean_usage:.0%} 提升到 {r_pag.mean_usage:.0%}+，")
    print(f"      并发序列数从 {r_pre.max_concurrent_seqs}~{r_dyn.max_concurrent_seqs} 提升到 {r_pag.max_concurrent_seqs}。")
    print("      核心思想：把操作系统的分页机制（页表 + 固定块 + 空闲池）搬到 KV Cache 管理。")
    print("=" * 76)


if __name__ == "__main__":
    main()