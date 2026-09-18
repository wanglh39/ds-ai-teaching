"""基准测试工具：计时 + 朴素 vs 优化对比 + 表格输出。

设计原则：
- 不依赖第三方库（Timer 用标准库 time.perf_counter）
- compare() 返回结构化结果，方便 viz 出图
- format_table() 输出对齐的文本表格，便于文档粘贴
"""

from __future__ import annotations

import time
from statistics import fmean, stdev
from typing import Callable, Sequence


class Timer:
    """上下文管理器 + 手动开关两用计时器。

    用法 1:
        with Timer() as t:
            ...
        print(t.elapsed_ms)

    用法 2:
        t = Timer()
        t.start()
        ...
        t.stop()
        print(t.elapsed_ms)
    """

    __slots__ = ("_start", "_end")

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> "Timer":
        self._end = time.perf_counter()
        return self

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def elapsed_s(self) -> float:
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1e3


def _bench_once(fn: Callable[[], None]) -> float:
    t = Timer()
    t.start()
    fn()
    t.stop()
    return t.elapsed_ms


def measure(
    fn: Callable[[], None],
    repeat: int = 5,
    warmup: int = 1,
) -> tuple[float, float]:
    """重复跑 fn，返回 (均值 ms, 标准差 ms)。"""
    for _ in range(warmup):
        fn()
    samples = [_bench_once(fn) for _ in range(repeat)]
    if len(samples) == 1:
        return samples[0], 0.0
    return fmean(samples), stdev(samples)


def compare(
    cases: dict[str, Callable[[], None]],
    repeat: int = 5,
    warmup: int = 1,
) -> dict[str, dict[str, float]]:
    """对比多个实现。

    Args:
        cases: {"朴素": fn_naive, "优化": fn_opt}
        repeat: 每个案例重复次数
        warmup: 预热次数

    Returns:
        {
            "朴素": {"mean_ms": ..., "std_ms": ...},
            "优化": {"mean_ms": ..., "std_ms": ...},
        }
    """
    results: dict[str, dict[str, float]] = {}
    for name, fn in cases.items():
        mean_ms, std_ms = measure(fn, repeat=repeat, warmup=warmup)
        results[name] = {"mean_ms": mean_ms, "std_ms": std_ms}
    return results


def format_table(
    results: dict[str, dict[str, float]],
    baseline: str | None = None,
) -> str:
    """把 compare() 的结果格式化成对齐的文本表格。

    Args:
        results: compare() 返回值
        baseline: 基准名，会算 speedup 列。None 则不显示 speedup。
    """
    header = f"{'实现':<16} {'均值(ms)':>12} {'标准差(ms)':>12}"
    if baseline:
        header += f" {'加速比':>10}"
    lines = [header, "-" * len(header)]
    base_mean = results[baseline]["mean_ms"] if baseline else 1.0
    for name, m in results.items():
        row = f"{name:<16} {m['mean_ms']:>12.4f} {m['std_ms']:>12.4f}"
        if baseline:
            speedup = base_mean / m["mean_ms"] if m["mean_ms"] > 0 else float("inf")
            row += f" {speedup:>10.2f}x"
        lines.append(row)
    return "\n".join(lines)