"""第 01 章 demo：分块矩阵乘法 vs 朴素实现。

展示「动态数组/连续内存」在 AI 里的核心价值：
- 矩阵乘法是 AI 最频繁的运算（神经网络全是矩阵乘）
- 朴素三重循环：缓存不友好，慢
- numpy 向量化：利用连续内存 + SIMD
- 分块矩阵乘法：让小块放进缓存，减少未命中
- BLAS（numpy @）：生产级实现，SIMD + 多线程 + 分块调优

跑法：
    python 01_array/python/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import compare, format_table, save_bar  # noqa: E402


def matmul_pure_python(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """最朴素的三重循环，纯 Python，n^3 次乘法。"""
    n = len(A)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def matmul_row_wise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """逐行用 numpy dot：部分向量化，但 B 按列访问不缓存友好。"""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        C[i] = A[i] @ B
    return C


def matmul_tiled(A: np.ndarray, B: np.ndarray, block: int = 32) -> np.ndarray:
    """分块矩阵乘法：分成 block x block 小块，让小块放进缓存。"""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(0, n, block):
        for j in range(0, n, block):
            for k in range(0, n, block):
                C[i:i + block, j:j + block] += (
                    A[i:i + block, k:k + block] @ B[k:k + block, j:j + block]
                )
    return C


def matmul_blas(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """numpy @ 直接调 BLAS：生产级实现。"""
    return A @ B


def main() -> None:
    print("=" * 60)
    print("第 01 章 demo：分块矩阵乘法 vs 朴素实现")
    print("=" * 60)

    # --- 1. 正确性验证（小矩阵）---
    print("\n[1] 正确性验证（n=16）")
    rng = np.random.default_rng(42)
    A_small = rng.standard_normal((16, 16))
    B_small = rng.standard_normal((16, 16))
    ref = A_small @ B_small

    c_pure = np.array(matmul_pure_python(A_small.tolist(), B_small.tolist()))
    c_row = matmul_row_wise(A_small, B_small)
    c_tiled = matmul_tiled(A_small, B_small, block=4)
    c_blas = matmul_blas(A_small, B_small)

    print(f"  pure_python  max err: {np.max(np.abs(c_pure - ref)):.2e}")
    print(f"  row_wise     max err: {np.max(np.abs(c_row - ref)):.2e}")
    print(f"  tiled(b=4)   max err: {np.max(np.abs(c_tiled - ref)):.2e}")
    print(f"  blas         max err: {np.max(np.abs(c_blas - ref)):.2e}")
    assert np.allclose(c_pure, ref, atol=1e-8)
    assert np.allclose(c_tiled, ref, atol=1e-8)

    # --- 2. 性能对比（纯 Python 用小矩阵，numpy 用大矩阵）---
    print("\n[2] 纯 Python 三重循环（n=80）")
    A80 = rng.standard_normal((80, 80)).tolist()
    B80 = rng.standard_normal((80, 80)).tolist()
    results_pure = compare({
        "pure_python_n80": lambda: matmul_pure_python(A80, B80),
    })
    print(format_table(results_pure))

    print("\n[3] numpy 各实现对比（n=512）")
    A = rng.standard_normal((512, 512))
    B = rng.standard_normal((512, 512))
    results = compare({
        "row_wise": lambda: matmul_row_wise(A, B),
        "tiled_b32": lambda: matmul_tiled(A, B, block=32),
        "tiled_b64": lambda: matmul_tiled(A, B, block=64),
        "blas(@)": lambda: matmul_blas(A, B),
    })
    print(format_table(results, baseline="row_wise"))

    # --- 3. 分块大小扫描 ---
    print("\n[4] 分块大小扫描（n=512）")
    block_sizes = [8, 16, 32, 64, 128, 256]
    tiled_results: dict[str, float] = {}
    for bs in block_sizes:
        r = compare({f"b{bs}": lambda bs=bs: matmul_tiled(A, B, block=bs)})
        tiled_results[f"b{bs}"] = r[f"b{bs}"]["mean_ms"]
        print(f"  block={bs:<4}  {r[f'b{bs}']['mean_ms']:.2f} ms")

    # --- 4. 保存图 ---
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {k: v["mean_ms"] for k, v in results.items()},
        fig_dir / "perf_compare.png",
        title="矩阵乘法 n=512：各实现耗时",
        ylabel="耗时 (ms)",
        baseline="row_wise",
    )

    save_bar(
        tiled_results,
        fig_dir / "block_size_scan.png",
        title="分块矩阵乘法 n=512：分块大小 vs 耗时",
        ylabel="耗时 (ms)",
        xlabel="分块大小",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print("  - 纯 Python 三重循环极慢（Python 解释器开销）")
    print("  - numpy 向量化后大幅提升（连续内存 + SIMD）")
    print("  - 分块 vs 逐行：分块让小块放进缓存，减少未命中")
    print("  - BLAS(@) 最快：SIMD + 多线程 + 分块调优")


if __name__ == "__main__":
    main()