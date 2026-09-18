"""第 04 章 demo：Ring-AllReduce vs 树形 AllReduce 通信量对比。

展示「循环队列」在 AI 里的核心价值：
- 分布式训练需要 AllReduce 同步梯度
- 树形 AllReduce：根节点带宽瓶颈，通信量随 log2(n) 增长
- Ring-AllReduce：环形拓扑，每节点通信量 = 2*(n-1)/n * data_size，n→∞ 趋近常数 2*data_size
- 循环队列的头尾指针回绕 = 环形通信拓扑的令牌传递

跑法：
    python 04_queue/python/demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import format_table, save_bar, save_line  # noqa: E402


def ring_all_reduce(n_workers: int, data_size: float) -> dict[str, float]:
    """模拟 Ring-AllReduce，返回每节点通信量与步数。

    两阶段：
    1. Scatter-Reduce：n-1 步，每步每节点发 data_size/n 数据
       每节点总发送：(n-1) * data_size/n
    2. AllGather：n-1 步，每步每节点发 data_size/n 数据
       每节点总发送：(n-1) * data_size/n
    合计每节点：2*(n-1)*data_size/n
    """
    chunk = data_size / n_workers
    scatter_reduce_steps = n_workers - 1
    allgather_steps = n_workers - 1
    per_node_send = 2 * (n_workers - 1) * chunk
    return {
        "per_node_comm": per_node_send,
        "total_steps": float(scatter_reduce_steps + allgather_steps),
        "scatter_reduce_steps": float(scatter_reduce_steps),
        "allgather_steps": float(allgather_steps),
        "chunk_size": chunk,
    }


def tree_all_reduce(n_workers: int, data_size: float) -> dict[str, float]:
    """模拟树形 AllReduce，返回根节点（瓶颈）通信量与步数。

    两阶段（以根节点为瓶颈分析）：
    1. Reduce：log2(n) 步，每步非根节点发 data_size（聚合后的部分和）
       根节点接收：log2(n) * data_size
    2. Broadcast：log2(n) 步，根节点发 data_size
       根节点发送：log2(n) * data_size
    根节点总通信量：2 * data_size * log2(n)
    """
    log_n = math.log2(n_workers)
    reduce_steps = int(log_n)
    broadcast_steps = int(log_n)
    per_node_comm = 2 * data_size * log_n
    return {
        "per_node_comm": per_node_comm,
        "total_steps": float(reduce_steps + broadcast_steps),
        "reduce_steps": float(reduce_steps),
        "broadcast_steps": float(broadcast_steps),
        "log2_n": log_n,
    }


def main() -> None:
    print("=" * 64)
    print("第 04 章 demo：Ring-AllReduce vs 树形 AllReduce 通信量对比")
    print("=" * 64)

    data_size = 1.0
    workers = [4, 8, 16, 32, 64]

    print(f"\n数据大小 data_size = {data_size}（归一化，单位可以是 1 GB 梯度向量）\n")

    print("[1] 不同 worker 数下的每节点通信量")
    print(f"{'workers':>8} {'ring':>12} {'tree':>12} {'ring/tree':>12}")
    print("-" * 48)
    ring_vals: list[float] = []
    tree_vals: list[float] = []
    for n in workers:
        r = ring_all_reduce(n, data_size)
        t = tree_all_reduce(n, data_size)
        ring_vals.append(r["per_node_comm"])
        tree_vals.append(t["per_node_comm"])
        ratio = r["per_node_comm"] / t["per_node_comm"]
        print(f"{n:>8} {r['per_node_comm']:>12.4f} {t['per_node_comm']:>12.4f} {ratio:>12.4f}")

    print("\n[2] 步数对比（同步轮数）")
    print(f"{'workers':>8} {'ring_steps':>12} {'tree_steps':>12}")
    print("-" * 36)
    for n in workers:
        r = ring_all_reduce(n, data_size)
        t = tree_all_reduce(n, data_size)
        print(f"{n:>8} {r['total_steps']:>12.0f} {t['total_steps']:>12.0f}")

    print("\n[3] 极限分析：n → ∞")
    print(f"  Ring 每节点通信量 = 2*(n-1)/n * data_size → 2 * data_size = {2 * data_size}")
    print(f"  Tree 根节点通信量  = 2 * log2(n) * data_size → ∞（随 log2(n) 增长）")
    for n in [256, 1024, 4096]:
        r = ring_all_reduce(n, data_size)
        t = tree_all_reduce(n, data_size)
        print(
            f"  n={n:<5}  ring={r['per_node_comm']:.4f}  "
            f"tree={t['per_node_comm']:.4f}  tree/ring={t['per_node_comm'] / r['per_node_comm']:.2f}x"
        )

    print("\n[4] 固定 n=16，阶段详情")
    n = 16
    r = ring_all_reduce(n, data_size)
    t = tree_all_reduce(n, data_size)
    print(
        f"  Ring:  Scatter-Reduce {r['scatter_reduce_steps']:.0f} 步 + "
        f"AllGather {r['allgather_steps']:.0f} 步 = {r['total_steps']:.0f} 步"
    )
    print(
        f"         chunk = data_size/n = {r['chunk_size']:.6f}, "
        f"每节点通信量 = {r['per_node_comm']:.4f}"
    )
    print(
        f"  Tree:  Reduce {t['reduce_steps']:.0f} 步 + "
        f"Broadcast {t['broadcast_steps']:.0f} 步 = {t['total_steps']:.0f} 步"
    )
    print(
        f"         log2(n) = {t['log2_n']:.0f}, "
        f"根节点通信量 = {t['per_node_comm']:.4f}"
    )

    print("\n[5] 通信量汇总表（n=64，用 format_table 展示）")
    n = 64
    r = ring_all_reduce(n, data_size)
    t = tree_all_reduce(n, data_size)
    results = {
        "Ring-AllReduce": {"mean_ms": r["per_node_comm"], "std_ms": 0.0},
        "Tree-AllReduce": {"mean_ms": t["per_node_comm"], "std_ms": 0.0},
    }
    print(format_table(results, baseline="Tree-AllReduce"))

    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    n_bar = 16
    r_bar = ring_all_reduce(n_bar, data_size)
    t_bar = tree_all_reduce(n_bar, data_size)
    save_bar(
        {
            "Ring-AllReduce": r_bar["per_node_comm"],
            "Tree-AllReduce": t_bar["per_node_comm"],
        },
        fig_dir / "perf_compare.png",
        title=f"每节点通信量对比（n={n_bar}）",
        ylabel="通信量（归一化）",
        baseline="Tree-AllReduce",
    )

    save_line(
        {"Ring-AllReduce": ring_vals, "Tree-AllReduce": tree_vals},
        fig_dir / "comm_vs_workers.png",
        title="每节点通信量 vs worker 数",
        ylabel="通信量（归一化）",
        xlabel="worker 数（4~64）",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print("  - Ring 每节点通信量 = 2*(n-1)/n * data_size，n→∞ 趋近 2*data_size（常数）")
    print("  - Tree 根节点通信量 = 2*log2(n) * data_size，随 log2(n) 增长")
    print("  - Ring 关键：数据分 n 块环形传递，每节点每步只发 1/n 的数据")
    print("  - 循环队列头尾指针回绕 = 环形拓扑令牌传递，O(1) 入队出队对应 O(1) 步通信")
    print("  - n=1024 时 tree/ring ≈ log2(n)/2 ≈ 5x，GPU 越多 Ring 优势越大")


if __name__ == "__main__":
    main()