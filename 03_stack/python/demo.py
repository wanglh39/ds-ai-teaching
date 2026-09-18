"""第 03 章 demo：手写反向传播栈，验证链式法则。

展示「栈」在 AI 里的核心价值：
- 自动微分的反向传播天然是栈式求值
- 前向传播时把操作压栈（记录求值顺序）
- 反向传播时弹栈（按反拓扑序应用链式法则）
- 栈的 LIFO 特性正好匹配「反向」传播

跑法：
    python 03_stack/python/demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import compare, format_table, save_bar, save_line  # noqa: E402


class Var:
    """自动微分变量：值 + 梯度 + 反向传播函数。

    每个操作创建新 Var，并记录局部导数（Jacobian），
    前向传播时把节点压栈，反向传播时弹栈累积梯度。
    """

    __slots__ = ("value", "grad", "_parents", "_local_grads")

    def __init__(self, value: float, parents: tuple = (), local_grads: tuple = ()) -> None:
        self.value = value
        self.grad = 0.0
        self._parents = parents
        self._local_grads = local_grads

    def __add__(self, other: "Var") -> "Var":
        return Var(self.value + other.value, (self, other), (1.0, 1.0))

    def __mul__(self, other: "Var") -> "Var":
        return Var(self.value * other.value, (self, other), (other.value, self.value))

    def __pow__(self, p: float) -> "Var":
        return Var(self.value ** p, (self,), (p * self.value ** (p - 1),))

    def sin(self) -> "Var":
        return Var(math.sin(self.value), (self,), (math.cos(self.value),))

    def exp(self) -> "Var":
        v = math.exp(self.value)
        return Var(v, (self,), (v,))


def topo_sort(root: Var) -> list[Var]:
    """用栈做 DFS 拓扑排序：前向传播的求值顺序。

    显式用栈（而非递归），展示栈在图遍历里的作用。
    """
    visited: set[int] = set()
    order: list[Var] = []
    stack: list[tuple[Var, bool]] = [(root, False)]

    while stack:
        node, processed = stack.pop()
        nid = id(node)
        if processed:
            order.append(node)
            continue
        if nid in visited:
            continue
        visited.add(nid)
        stack.append((node, True))
        for p in node._parents:
            if id(p) not in visited:
                stack.append((p, False))

    return order


def backward(root: Var) -> None:
    """反向传播：按拓扑逆序（栈的弹序）累积梯度。

    核心步骤：
    1. 前向传播时已经构建了计算图（每个 Var 记录 parents 和 local_grads）
    2. 拓扑排序得到求值顺序（用栈做 DFS）
    3. 逆序遍历（弹栈），应用链式法则：parent.grad += node.grad * local_grad
    """
    order = topo_sort(root)
    root.grad = 1.0
    for node in reversed(order):
        for parent, lg in zip(node._parents, node._local_grads):
            parent.grad += node.grad * lg


def numerical_grad(f, x: float, h: float = 1e-6) -> float:
    """数值梯度（有限差分），用于验证反向传播的正确性。"""
    return (f(x + h) - f(x - h)) / (2 * h)


def main() -> None:
    print("=" * 60)
    print("第 03 章 demo：反向传播栈 + 数值梯度验证")
    print("=" * 60)

    # --- 1. 正确性验证：f(x) = sin(x^2) * exp(x) ---
    print("\n[1] 正确性验证：f(x) = sin(x²) · exp(x)，x = 1.5")

    def f_func(x: float) -> float:
        return math.sin(x * x) * math.exp(x)

    x_val = 1.5
    x = Var(x_val)
    y = (x ** 2).sin() * x.exp()
    backward(y)

    num_grad = numerical_grad(f_func, x_val)
    print(f"  解析梯度（反向传播栈）: {x.grad:.10f}")
    print(f"  数值梯度（有限差分）  : {num_grad:.10f}")
    print(f"  误差                  : {abs(x.grad - num_grad):.2e}")
    assert abs(x.grad - num_grad) < 1e-5, "梯度不一致"

    # --- 2. 多变量：f(x,y) = (x + y) * (x * y) ---
    print("\n[2] 多变量：f(x,y) = (x+y) · (x·y)，x=2, y=3")
    x = Var(2.0)
    y = Var(3.0)
    z = (x + y) * (x * y)
    backward(z)
    print(f"  ∂z/∂x = {x.grad} (期望 (x·y) + (x+y)·y = 6 + 15 = 21)")
    print(f"  ∂z/∂y = {y.grad} (期望 (x·y) + (x+y)·x = 6 + 10 = 16)")
    assert x.grad == 21.0, f"∂z/∂x 应为 21，得到 {x.grad}"
    assert y.grad == 16.0, f"∂z/∂y 应为 16，得到 {y.grad}"

    # --- 3. 深层网络模拟：f(x) = sin(sin(...sin(x)...)) ---
    print("\n[3] 深层网络：f(x) = sin 嵌套 100 次，验证梯度爆炸/消失")
    depths = [10, 50, 100, 500, 1000]
    grad_results: dict[str, list[float]] = {"反向传播栈": [], "数值梯度": []}
    for depth in depths:
        x = Var(0.5)
        cur = x
        for _ in range(depth):
            cur = cur.sin()
        backward(cur)
        num = numerical_grad(lambda v: _nested_sin(v, depth), 0.5)
        grad_results["反向传播栈"].append(abs(x.grad))
        grad_results["数值梯度"].append(abs(num))
        print(f"  depth={depth:<5}  |grad|={abs(x.grad):.2e}  数值={abs(num):.2e}")

    # --- 4. 性能对比：多变量下反向传播 vs 数值梯度 ---
    print("\n[4] 多变量性能：N 个输入 → 1 个输出（反向传播 1 次反向 vs 数值 N 次前向）")

    def bp_multi(n: int) -> float:
        xs = [Var(0.1 * i) for i in range(n)]
        cur = xs[0]
        for x in xs[1:]:
            cur = cur + x * x
        backward(cur)
        return xs[0].grad

    def num_multi(n: int) -> list[float]:
        def f(vals: list[float]) -> float:
            s = vals[0]
            for v in vals[1:]:
                s += v * v
            return s
        vals = [0.1 * i for i in range(n)]
        grads = []
        h = 1e-6
        for i in range(n):
            orig = vals[i]
            vals[i] = orig + h
            fp = f(vals)
            vals[i] = orig - h
            fm = f(vals)
            vals[i] = orig
            grads.append((fp - fm) / (2 * h))
        return grads

    ns = [10, 50, 100, 500, 1000]
    perf_bp: list[float] = []
    perf_num: list[float] = []
    for n in ns:
        r = compare({"bp": lambda n=n: bp_multi(n), "num": lambda n=n: num_multi(n)}, repeat=3)
        perf_bp.append(r["bp"]["mean_ms"])
        perf_num.append(r["num"]["mean_ms"])
        print(f"  N={n:<5}  反向传播={r['bp']['mean_ms']:.3f}ms  数值={r['num']['mean_ms']:.3f}ms  加速比={r['num']['mean_ms']/r['bp']['mean_ms']:.1f}x")

    results = compare({
        "反向传播栈(N=1000)": lambda: bp_multi(1000),
        "数值梯度(N=1000)": lambda: num_multi(1000),
    })
    print(format_table(results, baseline="数值梯度(N=1000)"))

    # --- 5. 保存图 ---
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_line(
        grad_results,
        fig_dir / "grad_vs_depth.png",
        title="深层网络梯度：反向传播栈 vs 数值梯度",
        ylabel="|梯度|",
        xlabel="嵌套深度（10~1000）",
    )

    save_bar(
        {k: v["mean_ms"] for k, v in results.items()},
        fig_dir / "perf_compare.png",
        title="多变量反向传播 vs 数值梯度（N=1000）",
        ylabel="耗时 (ms)",
        baseline="数值梯度",
    )

    save_line(
        {"反向传播栈": perf_bp, "数值梯度": perf_num},
        fig_dir / "perf_vs_n.png",
        title="反向传播 vs 数值梯度：变量数 N 的影响",
        ylabel="耗时 (ms)",
        xlabel="变量数 N（10~1000）",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print("  - 反向传播用栈记录求值顺序，弹栈时应用链式法则")
    print("  - 栈的 LIFO 天然匹配「反向」传播的反拓扑序")
    print("  - 多变量时反向传播 1 次反向算出所有梯度，数值梯度要 N 次前向")
    print("  - N 越大反向传播优势越明显（这就是 PyTorch autograd 的基础）")
    print("  - 深层网络出现梯度消失（sin 的导数 cos < 1，连乘趋零）")


def _nested_sin(x: float, depth: int) -> float:
    for _ in range(depth):
        x = math.sin(x)
    return x


if __name__ == "__main__":
    main()