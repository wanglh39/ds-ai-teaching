# 反向传播栈：链式法则的栈式实现

> 本文档讲清「自动微分为什么用栈、反向传播怎么弹栈应用链式法则」。原理见 `principle.md`。

## 1. 自动微分：AI 训练的引擎

### 1.1 为什么需要梯度

神经网络训练用梯度下降：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

要算 $\nabla_\theta L$（损失对每个参数的梯度）。一个 LLM 有几十亿参数，算梯度是训练的核心开销。

### 1.2 三种求导方式

| 方式 | 原理 | 复杂度 |
|---|---|---|
| 符号微分 | 表达式变换 | 表达式膨胀 |
| 数值微分 | 有限差分 $\frac{f(x+h)-f(x-h)}{2h}$ | $O(N)$ 次前向（N 个变量） |
| **自动微分** | **链式法则 + 计算图** | **1 次前向 + 1 次反向** |

自动微分（AD）是 AI 的选择：**1 次反向传播就能算出所有变量的梯度**，与变量数无关。

## 2. 计算图与链式法则

### 2.1 计算图

把表达式拆成基本操作，每个操作是一个节点：

$$
f(x, y) = (x + y) \cdot (x \cdot y)
$$

计算图：

```
x ──┬──→ (+) ──┐
y ──┤         │
    │         ├──→ (×) ──→ f
x ──┤         │
y ──┴──→ (×) ──┘
```

- 节点：变量或操作
- 边：数据流
- 前向传播：从叶到根算值
- 反向传播：从根到叶算梯度

### 2.2 链式法则

对 $f = (x+y) \cdot (xy)$，设 $u = x+y$，$v = xy$，则 $f = uv$。

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x} = v \cdot 1 + u \cdot y
$$

每个节点的梯度 = 上游传来的梯度 × 局部导数，累加到所有输入。

### 2.3 反向传播的顺序

关键：**梯度必须从输出往输入算**（先算 $\frac{\partial f}{\partial u}$，才能算 $\frac{\partial f}{\partial x}$）。

这就是「反向」传播。而前向传播是从输入往输出算值。**两者的顺序相反**。

## 3. 栈在反向传播里的角色

### 3.1 前向传播：压栈

前向传播时，每算一个节点就把它压栈：

```
前向: 算 x → 压栈; 算 y → 压栈; 算 u=x+y → 压栈; 算 v=xy → 压栈; 算 f=uv → 压栈
栈:  [x, y, u, v, f]  ← 顶
```

栈记录了求值顺序。

### 3.2 反向传播：弹栈

反向传播时，弹栈得到节点的**逆序**：

```
弹栈: f → v → u → y → x
```

这正是反向传播需要的顺序：先算 f 的梯度（=1），再算 v 的梯度，再算 u 的梯度，...

**栈的 LIFO 天然给出反拓扑序**，这就是栈在自动微分里的核心作用。

### 3.3 代码实现

本 demo 的 `backward` 函数：

```python
def backward(root):
    order = topo_sort(root)      # 拓扑排序（用栈做 DFS）
    root.grad = 1.0
    for node in reversed(order):  # 逆序遍历（弹栈）
        for parent, lg in zip(node._parents, node._local_grads):
            parent.grad += node.grad * lg  # 链式法则
```

- `topo_sort`：用栈做 DFS，得到前向求值顺序
- `reversed(order)`：逆序遍历，就是反向传播
- 每个节点把梯度传给父节点：`parent.grad += node.grad * local_grad`

### 3.4 拓扑排序也用栈

`topo_sort` 用显式栈做 DFS（而非递归）：

```python
def topo_sort(root):
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        stack.append((node, True))
        for p in node._parents:
            stack.append((p, False))
    return order
```

- 用 `(node, False)` 表示「待处理」，`(node, True)` 表示「已处理子节点」
- 弹出 `True` 的节点加入 order（后序）
- 弹出 `False` 的节点，压回 `(node, True)` + 压入所有父节点

**栈在两处用到**：拓扑排序（DFS）和反向传播（逆序遍历）。

## 4. demo 结果解读

### 4.1 正确性验证

- $f(x) = \sin(x^2) \cdot e^x$，$x=1.5$：解析梯度 -4.9588，数值梯度 -4.9588，误差 4e-10
- $f(x,y) = (x+y)(xy)$，$x=2, y=3$：$\frac{\partial f}{\partial x} = 21$，$\frac{\partial f}{\partial y} = 16$，完全正确

### 4.2 多变量性能（反向传播的核心优势）

| 变量数 N | 反向传播 (ms) | 数值梯度 (ms) | 加速比 |
|---|---|---|---|
| 10 | 0.054 | 0.019 | 0.4x |
| 50 | 0.215 | 0.217 | 1.0x |
| 100 | 0.396 | 0.839 | 2.1x |
| 500 | 2.66 | 19.1 | 7.2x |
| 1000 | 4.16 | 67.9 | 16.3x |

**解读**：
- **N 小时反向传播慢**：Python 对象创建开销大，数值梯度只需 2 次前向
- **N 大时反向传播快**：数值梯度要 $2N$ 次前向，反向传播只要 1 次前向 + 1 次反向
- **N=1000 时快 16x**：这就是自动微分的价值

**关键**：反向传播的复杂度是 $O(\text{操作数})$，与变量数无关（1 次反向算所有梯度）。数值梯度是 $O(N \times \text{操作数})$（每个变量都要 2 次前向）。

### 4.3 梯度消失

深层网络 $f(x) = \sin(\sin(\dots\sin(x)\dots))$（嵌套 1000 次）：

| 深度 | |梯度| |
|---|---|
| 10 | 3.8e-1 |
| 100 | 3.3e-2 |
| 1000 | 1.2e-3 |

梯度随深度衰减——因为 $\sin' = \cos < 1$，连乘趋零。这就是**梯度消失**，深层网络训练困难的根源。反向传播栈让这个现象清晰可见（每弹一层乘一个 $\cos$）。

## 5. PyTorch 的实现

PyTorch 的 `autograd` 就是生产级的反向传播栈：

```python
import torch
x = torch.tensor(1.5, requires_grad=True)
y = torch.sin(x**2) * torch.exp(x)
y.backward()  # 触发反向传播
print(x.grad)  # -4.9588
```

- 前向传播时构建计算图（每个 op 记录输入和局部导数）
- `backward()` 触发反向传播：拓扑排序 + 逆序遍历
- 用 C++ 实现，比纯 Python 快 100 倍

PyTorch 的计算图节点叫 `Node`，存在 `torch/csrc/autograd/function.h`。每个 op 注册一个 `backward` 函数，反向传播时按拓扑逆序调用。

## 6. 反向传播 vs 前向传播自动微分

自动微分有两种模式：

| 模式 | 方向 | 算所有梯度 | 复杂度 |
|---|---|---|---|
| 前向模式 | 输入→输出 | 每次算 1 个梯度 | $O(N)$ 次遍历 |
| **反向模式** | **输出→输入** | **1 次算所有梯度** | **$O(1)$ 次遍历** |

AI 用反向模式（reverse mode AD），因为：
- 输入多（几亿参数）、输出少（1 个 loss）
- 反向模式 1 次遍历算所有输入的梯度
- 前向模式要 N 次遍历

**反向模式天然用栈**（逆序），这就是栈在 AI 里的不可替代角色。

## 7. 与其他章节的关联

- **数组**（第 01 章）：栈的底层存储
- **图**（第 09 章）：计算图是 DAG，拓扑排序用栈
- **PagedAttention**（第 11 章）：反向传播也要存中间激活，KV Cache 的管理影响反向传播

## 8. 小结

反向传播栈的核心：
1. **链式法则的逆序**天然匹配栈的 LIFO
2. **前向压栈记录顺序，反向弹栈应用链式法则**
3. **1 次反向算所有梯度**，比数值梯度快 N 倍
4. **PyTorch autograd 就是生产级反向传播栈**

栈在 AI 里是自动微分的引擎，没有栈就没有高效的反向传播，就没有深度学习训练。

下一篇：第 04 章 循环队列，讲 Ring-AllReduce 分布式通信。