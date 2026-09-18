# 图的 AI 应用：PageRank + GNN 消息传递

> 本文档讲清「图在 AI 里干什么」。原理见 `principle.md`。两个核心应用：**PageRank**（网页排名，图上随机游走）和 **GNN 消息传递**（图神经网络，沿边聚合邻居特征）。两者本质都是「沿边聚合」，邻接表让聚合只碰真实存在的边 $O(|V|+|E|)$，邻接矩阵则扫整个 $O(|V|^2)$——稀疏图上差距巨大。

## 1. PageRank：图上随机游走

**PageRank** 是 Google 创始人 Larry Page 1998 年提出的网页排名算法，核心思想：**一个网页重要，如果有很多重要网页链接到它**。这是图上随机游走的平稳分布。

### 1.1 从「投票」到迭代

把网页看成顶点，超链接看成有向边，整个网页图是一个有向图。PageRank 的直觉：

- 每个网页把自己的「重要度」均分给所有出链指向的网页
- 每个网页的重要度 = 收到的所有贡献之和

数学上，网页 $v$ 的 PageRank：

$$PR(v) = \frac{1-d}{n} + d \sum_{u \in \text{in}(v)} \frac{PR(u)}{\deg_{\text{out}}(u)}$$

- $n$：网页总数
- $d$：阻尼系数（通常 0.85），表示用户有 85% 概率沿链接走、15% 概率随机跳到任意页
- $\text{in}(v)$：指向 $v$ 的所有网页
- $\deg_{\text{out}}(u)$：$u$ 的出链数

$(1-d)/n$ 是「随机跳转」项，保证每个页都有基础分；求和项是「沿链接走」的贡献。

### 1.2 迭代求解

PageRank 是不动点方程，用**幂迭代**求解：

```
PR = [1/n, 1/n, ..., 1/n]  # 初始均匀
repeat iters:
    new_PR = [(1-d)/n, ...]  # 随机跳转项
    for each vertex u:
        share = d * PR[u] / deg_out(u)
        for each v in out_neighbors(u):  # 沿出边推送
            new_PR[v] += share
    PR = new_PR
```

每轮迭代，每个顶点把自己的 PR 值沿出边推给邻居。**核心操作是遍历所有边**——这正是邻接表的强项。

### 1.3 邻接表版 vs 邻接矩阵版

**邻接表版**（`pagerank_adj`）：

```python
for u in range(n):
    share = damping * pr[u] / out_degree[u]
    new_pr[adj_array[u]] += share  # 只碰 u 的实际出边
```

每轮 $O(|V| + |E|)$：遍历所有顶点 + 所有边。稀疏图 $|E| = O(|V|)$ 时，每轮 $O(|V|)$。

**邻接矩阵版**（`pagerank_mat`）：

```python
pr = teleport + damping * (MT @ pr)  # 矩阵-向量乘
```

每轮 $O(|V|^2)$：矩阵-向量乘扫整个 $n \times n$ 矩阵，包括所有 0。BLAS 向量化让常数极小，但渐近是 $O(|V|^2)$。

### 1.4 实测对比

n=1000, 边=5000, 50 轮迭代：

| 实现 | 耗时 | 内存 | 说明 |
|---|---|---|---|
| 邻接表（dict + 循环） | 93 ms | 286 KB | 逻辑 $O(\|V\|+\|E\|)$，但 Python 循环常数大 |
| 邻接矩阵（dense + BLAS） | 23 ms | 7812 KB | $O(\|V\|^2)$ 但 BLAS 向量化极快 |
| CSR（scipy.sparse） | 1.6 ms | 121 KB | $O(\|V\|+\|E\|)$ + C 层稀疏运算 |

纯 Python 邻接表最慢——不是算法问题，是 Python 解释器开销。CSR 版（邻接表的 C 层紧凑存储）比 dense 矩阵快 14x，同时内存省 65x。这就是「邻接表数据结构是对的，但要用 C/Cython 实现发挥 $O(|V|+|E|)$ 优势」。

### 1.5 不同稀疏度下的趋势

边数从 1k 到 100k（n=1000 固定）：

| 边数 | 密度 | 邻接表 | 矩阵 | CSR |
|---|---|---|---|---|
| 1000 | 0.002 | 33 ms | 15 ms | 0.5 ms |
| 5000 | 0.010 | 39 ms | 14 ms | 1.2 ms |
| 20000 | 0.040 | 44 ms | 14 ms | 1.7 ms |
| 50000 | 0.100 | 40 ms | 14 ms | 6.8 ms |
| 100000 | 0.200 | 49 ms | 14 ms | 14.6 ms |

- **邻接表/CSR 随边数线性增长**（$O(|V|+|E|)$），稀疏时极快
- **矩阵几乎不变**（$O(|V|^2)$），因为不管有没有边都扫整个矩阵
- 边数到 100k（密度 0.2）时 CSR 追上矩阵——图越稠密，矩阵的向量化优势越能抵消 $O(V^2)$ 的浪费

## 2. GNN 消息传递：沿边聚合特征

**图神经网络**（Graph Neural Network, GNN）是处理图结构数据的神经网络。核心机制是**消息传递**（message passing）：每个顶点聚合邻居的特征来更新自己的特征。

### 2.1 为什么需要 GNN

传统神经网络（CNN、RNN）处理的是规则结构（网格、序列）。但很多数据天然是图：

- 分子：原子是顶点，化学键是边
- 社交网络：用户是顶点，关注是边
- 知识图谱：实体是顶点，关系是边
- 引用网络：论文是顶点，引用是边

GNN 让神经网络能直接吃图结构数据，在药物发现、推荐系统、社交网络分析等领域是 SOTA。

### 2.2 消息传递框架

每一层 GNN 做两步：

```
对每个顶点 v:
  1. 聚合（Aggregate）：收集邻居特征
     agg_v = AGGREGATE({h_u : u in neighbors(v)})
  2. 更新（Update）：结合自己特征和聚合结果
     h_v' = UPDATE(h_v, agg_v)
```

最简单的版本（GCN 的核心）：

$$h_v' = h_v + \sum_{u \in \text{neighbors}(v)} h_u$$

每个顶点的新特征 = 自己 + 邻居特征之和。**核心操作是遍历所有边聚合特征**——和 PageRank 一样，都是「沿边聚合」。

### 2.3 邻接表版 vs 邻接矩阵版

**邻接表版**（`gnn_message_passing_adj`）：

```python
for v in range(n):
    nb = adj_array[v]
    if nb.size:
        new_features[v] += features[nb].sum(axis=0)  # 只碰实际邻居
```

每条边访问一次，$O(|V| + |E| \cdot d)$（$d$ 是特征维度）。稀疏图上极快。

**邻接矩阵版**（`gnn_message_passing_mat`）：

```python
new_features = features + adj_mat @ features  # 矩阵乘
```

$O(|V|^2 \cdot d)$，矩阵乘扫整个 $n \times n$。BLAS gemm 向量化极快，但稀疏图上大量算的是 $0 \times h = 0$ 的无用乘法。

### 2.4 实测对比

n=1000, 边=5000, 特征维度=32：

| 实现 | 耗时 | 说明 |
|---|---|---|
| 邻接表（dict + 循环） | 5.8 ms | 每顶点一次 numpy fancy index + sum |
| 邻接矩阵（dense + BLAS） | 1.15 ms | BLAS gemm，$O(V^2 d)$ 但常数极小 |
| CSR（scipy.sparse） | 0.19 ms | 稀疏矩阵乘，$O(|E|d)$ + C 层 |

CSR 比密集矩阵快 6x，比纯 Python 邻接表快 30x。GNN 的消息传递本质是稀疏矩阵-矩阵乘 $H' = H + AH$，CSR 让这个乘法只碰非零边。

### 2.5 消息传递 = 矩阵乘 = 边遍历

GNN 消息传递的三种视角，本质同一件事：

```
1. 顶点视角（邻接表）:
   for v: h_v' = h_v + sum(h_u for u in neighbors(v))

2. 边视角（沿边推送）:
   agg = zeros
   for edge (u,v): agg[v] += h_u
   h' = h + agg

3. 矩阵视角（矩阵乘）:
   h' = h + A @ h
```

邻接表实现「顶点视角」最自然（遍历每个顶点的邻居链表）；CSR 实现「边视角」最紧凑（连续遍历边数组）；密集矩阵实现「矩阵视角」向量化最强但浪费。**三者数学等价，性能差异来自数据结构**。

## 3. 稀疏图为什么必须用邻接表

### 3.1 空间：$O(|V|+|E|)$ vs $O(|V|^2)$

PageRank 和 GNN 处理的图都是稀疏的：

- 网页图：$|V| = 10^{11}$，平均度 30，$|E|/|V|^2 \approx 10^{-10}$
- 社交网络：$|V| = 10^9$，平均度 100，$|E|/|V|^2 \approx 10^{-7}$

邻接矩阵存 $10^{11} \times 10^{11} = 10^{22}$ 个元素——**宇宙里原子数都不到这个量级**。邻接表只存 $O(|V| + |E|) = O(10^{11} \times 30) = 3 \times 10^{12}$，差 $10^{10}$ 倍。

本 demo n=1000 的小图上，矩阵已经比邻接表多 27 倍内存。真实图上差距是天文数字。

### 3.2 性能：$O(|V|+|E|)$ vs $O(|V|^2)$

PageRank 每轮迭代、GNN 每层消息传递，都要遍历所有边一次。

- 邻接表：$O(|V| + |E|)$，稀疏图 $|E| = O(|V|)$ 时每轮 $O(|V|)$
- 邻接矩阵：$O(|V|^2)$，不管有没有边都扫整个矩阵

n=1000, $|E|=5000$ 时，邻接表碰 6000 个元素，矩阵碰 100 万个——差 166 倍。虽然 BLAS 向量化让矩阵的常数极小（本 demo 小图上矩阵比纯 Python 邻接表快），但：

1. **n 一大，$O(|V|^2)$ 主导**：n=10000 时矩阵是 $10^8$ 格，邻接表还是 $O(|E|)$
2. **内存墙**：矩阵扫 $10^8$ 格受内存带宽限制，CSR 只扫 $|E|$ 格
3. **CSR 兼顾两者**：$O(|V|+|E|)$ 空间 + C 层运算速度，本 demo 比 dense 快 6-16x

### 3.3 demo 的诚实结论

本 demo 在 n=1000 的小图上，纯 Python 邻接表比 dense 矩阵**慢**（PageRank 93ms vs 23ms）。这不是邻接表算法错了，是两层因素叠加：

1. **Python 循环常数大**：1000 次 Python for 循环 + numpy fancy index 调用，每次几微秒
2. **BLAS 矩阵乘常数小**：dense 矩阵-向量乘是 C/Fortran 层连续内存 BLAS，每次几纳秒

但 CSR 版（同样是 $O(|V|+|E|)$，但用 C 层稀疏运算）比 dense 快 16x。这说明：

> **邻接表数据结构是对的，但要用 C/Cython 实现才能发挥 $O(|V|+|E|)$ 优势。纯 Python 循环的邻接表只有教学价值，生产用 CSR。**

## 4. demo 结果解读

### 4.1 内存对比

```
邻接表内存     ≈    292952 字节 (286.1 KB)
邻接矩阵内存   =   8000000 字节 (7812.5 KB)
CSR 稀疏矩阵   =    124004 字节 (121.1 KB)
比值（矩阵/表） = 27.31x
比值（矩阵/CSR）= 64.51x
```

n=1000, 边=5000, 密度 0.01。邻接矩阵 8MB，邻接表 286KB，CSR 121KB。矩阵的 99% 是 0，纯浪费。CSR 比邻接表还省——没有动态数组的 header/capacity 开销，纯三个连续数组。

### 4.2 PageRank 性能

```
邻接表 PageRank          93.5698 ms   0.24x
邻接矩阵 PageRank         22.7981 ms   1.00x
CSR PageRank              1.6081 ms  14.18x
```

CSR 完胜：比密集矩阵快 14x，比纯 Python 邻接表快 58x。PageRank 每轮遍历所有边，CSR 的连续边数组 + C 层稀疏矩阵乘让这遍历极快。

### 4.3 GNN 消息传递性能

```
邻接表 GNN                5.8377 ms   0.20x
邻接矩阵 GNN               1.1544 ms   1.00x
CSR GNN                   0.1926 ms   5.99x
```

CSR 比密集矩阵快 6x。GNN 消息传递是 $H' = H + AH$（稀疏矩阵-矩阵乘），CSR 让乘法只碰非零边。

### 4.4 正确性验证

三种实现结果完全一致（max diff < 1e-10），证明邻接表、密集矩阵、CSR 三种表示数学等价，差异纯粹是数据结构 + 实现层。

## 5. 生产级实现：PyG 和 DGL

### 5.1 PyTorch Geometric (PyG)

PyG 是 PyTorch 的图神经网络库，底层用 **COO + CSR** 格式存图：

```python
from torch_geometric.data import Data
import torch

# edge_index: COO 格式，[2, num_edges]
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # 边 (0,1),(1,0),(1,2),(2,1)
x = torch.randn(3, 16)  # 3 顶点，16 维特征
data = Data(x=x, edge_index=edge_index)

from torch_geometric.nn import GCNConv
conv = GCNConv(16, 32)
out = conv(data.x, data.edge_index)  # 消息传递，底层 CSR 稀疏乘
```

PyG 的 `edge_index` 是 COO（Coordinate）格式——两个数组分别存边的源和目标。内部转 CSR 做稀疏矩阵乘，就是本 demo CSR 版的工业级实现。

### 5.2 Deep Graph Library (DGL)

DGL 是另一个主流 GNN 框架，支持多种后端（PyTorch、TF、MXNet），底层同样用 CSR：

```python
import dgl
import torch

g = dgl.graph(([0, 1, 1, 2], [1, 0, 2, 1]))  # 边
g.ndata['h'] = torch.randn(3, 16)  # 顶点特征

import dgl.nn as dglnn
conv = dglnn.GraphConv(16, 32)
out = conv(g, g.ndata['h'])  # 消息传递
```

DGL 的核心是 `g.update_all(message_func, reduce_func)`——显式表达「沿边发消息 + 顶点聚合」两步，底层用 CSR 稀疏运算加速。

### 5.3 为什么生产级都用 CSR

| 特性 | 邻接表（dict） | 邻接矩阵（dense） | CSR |
|---|---|---|---|
| 空间 | $O(\|V\|+\|E\|)$，常数大 | $O(\|V\|^2)$ | $O(\|V\|+\|E\|)$，常数小 |
| 缓存 | 差（指针跳转） | 好（连续） | 极好（连续） |
| 运算层 | Python 循环 | BLAS（C/Fortran） | 稀疏 BLAS（C/Fortran） |
| GPU 支持 | 无 | 有（cuBLAS） | 有（cuSPARSE） |
| 加删边 | $O(1)$ | $O(1)$ | $O(\|E\|)$（需重建） |

CSR 唯一劣势是加删边要重建（适合静态图）。GNN 的图通常静态（分子、引用网络），所以 CSR 是最优选择。需要动态加删边的场景用 COO 或专门的动态图结构。

## 6. 从 PageRank 到 GNN：统一的「沿边聚合」

PageRank 和 GNN 消息传递看似不同，本质都是**沿边聚合**：

| 算法 | 聚合什么 | 聚合方式 | 更新 |
|---|---|---|---|
| PageRank | 邻居的 PR 值 | 加权求和 $\sum PR(u)/\deg(u)$ | $PR \leftarrow (1-d)/n + d \cdot \text{agg}$ |
| GCN | 邻居的特征 | 求和 $\sum h_u$ | $h' = h + \text{agg}$ |
| GraphSAGE | 邻居特征 | 采样 + 聚合（mean/max/LSTM） | $h' = \sigma(W \cdot \text{concat}(h, \text{agg}))$ |
| GAT | 邻居特征 | 注意力加权求和 | $h' = \sum \alpha_{uv} W h_u$ |

所有这些算法的核心循环都是：

```
for each edge (u, v):
    agg[v] += something_from(u)  # 沿边推送
```

这个循环用邻接表是 $O(|E|)$，用密集矩阵是 $O(|V|^2)$。**稀疏图上邻接表/CSR 是唯一可行的选择**——这就是本 demo 的核心结论。

## 7. 总结

| 问题 | 答案 |
|---|---|
| 稀疏图为什么必须用邻接表？ | 空间 $O(\|V\|+\|E\|)$ vs $O(\|V\|^2)$，稀疏时差 $\|V\|/\text{avg\_deg}$ 倍；遍历只碰真实边 |
| GNN 消息传递怎么对应到边遍历？ | 聚合邻居特征 = 沿边推送消息，每条边访问一次 $O(\|E\|)$ |
| 邻接表 vs 矩阵内存差多少？ | n=1000 边=5000 时差 27x；真实图（密度 $10^{-7}$）差千万倍 |
| 邻接表 vs 矩阵性能差多少？ | CSR 比密集矩阵快 6-16x（本 demo）；纯 Python 邻接表因循环常数慢 |
| 生产级用什么？ | CSR 稀疏矩阵（PyG、DGL、scipy.sparse），$O(\|V\|+\|E\|)$ 空间 + C/CUDA 运算 |

**一句话**：PageRank 和 GNN 都是「沿边聚合」算法，邻接表/CSR 让聚合只碰真实存在的边 $O(|V|+|E|)$，密集矩阵扫 $O(|V|^2)$ 大量 0。稀疏图上邻接表是唯一选择，CSR 是它的生产级形态。