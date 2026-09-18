# 二叉堆的 AI 应用：Top-K 采样 + Beam Search

> 本文档讲清「堆在 AI 里干什么」。原理见 `principle.md`。核心两个应用：Top-K 采样（LLM 从 logits 选最大 K 个，为什么用小顶堆而不是排序）和 Beam Search（用优先队列剪枝保留 beam 个最优候选）。

## 1. Top-K 采样：LLM 解码的第一步

### 1.1 LLM 生成一个 token 的流程

大语言模型（LLM）每生成一个 token，最后会输出一个 **logits 向量** $\in \mathbb{R}^V$，$V$ 是词表大小：

```
vocab_size V = 128256 (LLaMA-3)
logits = [-3.2, 0.5, 8.7, -1.1, ..., 4.3]   ← V 个实数，对应每个词的"原始分数"
```

下一步要从中选一个（或几个）词作为输出。常见解码策略：

| 策略 | 做法 | 用到 Top-K？ |
|---|---|---|
| Greedy | 取 logits 最大的 1 个 | 是（k=1） |
| **Top-K 采样** | 取 logits 最大的 K 个，softmax 后按概率采样 | **是** |
| Top-P 采样（核采样） | 取最小的若干个使累计概率 $\ge$ p | 是（变体） |
| **Beam Search** | 每步保留 beam 个最优候选 | **是** |
| Temperature | logits / T 后再采样 | 配合 Top-K |

**Top-K 采样**（Holtzman et al. 2020）是最常见的解码策略之一：只在前 K 个高概率词里采样，过滤掉长尾低概率词（避免生成乱码）。

```
logits = [−3.2, 0.5, 8.7, −1.1, 6.3, 2.8, 5.1, ..., 4.3]   (V=128256)
K = 10

Top-K 采样:
1. 找出 logits 最大的 10 个 → [8.7, 6.3, 5.1, 4.3, ..., 2.8]
2. 对这 10 个做 softmax → 概率分布
3. 按概率随机选一个
```

第 1 步就是 **Top-K 问题**：从 $V$ 个数里找最大的 $K$ 个。$V$ 通常 3 万到 13 万（LLaMA-3 128256），$K$ 通常 10-50。

### 1.2 Top-K 问题的三种解法

| 方法 | 复杂度 | 说明 |
|---|---|---|
| 全排序取前 K | $O(n \log n)$ | 排完取前 K 个，杀鸡用牛刀 |
| **小顶堆维护 K 大** | **$O(n \log k)$** | **堆大小 K，堆顶是 K 里最小** |
| 快速选择 (quickselect) | $O(n)$ 平均 | 类似快排的划分，但只递归一侧 |
| introselect (numpy) | $O(n)$ 最坏 | quickselect + 退化时切中位数 |

**全排序**做了远超必要的工作：Top-K 只需要前 K 个的相对顺序（甚至不需要顺序，只要集合），排序却把全部 $n$ 个排好。$n=10^5, k=10$ 时，排序做 $\sim 1.7 \times 10^6$ 次比较，Top-K 只需 $\sim 3.3 \times 10^5$，差 5 倍。

### 1.3 小顶堆维护 Top-K 的算法

**核心思想**：维护一个大小为 $K$ 的**小顶堆**，堆顶是当前 K 个候选里**最小的**。遍历所有元素：

- 堆未满（size < K）：直接 push
- 堆满且新元素 $>$ 堆顶：新元素有资格进 Top-K，弹出最小（堆顶），压入新元素
- 堆满且新元素 $\le$ 堆顶：新元素比当前 K 个都小，丢弃

```python
def topk_heap(logits, k):
    heap = []
    for x in logits:
        if len(heap) < k:
            heapq.heappush(heap, x)       # 堆未满，直接进
        elif x > heap[0]:
            heapq.heapreplace(heap, x)    # 比堆顶大，替换
    return sorted(heap, reverse=True)     # 最后排序输出
```

**为什么用小顶堆而不是大顶堆？** 这是个反直觉的点：
- 要找最大的 K 个，直觉用大顶堆
- 但大顶堆要存全部 $n$ 个元素再取 K 个根，$O(n)$ 空间 + $O(n \log n)$ 时间
- **小顶堆只存 K 个**，堆顶是 K 里最小的，新元素只需和堆顶比一次就能决定是否入选
- $O(k)$ 空间 + $O(n \log k)$ 时间，$k \ll n$ 时远优

```
找 top-3 从 [5, 1, 8, 3, 9, 2, 7, 4, 6]:

x=5: 堆未满 → [5]
x=1: 堆未满 → [1, 5]           (小顶堆，1 在顶)
x=8: 堆未满 → [1, 5, 8]
x=3: 3 > 堆顶 1 → 替换 → [3, 5, 8]
x=9: 9 > 堆顶 3 → 替换 → [5, 8, 9]
x=2: 2 < 堆顶 5 → 丢弃
x=7: 7 > 堆顶 5 → 替换 → [7, 8, 9]
x=4: 4 < 堆顶 7 → 丢弃
x=6: 6 < 堆顶 7 → 丢弃

最终堆 [7, 8, 9] → top-3 = [9, 8, 7] ✓
```

### 1.4 复杂度分析：堆 vs 排序

**全排序**：
- 排序 $O(n \log n)$，取前 K $O(k)$，总计 $O(n \log n)$
- $n = 10^6, k = 10$：$10^6 \times 20 \approx 2 \times 10^7$ 次比较

**小顶堆 Top-K**：
- 遍历 $n$ 个元素，每个做一次比较 + 可能的 heapreplace
- heapreplace 是 $O(\log k)$（一次 sift_down）
- 但大部分元素比堆顶小，直接跳过（不进堆）
- 最坏 $O(n \log k)$，平均更好（只有约 $K \ln(n/K)$ 个元素会进堆）
- $n = 10^6, k = 10$：$10^6 \times 3.3 \approx 3.3 \times 10^6$ 次比较

**理论比值**：

$$\frac{O(n \log n)}{O(n \log k)} = \frac{\log n}{\log k} = \frac{\log_2 10^6}{\log_2 10} = \frac{19.9}{3.3} \approx 6.0$$

即 $n = 10^6, k = 10$ 时，**堆应比排序快约 6 倍**。

**$k$ 增大时优势缩小**：

| $k$ | $\log_2 k$ | $\log_2 n / \log_2 k$ | 堆优势 |
|---|---|---|---|
| 10 | 3.3 | 6.0 | 6x |
| 100 | 6.6 | 3.0 | 3x |
| 1000 | 10.0 | 2.0 | 2x |
| $n$ | $\log_2 n$ | 1.0 | 1x（退化为排序） |

$k \to n$ 时堆要存全部元素，退化为 $O(n \log n)$，和排序一样。**Top-K 堆的优势只在 $k \ll n$ 时显著**。

## 2. demo 结果解读

运行 `python 06_heap/python/demo.py` 的关键输出：

### 2.1 Top-K 性能对比（n=1e6）

```
n = 1,000,000, k = 10
实现                     均值(ms)      标准差(ms)        加速比
heap O(n log k)       59.3341       3.0776       5.39x
sort O(n log n)      320.0403       3.5718       1.00x
heap on ndarray      114.8984       5.4835       2.79x
numpy argpartition       7.2434       0.7092      44.18x
```

读法：从 100 万个 logits 选最大的 10 个：
- **小顶堆**：59 ms，$O(n \log k)$
- **全排序**：320 ms，$O(n \log n)$
- **numpy.argpartition**：7 ms，$O(n)$ 向量化

**堆比排序快 5.4 倍**，理论值 6.0x。差距来自：
- Python `heapq` 是 C 实现但每步有 Python 层循环开销
- `sorted` 是高度优化的 Timsort（C 实现），常数小
- 实测比值略低于理论是正常的（常数项摊薄）

### 2.2 不同 k 下的趋势

```
  k      heap_ms      sort_ms     numpy_ms  sort/heap  log2(n)/log2(k)
 10       59.334      320.040        7.243       5.39x             6.00
 50       58.373      323.631        7.003       5.54x             3.53
100       56.905      321.825        6.952       5.66x             3.00
```

观察：
- `sort_ms` 几乎不随 $k$ 变（320 ms 左右）——排序代价主要在 $n \log n$，和 $k$ 无关
- `heap_ms` 也几乎不随 $k$ 变（57-59 ms）——$k$ 从 10 到 100，$\log k$ 从 3.3 到 6.6 翻倍，但大部分元素根本不进堆（被堆顶挡掉），实际 heapreplace 次数远少于 $n$
- `numpy_ms` 最快且稳定（~7 ms）——C 向量化 + introselect，$O(n)$

**为什么堆耗时也不随 $k$ 明显增长？** 因为遍历 $n$ 个元素的 $O(n)$ 主导，每个元素和堆顶比一次是 $O(1)$，只有比堆顶大的才 heapreplace（$O(\log k)$）。$n = 10^6$ 个随机数里，期望只有约 $k \ln(n/k)$ 个会进堆（$k=10$ 时约 115 个），heapreplace 总开销 $O(k \log k \cdot \ln(n/k))$，远小于 $O(n)$。

### 2.3 numpy.argpartition 为什么快 8 倍

```
numpy argpartition:  7.2 ms   (比纯 Python 堆快 8.2x)
```

`np.argpartition` 用的是 **introselect**（introspective select）：
1. 主体是 quickselect（类似快排但只递归一侧），$O(n)$ 平均
2. 退化时（划分不平衡）切中位数做 pivot，保证 $O(n)$ 最坏
3. **C 实现 + SIMD 向量化**，遍历 $n$ 个元素在 C 层一次完成

纯 Python 堆（`heapq`）虽然 `heappush`/`heapreplace` 是 C 实现，但**外层 for 循环在 Python 层**，每次迭代有解释器开销。$10^6$ 次 Python 循环本身就几十毫秒，主导了耗时。

**结论**：算法层面堆 $O(n \log k)$ 比 argpartition $O(n)$ 略差（$k=10$ 时 $\log k = 3.3$，差 3.3 倍），但实现层面 numpy 的 C 向量化比 Python 循环快几十倍，综合下来 numpy 快 8 倍。

### 2.4 关键数字总结

```
n=1e6, k=10:
  小顶堆 top-10    : 59.334 ms  (O(n log k))
  全排序 top-10    : 320.040 ms (O(n log n))
  numpy argpartition: 7.243 ms  (O(n) 向量化)
  排序/堆 = 5.4x  |  排序/numpy = 44.2x  |  堆/numpy = 8.2x
  理论比值 log(n)/log(k) = 6.00x
```

## 3. Beam Search：用优先队列剪枝

### 3.1 什么是 Beam Search

**Beam Search** 是序列生成（机器翻译、LLM 解码）的经典算法。不同于贪心（每步只留 1 个）或穷举（留所有），Beam Search 每步保留 **beam_width 个最优候选**，在质量和效率间折中。

```
beam_width = 2, 生成 3 步:

步骤 0: [起] (1 个候选)
步骤 1: [起] 扩展 vocab → vocab 个候选 → 保留 top-2 → [起A, 起B]
步骤 2: [起A] 扩展 vocab + [起B] 扩展 vocab → 2*vocab 个候选 → 保留 top-2 → [起AC, 起BD]
步骤 3: [起AC] 扩展 vocab + [起BD] 扩展 vocab → 2*vocab 个候选 → 保留 top-2 → [起ACE, 起BDF]
```

每步核心操作：从 `beam_width × vocab_size` 个候选里选 **top beam_width** 个。这就是 Top-K 问题，$n = \text{beam} \times \text{vocab}$，$k = \text{beam}$。

### 3.2 Beam Search 一步的 Top-K

```python
def beam_search_step_heap(beam_scores, vocab_logits, beam_width):
    """一步扩展：每个 beam 产生 vocab 个候选，选 top beam_width。"""
    heap = []
    for bs in beam_scores:
        for logit in vocab_logits:
            score = bs + logit              # 新候选分数 = 累计分数 + 当前 logit
            if len(heap) < beam_width:
                heapq.heappush(heap, score)
            elif score > heap[0]:
                heapq.heapreplace(heap, score)
    return sorted(heap, reverse=True)
```

对比排序版：

```python
def beam_search_step_sort(beam_scores, vocab_logits, beam_width):
    candidates = [bs + logit for bs in beam_scores for logit in vocab_logits]
    return sorted(candidates, reverse=True)[:beam_width]
```

**复杂度**（$m = \text{beam\_width}$, $v = \text{vocab\_size}$）：
- 堆版：$O(mv \log m)$，遍历 $mv$ 个候选，每个 heapreplace $O(\log m)$
- 排序版：$O(mv \log(mv))$，全排序 $mv$ 个候选

比值 $\frac{\log(mv)}{\log m} = 1 + \frac{\log v}{\log m}$。$m=5, v=10000$ 时 $\frac{\log 50000}{\log 5} = \frac{15.6}{2.3} \approx 6.8$，堆应快约 7 倍。

### 3.3 Beam Search demo 结果

```
beam_width = 5, vocab_size = 1000, 候选数 = 5000
heap O(mv log beam)       0.4597 ms    1.68x
sort O(mv log mv)         0.7732 ms    1.00x
numpy argpartition        0.0285 ms   27.10x

beam_width = 5, vocab_size = 5000, 候选数 = 25000
heap O(mv log beam)       2.1626 ms    2.64x
sort O(mv log mv)         5.7081 ms    1.00x
numpy argpartition        0.0856 ms   66.68x

beam_width = 5, vocab_size = 10000, 候选数 = 50000
heap O(mv log beam)       4.1002 ms    3.03x
sort O(mv log mv)        12.4396 ms    1.00x
numpy argpartition        0.1649 ms   75.42x
```

观察：
- **候选数越多，堆优势越大**：5000 候选 1.68x → 50000 候选 3.03x。因为 $\log(mv)$ 随 $v$ 增长，而 $\log m$ 不变，比值拉大
- **numpy 远快**：50000 候选时 numpy 比堆快 25x（0.16 vs 4.1 ms），C 向量化 + $O(n)$ 算法双重优势
- **完整 Beam Search 10 步**：堆版和排序版结果完全一致（最大差异 0.0），验证算法正确

### 3.4 Beam Search 为什么用优先队列剪枝

不用堆的替代方案：

| 方案 | 每步代价 | 问题 |
|---|---|---|
| 全排序 + 取前 beam | $O(mv \log mv)$ | 排了 $mv$ 个却只用 beam 个，浪费 |
| **小顶堆剪枝** | $O(mv \log \text{beam})$ | 只维护 beam 个，其余 $mv - \text{beam}$ 个直接丢弃 |
| quickselect | $O(mv)$ 平均 | 需要存全部 $mv$ 个候选再选，空间 $O(mv)$ |
| numpy argpartition | $O(mv)$ 向量化 | 同上，且 C 层快 |

**堆的优势在在线/流式场景**：候选一个一个产生（逐 beam 扩展），堆可以边产生边剪枝，不需要存全部 $mv$ 个候选。内存只 $O(\text{beam})$。这在长序列生成（每步 vocab=3万，beam=4，候选 12 万）时省内存。

**numpy 的优势在批量场景**：候选一次性全算出来（向量化外积），argpartition 一次选完。生产级 LLM 推理走这条路。

## 4. 生产级实现：从堆到 CUDA Top-K

### 4.1 numpy.argpartition（CPU 推理）

```python
import numpy as np
logits = np.random.randn(128256)  # LLaMA-3 词表
k = 10
idx = np.argpartition(logits, -k)[-k:]  # top-k 索引，O(n) 向量化
top_k_logits = logits[idx]
```

`argpartition` 内部用 introselect（quickselect + 中位数退化保护），C 实现 + SIMD。比纯 Python 堆快 8 倍，是 CPU 上 Top-K 的事实标准。

### 4.2 PyTorch torch.topk（GPU 推理）

```python
import torch
logits = torch.randn(128256, device='cuda')
values, indices = torch.topk(logits, k=10)  # CUDA kernel
```

`torch.topk` 在 GPU 上用专门的 top-k kernel：
- 小 $k$：每个 block 维护一个寄存器堆，并行归约
- 大 $k$：先粗粒度 bucket sort 再精筛
- 比 GPU 上全排序快一个量级

### 4.3 CUDA Top-K kernel 的堆思想

GPU top-k 的核心仍是**堆的并行化变体**：

```cuda
// 每个 thread block 维护一个共享内存里的堆
__shared__ float heap[K];
// 每个 thread 处理一段 logits，本地筛选后归约到 block 堆
for (int i = tid; i < V; i += blockDim.x) {
    if (logits[i] > heap[0]) {
        heap_replace(heap, K, logits[i]);  // 原子操作
    }
}
// block 间再归约到全局 top-k
```

思想和小顶堆 Top-K 完全一致：维护大小 $K$ 的小顶堆，新元素比堆顶大就替换。区别是并行化（多 block 各自堆 + 归约）和硬件优化（共享内存、warp 原语）。

### 4.4 HuggingFace Transformers 里的 Top-K

```python
from transformers import LogitsProcessor

class TopKLogitsProcessor(LogitsProcessor):
    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, input_ids, scores):
        top_k = min(self.top_k, scores.size(-1))
        # 取 top-k 的阈值（第 k 大的值）
        kth_values = torch.topk(scores, top_k)[0][..., -1, None]
        # 把所有 < 阈值的 logit 设为 -inf（过滤）
        indices_to_remove = scores < kth_values
        scores = scores.masked_fill(indices_to_remove, -float("inf"))
        return scores
```

策略：用 `torch.topk` 找到第 $K$ 大的值作为阈值，把所有小于它的 logit 设为 $-\infty$（softmax 后概率为 0）。底层 `torch.topk` 在 GPU 上走 CUDA top-k kernel，在 CPU 上走 numpy/ATen 的 argpartition。

### 4.5 vLLM / TensorRT-LLM 的 Top-K

生产级 LLM 推理引擎（vLLM、TensorRT-LLM、TGI）的 Top-K 进一步优化：

1. **融合 kernel**：softmax + top-k + 采样在一个 kernel 里，避免中间结果写回显存
2. **Top-P 适配**：Top-P（核采样）需要排序后累计概率，用 radix sort（GPU 友好）
3. **批量处理**：一个 batch 的多个序列并行 top-k，共享 kernel 启动开销
4. **投机解码**：draft model 生成多个候选，target model 一次验证，top-k 选接受

这些优化的**算法内核仍是 top-k**，只是实现层面把堆/argpartition 搬到 GPU 并融合到解码流水线。

## 5. 堆在 AI 里的其他应用

| 应用 | 堆的作用 | 为什么用堆 |
|---|---|---|
| **Top-K 采样** | 从 logits 选最大 K 个 | $O(n \log k)$，$k \ll n$ |
| **Beam Search** | 每步保留 beam 个最优 | 在线剪枝，$O(\text{beam})$ 空间 |
| **Dijkstra 路由** | 优先队列取最小距离节点 | $O((V+E) \log V)$，图最短路 |
| **A* 搜索** | 优先队列按 $f = g + h$ 排序 | 启发式搜索，规划/游戏 AI |
| **KNN 检索** | 维护 K 个最近邻 | 暴力 KNN $O(n \log k)$（见第 08 章 kd-tree） |
| **MoE 负载均衡** | 把溢出专家的 token 转移 | 优先处理超载专家 |
| **Loss 求解** | top-k loss 只对最难的样本算 | hard example mining |
| **采样器** | reservoir sampling 的变体 | 流式 top-k |

### 5.1 Dijkstra 最短路（图 AI）

图神经网络（GNN）、知识图谱推理里的最短路径用 Dijkstra，核心是**优先队列**：

```python
import heapq
def dijkstra(graph, src):
    dist = {src: 0}
    heap = [(0, src)]           # (距离, 节点) 小顶堆
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

每次取距离最小的未处理节点（堆 pop $O(\log V)$），更新邻居距离（堆 push $O(\log V)$）。总 $O((V+E) \log V)$。堆是 Dijkstra 的心脏。

### 5.2 KNN 检索（第 08 章）

$k$-近邻检索的暴力实现：遍历所有 $n$ 个点，维护距离最小的 $k$ 个——和小顶堆 Top-K 完全同构。第 08 章的 kd-tree 是对此的优化（剪枝避免遍历全部），但叶子节点内部仍用堆维护 top-k。

### 5.3 Hard Example Mining（训练优化）

训练分类器时，只对 loss 最大的 top-k 个样本算梯度（hard example mining），堆维护 top-k loss：

```python
losses = model(batch)               # [batch_size] 每个样本的 loss
top_k_idx = topk_heap(losses, k=64) # 只对最难的 64 个回传梯度
loss = losses[top_k_idx].mean()
loss.backward()
```

减少梯度计算量，聚焦难样本，提升训练效率。

## 6. Top-K 采样的变体

### 6.1 Top-P（核采样）

Top-P 不固定 $K$，而是取最小的若干个使**累计概率 $\ge p$**：

```python
def top_p_sampling(logits, p=0.9):
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]      # 降序排
    cumsum = np.cumsum(probs[sorted_idx])
    # 累计概率刚到 p 的位置截断
    cutoff = np.searchsorted(cumsum, p) + 1
    return sorted_idx[:cutoff]
```

Top-P 需要排序（不能只堆 top-k，因为要累计概率），$O(n \log n)$。但实际只排序一次，且 $n$ 是词表大小（固定），开销可接受。

### 6.2 Top-K + Top-P 组合

生产级采样器常组合 Top-K 和 Top-P：先 Top-K 截断到 $K$ 个，再在 $K$ 个里做 Top-P。Top-K 用堆/argpartition $O(n \log k)$，Top-P 在 $K$ 个里排序 $O(k \log k)$，总计 $O(n \log k)$。

### 6.3 Typical Sampling

更新的解码策略（Typical Sampling, Meister et al. 2023）按信息熵选词，也需要 top-k 式的截断，底层仍是堆/argpartition。

## 7. 性能优化方向

### 7.1 从 Python 堆到 numpy

纯 Python `heapq` 的瓶颈是 Python 层循环。`np.argpartition` 把循环搬到 C，快 8 倍。CPU 推理应始终用 numpy/torch 的 topk。

### 7.2 从 CPU 到 GPU

GPU top-k kernel 并行化，比 CPU numpy 快 10-100x（取决于 $n$ 和 $k$）。LLM 推理 logits 在 GPU 上，直接 `torch.topk` 零拷贝。

### 7.3 融合 softmax + top-k + 采样

分开做：softmax $O(n)$ → top-k $O(n)$ → 采样 $O(k)$，三次读 logits。融合 kernel 一次读 logits 完成，省 2/3 显存带宽。vLLM/TensorRT-LLM 的采样 kernel 都走融合。

### 7.4 小词表的排序

$n$ 很小（如 $< 1000$）时，排序的常数优势可能压过堆的渐近优势（堆有 sift 的分支预测开销）。微基准测试：$n < 200$ 时 `sorted()[:k]` 可能比 `heapq` 快。生产级实现会按 $n$ 选策略。

## 8. 小结

堆在 AI 里的价值一句话：**把"从 $n$ 个里选 $K$ 个最大"从 $O(n \log n)$ 降到 $O(n \log k)$，在 $n=10^5$ 词表、$k=10$ 的 LLM 采样里，省下 5-6 倍的计算**。

两个核心应用：
1. **Top-K 采样**：LLM 从 logits 选最大 $K$ 个再 softmax 采样，小顶堆维护 $K$ 大元素，$O(n \log k)$
2. **Beam Search 剪枝**：每步从 beam×vocab 候选保留 top-beam，堆在线剪枝省内存

生产级实现用 numpy/torch 的 argpartition/topk（C/CUDA 向量化），但**算法内核仍是小顶堆 Top-K**——维护大小 $K$ 的小顶堆，新元素比堆顶大就替换。理解了这个 $O(n \log k)$ 的堆算法，就理解了为什么 LLM 解码能在 10 万词表上每秒生成上百个 token。