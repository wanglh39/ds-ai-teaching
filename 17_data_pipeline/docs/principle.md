# 第 17 章 · 数据管道与流式处理 — 数据结构原理

> 本章涉及 4 种数据结构：**队列**、**环形缓冲**、**树状数组（BIT）**、**布隆过滤器**。
> 它们在 AI 数据管道中各司其职，协作完成「流式加载 → 窗口保留 → 区间统计 → 去重」的全链路。

---

## 1. 队列（Queue）— 多阶段管道的连接器

### 1.1 基本概念

队列是先进先出（FIFO）的线性数据结构：一端入队（push），另一端出队（pop）。
在数据管道中，队列扮演「阶段间缓冲区」的角色——上游生产者 push，下游消费者 pop。

### 1.2 有界队列与背压

AI 数据管道中，各阶段处理速度不同（读盘快、GPU 训练慢）。如果用无界队列连接，
上游会不断堆积数据，最终 OOM（内存溢出）。**有界队列**（bounded queue）限制容量，
满了就触发**背压**（backpressure）：

- **阻塞式**：上游 push 满了就等，下游 pop 空了就等（生产级做法，如 Go channel）
- **丢弃式**：满了丢最老的（本 demo 做法，适合可丢帧的流式场景）
- **拒绝式**：满了就拒绝入队，返回错误（适合需要精确控制的场景）

### 1.3 多阶段管道

AI 训练数据的典型 4 阶段管道：

```
producer (读盘)  →  processor (解码/归一化)  →  augmenter (数据增强)  →  batcher (组 batch)
```

每两个阶段之间用一个有界队列连接。这样：

- 各阶段**解耦**：上游不用等下游，下游不用等上游
- 各阶段可**并行**：多线程/多进程各跑一个阶段，I/O 与 CPU 重叠
- 内存**可控**：有界队列限制每阶段的积压量

### 1.4 实现：deque + 容量上限

本 demo 用 `collections.deque(maxlen=capacity)` 实现有界队列：

```python
from collections import deque

class PipelineQueue:
    def __init__(self, capacity=64):
        self._q = deque(maxlen=capacity)  # 满了自动丢最老的

    def push(self, item):
        self._q.append(item)  # 满则 popleft（丢弃式背压）

    def pop(self):
        return self._q.popleft() if self._q else None
```

`deque(maxlen=N)` 在 CPython 中是 C 实现，push/pop 都是 O(1)，且自动处理容量上限。

### 1.5 复杂度

| 操作 | 复杂度 |
|------|--------|
| push | O(1) |
| pop  | O(1) |
| size | O(1) |

### 1.6 生产级实现

- **Python**：`queue.Queue`（线程安全，阻塞式）、`asyncio.Queue`（异步）、`multiprocessing.Queue`（进程间）
- **PyTorch**：`DataLoader(num_workers=N)` 内部用多进程队列连接 worker → 主进程
- **Kafka**：分布式消息队列，分区 + 消费者组实现多阶段管道
- **tf.data**：`Dataset.prefetch(N)` 用有界队列实现预取和背压

---

## 2. 环形缓冲（Ring Buffer）— 流式数据的滑动窗口

### 2.1 基本概念

环形缓冲是固定大小的数组 + 一个写指针（自增取模）。写入新数据时，指针前移，
超过容量就绕回数组开头，**覆盖最老的数据**。几何上像环形，故名。

### 2.2 为什么不用 list 或 deque？

| 方案 | 写入 | 覆盖最老 | 随机访问 | 内存 |
|------|------|----------|----------|------|
| `list + pop(0)` | O(1) append | O(n) pop(0) 左移 | O(1) | 动态 |
| `deque(maxlen=N)` | O(1) | O(1) 自动丢 | O(1) index | 固定 |
| `RingBuffer` | O(1) | O(1) 指针绕回 | O(1) | 固定 |

`deque(maxlen=N)` 已经是很好的滑动窗口实现。本 demo 额外实现 `RingBuffer` 的目的：
- 展示「环形」的几何直觉（写指针取模）
- 提供底层数组访问，便于与 BIT 等需要下标的数据结构协作

### 2.3 实现：定长数组 + 写指针

```python
class RingBuffer:
    def __init__(self, capacity):
        self._buf = [None] * capacity  # 预分配
        self._write = 0                # 写指针
        self._size = 0

    def push(self, x):
        self._buf[self._write] = x
        self._write = (self._write + 1) % self._cap  # 取模绕回
        if self._size < self._cap:
            self._size += 1

    def window(self):
        # 按写入顺序返回窗口内所有元素
        if self._size < self._cap:
            return [self._buf[i] for i in range(self._size)]
        return [self._buf[(self._write + i) % self._cap] for i in range(self._cap)]
```

关键点：
- **预分配定长数组**：不用动态扩容，内存固定
- **写指针取模**：`(write + 1) % cap` 实现环形绕回
- **覆盖最老**：写入位置已存的数据被直接覆盖，无需显式删除

### 2.4 AI 场景：滑动窗口监控

流式训练中，我们常需要「最近 N 步的 loss」「最近 N 个样本的梯度范数」等滑动窗口指标。
环形缓冲天然适合：每步 push 新值，窗口自动保留最近 N 个。

```
步 1: push(0.8) → 窗口 [0.8]
步 2: push(0.6) → 窗口 [0.8, 0.6]
...
步 N+1: push(0.5) → 窗口 [0.6, ..., 0.5]  # 0.8 被覆盖
```

### 2.5 复杂度

| 操作 | 复杂度 |
|------|--------|
| push | O(1) |
| window() | O(N) 遍历 |
| latest(k) | O(k) |

### 2.6 性能对比（demo 实测）

n=20000 次写入，cap=8192：

| 实现 | 耗时 (ms) | 加速比 |
|------|-----------|--------|
| list.pop(0) O(n) | ~112 | 1.00x |
| RingBuffer O(1) | ~5.5 | **20.31x** |

cap=8192 时，list.pop(0) 每次左移 8192 个元素，O(n) 开销显著；
RingBuffer 只改指针，O(1) 常数操作。窗口越大，RingBuffer 优势越明显。

---

## 3. 树状数组（Binary Indexed Tree, BIT）— 滑动窗口的区间统计

### 3.1 基本概念

树状数组（又称 Fenwick Tree）是一种支持**单点更新**和**前缀和查询**的数据结构，
两者都是 O(log n)。朴素数组更新 O(1) 但区间和 O(n)；前缀和数组区间和 O(1) 但更新 O(n)。
BIT 在两者之间取平衡：都是 O(log n)。

### 3.2 核心思想：lowbit 分块

定义 `lowbit(x) = x & -x`（x 的最低位 1 对应的值，如 lowbit(6)=2，lowbit(8)=8）。

BIT 用一个 `tree[]` 数组，其中 `tree[i]` 维护原数组区间 `[i - lowbit(i) + 1, i]` 的和。

```
原数组 a:  [a1, a2, a3, a4, a5, a6, a7, a8]
tree[1] = a1                          (lowbit=1, 区间 [1,1])
tree[2] = a1 + a2                     (lowbit=2, 区间 [1,2])
tree[3] = a3                          (lowbit=1, 区间 [3,3])
tree[4] = a1 + a2 + a3 + a4           (lowbit=4, 区间 [1,4])
tree[5] = a5                          (lowbit=1, 区间 [5,5])
tree[6] = a5 + a6                     (lowbit=2, 区间 [5,6])
tree[7] = a7                          (lowbit=1, 区间 [7,7])
tree[8] = a1 + a2 + ... + a8          (lowbit=8, 区间 [1,8])
```

每个 `tree[i]` 管一段，段长 = `lowbit(i)`。这种分块让更新和查询都只需碰 O(log n) 个节点。

### 3.3 两个核心操作

**单点更新 `update(i, delta)`**：把 a[i] 加上 delta，从 i 往上跳更新所有包含 a[i] 的 tree 节点。

```python
def update(self, i, delta):
    while i <= self._n:
        self._tree[i] += delta
        i += i & -i  # lowbit 跳：i → i + lowbit(i)
```

跳跃路径：i=3 → 4 → 8（3 的 lowbit=1，3+1=4；4 的 lowbit=4，4+4=8）。
每跳一步，区间长度翻倍，总共跳 O(log n) 步。

**前缀和查询 `query(i)`**：求 a[1] + a[2] + ... + a[i]，从 i 往下跳累加 tree 节点。

```python
def query(self, i):
    s = 0
    while i > 0:
        s += self._tree[i]
        i -= i & -i  # lowbit 跳：i → i - lowbit(i)
    return s
```

跳跃路径：i=7 → 6 → 4（7 的 lowbit=1，7-1=6；6 的 lowbit=2，6-2=4；4 的 lowbit=4，4-4=0 停）。
query(7) = tree[7] + tree[6] + tree[4] = a7 + (a5+a6) + (a1+a2+a3+a4) = a1+...+a7。

**区间和 `range_sum(l, r) = query(r) - query(l-1)`**。

### 3.4 实现

```python
class BIT:
    def __init__(self, n):
        self._n = n
        self._tree = [0.0] * (n + 1)  # 1-indexed
        self._arr = [0.0] * (n + 1)   # 原数组副本，支持 set 语义

    def update(self, i, delta):       # a[i] += delta
        self._arr[i] += delta
        while i <= self._n:
            self._tree[i] += delta
            i += i & -i

    def set(self, i, value):          # a[i] = value（覆盖）
        self.update(i, value - self._arr[i])

    def query(self, i):               # 前缀和 a[1..i]
        s = 0.0
        while i > 0:
            s += self._tree[i]
            i -= i & -i
        return s

    def range_sum(self, l, r):        # 区间和 a[l..r]
        return self.query(r) - self.query(l - 1)
```

### 3.5 AI 场景：滑动窗口统计

流式训练中需要频繁查询窗口内指标的区间和：

- **最近 N 步的 loss 之和**：监控训练趋势（上升 → 过拟合？下降 → 正常）
- **最近 N 步的梯度范数之和**：判断是否收敛
- **最近 N 个样本的某个特征分位数**：数据分布漂移检测

朴素做法：每次区间和都 O(n) 累加。窗口大且查询频繁（每步都查）时，总开销 O(n × T)。
BIT：更新 O(log n)，查询 O(log n)，总开销 O(log n × T)。

### 3.6 复杂度

| 操作 | 朴素数组 | BIT |
|------|----------|-----|
| 单点更新 | O(1) | O(log n) |
| 区间和 | O(n) | **O(log n)** |
| 空间 | O(n) | O(n) |

BIT 牺牲了更新的常数（O(1) → O(log n)），换取查询的大幅提升（O(n) → O(log n)）。
适合「更新和查询都频繁」的场景。

### 3.7 性能对比（demo 实测）

n=4096，做 n 次更新 + n 次区间和查询：

| 实现 | 耗时 (ms) | 加速比 |
|------|-----------|--------|
| 朴素累加 O(n) | ~68 | 1.00x |
| BIT O(log n) | ~15 | **4.41x** |

n=4096 时 BIT 加速 4.41x。n 更大时加速更明显（O(n) vs O(log n) 的差距随 n 增长）。

### 3.8 与其他区间统计数据结构的对比

| 数据结构 | 单点更新 | 区间和 | 区间最值 | 适用场景 |
|----------|----------|--------|----------|----------|
| 朴素数组 | O(1) | O(n) | O(n) | 更新多、查询少 |
| 前缀和数组 | O(n) | O(1) | — | 查询多、更新少 |
| BIT | O(log n) | O(log n) | — | 更新和查询都频繁（和） |
| 线段树 | O(log n) | O(log n) | O(log n) | 更新和查询都频繁（和/最值/等） |

BIT 比线段树常数更小、代码更短，但只支持「可减」的运算（和、异或）；线段树更通用。

---

## 4. 布隆过滤器（Bloom Filter）— 流式数据的近似去重

### 4.1 基本概念

布隆过滤器是一种**概率型集合**，支持 `add(x)` 和 `contains(x)`：

- `contains(x) = False` → x **一定不在**集合中（无假阴性）
- `contains(x) = True` → x **可能在**集合中（有假阳性，即误判）

用 m 位的位数组 + k 个哈希函数实现。空间 O(m bits)，远小于精确集合的 O(n × key_size)。

### 4.2 核心思想：多哈希 + 位数组

**add(x)**：把 k 个哈希位置都置 1。
```
h1(x), h2(x), ..., hk(x) → 位置都设为 1
```

**contains(x)**：检查 k 个位置是否全为 1。
```
全为 1 → 可能存在（可能误判）
有 0  → 一定不存在
```

**为什么不会漏（无假阴性）**：如果 x 在集合中，add 时把 k 个位置都置 1 了，
后续检查必然全为 1。

**为什么会误判（有假阳性）**：x 不在集合中，但其他元素的 add 恰好把 x 的 k 个位置都置 1 了。
概率随集合变大而增加，但可通过参数控制。

### 4.3 参数选择

给定预期元素数 n 和目标误判率 p：

```
m = -n × ln(p) / (ln2)²    # 位数
k = (m / n) × ln2           # 哈希函数个数
```

推导来自使误判率最小的优化。例如 n=10000, p=0.01：
- m = -10000 × ln(0.01) / (ln2)² ≈ 95850 bits ≈ 12 KB
- k = (95850 / 10000) × ln2 ≈ 6.6 → 6 个哈希

对比 set 存 10000 个 int：每个 int 约 28 字节 + 哈希表开销，约 400 KB。
布隆 12 KB vs set 400 KB，**内存省 33x**。

### 4.4 双哈希线性组合

用 k 个独立哈希函数不现实。实践中用**双哈希线性组合**：

```
h_i(x) = (h1(x) + i × h2(x)) mod m,  i = 0, 1, ..., k-1
```

只需两个独立哈希 h1、h2，即可生成 k 个哈希。证明见 Kirsch & Mitzenmüller (2006)。

本 demo 用 splitmix64 做整数混合哈希（快速且质量好）：

```python
def _hashes(self, key):
    x = key & 0xFFFFFFFFFFFFFFFF
    h1 = self._mix(x)
    h2 = self._mix(x ^ 0x9E3779B97F4A7C15)  # 黄金分割常数
    return [(h1 + i * h2) % self._m for i in range(self._k)]
```

### 4.5 实现

```python
class BloomFilter:
    def __init__(self, n, p=0.01):
        self._m = int(-n * math.log(p) / (math.log(2) ** 2))  # 位数
        self._k = int((self._m / n) * math.log(2))             # 哈希数
        self._bits = bytearray((self._m + 7) // 8)             # 位数组

    def add(self, key):
        for pos in self._hashes(key):
            self._bits[pos >> 3] |= 1 << (pos & 7)  # 置位

    def contains(self, key):
        return all(
            (self._bits[pos >> 3] >> (pos & 7)) & 1
            for pos in self._hashes(key)
        )
```

位数组用 `bytearray`（每字节 8 位），`pos >> 3` 定位字节，`pos & 7` 定位位。

### 4.6 AI 场景：流式数据去重

训练数据中常有重复样本（爬虫重复抓取、数据增强重复生成等）。重复样本会：

- **污染统计**：loss 被重复样本拉低，高估模型性能
- **浪费算力**：重复计算已知样本的梯度
- **导致过拟合**：模型对重复样本过拟合

精确去重用 `set`：内存 O(n × key_size)，百万级样本时内存压力大。
布隆去重：内存 O(m bits)，可控误判率。误判的代价是多丢弃少量「看似重复」的新样本，
对训练影响可控（训练数据本就有冗余）。

### 4.7 复杂度

| 操作 | set | 布隆 |
|------|-----|------|
| add | O(1) 平均 | O(k) ≈ O(1) |
| contains | O(1) 平均 | O(k) ≈ O(1) |
| 空间 | O(n × key_size) | **O(m bits)** |
| 准确性 | 精确 | 可能误判（假阳性） |

### 4.8 性能与内存对比（demo 实测）

n=50000 个 int key，插入 + 查询：

| 维度 | set | 布隆 | 说明 |
|------|-----|------|------|
| 速度 | 7.3 ms | 288 ms | Python 中 set 是 C 优化，布隆是纯 Python 循环 |
| 内存 | 3,497,340 bytes | 59,907 bytes | **布隆是 set 的 1/58** |

速度上 Python 层布隆不占优（set 是 C 实现），但内存上布隆优势巨大（58x）。
在 C/C++/Rust 中，布隆的 k 次哈希 + 位数组操作也是 O(1) 常数，速度可与 set 媲美甚至更快。

### 4.9 变体

- **Counting Bloom Filter**：每位用计数器（而非 1 位），支持删除
- **Cuckoo Filter**：用布谷鸟哈希，支持删除且空间更优
- **Scalable Bloom Filter**：动态扩容，无需预知 n

---

## 5. 四种数据结构的协作

### 5.1 协作流程

```
原始样本流
  │
  ▼
[PipelineQueue] 多阶段管道：producer → processor → augmenter → batcher
  │               队列连接各阶段，背压防 OOM
  ▼
[RingBuffer]    滑动窗口：保留最近 N 个样本的 loss，O(1) 写入/覆盖
  │               新数据覆盖最老的，内存固定
  ▼
[BIT]           窗口统计：O(log n) 区间和 / 前缀和
  │               朴素 O(n) 累加 → BIT O(log n)，窗口越大加速越明显
  ▼
[BloomFilter]   流式去重：O(1) 判重，内存远小于 set
  │               代价：可控误判率（假阳性，不漏真新样本）
  ▼
batch 输出 → 喂给模型训练
```

### 5.2 各数据结构的角色

| 数据结构 | 角色 | 解决的问题 | 复杂度 |
|----------|------|-----------|--------|
| 队列 | 管道连接 | 阶段间缓冲、背压、并行 | O(1) push/pop |
| 环形缓冲 | 窗口保留 | 固定内存保留最近 N 个 | O(1) push |
| BIT | 窗口统计 | O(log n) 区间和 | O(log n) update/query |
| 布隆过滤器 | 流式去重 | O(1) 判重，内存省 | O(k) add/contains |

### 5.3 为什么需要协作？

单一数据结构无法完成流式数据管道的全部需求：

- 只有队列：能连接管道，但无法统计窗口内指标
- 只有环形缓冲：能保留窗口，但查区间和要 O(n)
- 只有 BIT：能快速查区间和，但无法去重
- 只有布隆：能去重，但无法组织多阶段处理

四种协作：队列管「流」，环形缓冲管「窗」，BIT 管「窗内统计」，布隆管「去重」。
各司其职，共同支撑流式训练数据管道。

### 5.4 协作中的数据流

1. **样本进入管道**：producer 生成样本，push 到 q_load
2. **多阶段处理**：processor → augmenter，各阶段间用队列连接
3. **窗口更新**：每个有效样本的 loss push 到 RingBuffer，同时更新 BIT 对应槽位
4. **去重判断**：每个样本的 ID 先查布隆，可能重复则丢弃
5. **区间统计**：训练过程中随时用 BIT 查询窗口内 loss 之和，O(log n)
6. **batch 输出**：攒够 batch_size 个样本，输出一个 batch 喂给模型

### 5.5 生产级协作的例子

- **PyTorch DataLoader**：多进程队列（队列）+ prefetch buffer（环形缓冲）+ sampler 去重
- **WebDataset**：流式管道（队列）+ shard 缓存（环形缓冲）+ key 去重（布隆/集合）
- **Kafka Streams**：分区队列（队列）+ 窗口存储（环形缓冲/ RocksDB）+ exactly-once 去重（布隆）
- **TensorFlow tf.data**：prefetch（队列）+ sliding window（环形缓冲）+ cache 去重

---

## 6. 小结

| 数据结构 | 核心思想 | 关键操作 | AI 管道角色 |
|----------|----------|----------|-------------|
| 队列 | FIFO + 有界背压 | push/pop O(1) | 阶段间连接 |
| 环形缓冲 | 定长数组 + 写指针取模 | push O(1) 覆盖最老 | 滑动窗口保留 |
| BIT | lowbit 分块前缀和 | update/query O(log n) | 窗口区间统计 |
| 布隆过滤器 | 多哈希 + 位数组 | add/contains O(k) | 流式去重 |

四种数据结构分别解决流式管道的「连接」「保留」「统计」「去重」四个子问题，
协作完成 AI 训练数据的流式加载与处理。