# LRU 缓存与 KV Cache 淘汰：链表在 AI 里的不可替代场景

> 本文档讲清「LRU 缓存为什么必须用哈希表 + 双向链表、KV Cache 在长序列推理时怎么淘汰」。原理见 `principle.md`。

## 1. KV Cache：LLM 推理的显存瓶颈

### 1.1 什么是 KV Cache

Transformer 自回归生成时，每生成一个 token，要计算它对所有历史 token 的注意力。如果不缓存，每步都要重算所有历史 token 的 K 和 V，复杂度 $O(n^2)$。

KV Cache 把每层的 K 和 V 存起来，新 token 只算自己的 K/V 追加进去：

```
第 1 步: K=[k1],            V=[v1]
第 2 步: K=[k1,k2],         V=[v1,v2]
第 3 步: K=[k1,k2,k3],      V=[v1,v2,v3]
...
第 n 步: K=[k1,...,kn],     V=[v1,...,vn]
```

有了 KV Cache，每步注意力是 $O(n)$（新 token 对所有历史），总复杂度 $O(n^2)$ 但常数小很多。

### 1.2 KV Cache 的显存问题

KV Cache 的大小随序列长度线性增长：

$$
\text{KV Cache 大小} = 2 \times n_\text{layers} \times n_\text{heads} \times d_\text{head} \times \text{seq\_len} \times \text{batch} \times \text{sizeof(elem)}
$$

对 LLaMA-7B（32 层，32 头，128 维，batch=1）：
- 每个 token 的 KV Cache：$2 \times 32 \times 32 \times 128 \times 2\text{B} = 512\text{KB}$
- 4096 token：$512\text{KB} \times 4096 = 2\text{GB}$
- 32768 token：$16\text{GB}$

**KV Cache 是长序列推理的显存瓶颈**。batch 多个请求时，显存更紧张。

### 1.3 为什么需要淘汰

当多个请求共享 GPU 时，显存有限，不能所有请求的 KV Cache 都留着。需要淘汰「暂时不活跃」的请求的 KV Cache，给新请求腾空间。

**淘汰策略**：LRU（Least Recently Used）——淘汰最久没访问的。这就是链表 + 哈希表的用武之地。

## 2. LRU 缓存：哈希表 + 双向链表

### 2.1 LRU 的需求

LRU 缓存支持两个操作：
- `get(key)`：查 key，如果命中则「标记为最近访问」
- `put(key, value)`：存 key，如果容量满则淘汰「最久未访问」

两个操作都要求 $O(1)$。

### 2.2 朴素实现的瓶颈

**朴素方案**：用一个 list 维护访问顺序（最近访问在头），一个 dict 存数据。

```python
class LRUCacheNaive:
    def get(self, key):
        if key not in self.data: return None
        self.order.remove(key)   # O(n)！要在 list 里找 key
        self.order.insert(0, key)
        return self.data[key]
```

`list.remove(key)` 是 $O(n)$：要在 list 里线性搜索 key 的位置。每次 get 都 $O(n)$，n 次操作总 $O(n^2)$。

**这就是不用链表的代价**：list 的 `remove` 要先找位置（$O(n)$），再删（$O(n)$ 移动元素）。

### 2.3 哈希表 + 双向链表的 O(1) 方案

**核心思路**：
- **双向链表**维护访问顺序（头 = 最近，尾 = 最久）
- **哈希表**存 `key -> 链表节点`，让找节点 $O(1)$

```
哈希表: {1: node1, 2: node2, 3: node3}
链表:   head → [1] → [2] → [3] → tail
                          ↑ 最近访问 1，淘汰对象 3
```

`get(key)`：
1. 哈希表找节点：$O(1)$
2. 把节点移到链表头部：$O(1)$（双向链表，改几个指针）

`put(key, value)`：
1. 如果满，删链表尾节点 + 哈希表对应项：$O(1)$
2. 新节点插到链表头 + 哈希表：$O(1)$

**全程 $O(1)$**。这就是「哈希表 + 双向链表」的威力。

### 2.4 为什么必须双向链表

单链表不行，因为：
- 淘汰尾节点时，要把尾节点的前驱的 `next` 置空，但单链表只有 `next`，找前驱要 $O(n)$
- `move_to_front` 时，要脱链（改前驱的 next），单链表找前驱 $O(n)$

双向链表的 `prev` 指针让「找前驱」$O(1)$，这是 LRU 的关键。

## 3. demo 结果解读

### 3.1 性能对比（vocab=2000, cap=800, accesses=30000）

| 实现 | 耗时 (ms) | 加速比 |
|---|---|---|
| naive_list O(n) | 138.6 | 1.00x |
| hash+dlist O(1) | 14.7 | 9.41x |
| OrderedDict (C) | 9.5 | 14.64x |

**解读**：
- **朴素 list**：每次 `get` 要 `order.remove(key)`（$O(\text{cap})$），30000 次访问 × $O(800)$ = 24M 次操作，慢
- **哈希表 + 双向链表**：每次 `get` 是 $O(1)$，但纯 Python 的链表操作（对象创建、指针赋值）有解释器开销，仍比 C 慢
- **OrderedDict**：Python 内置，C 实现的哈希表 + 双向链表，最快

**关键**：哈希表 + 双向链表比朴素快 **9.4 倍**，这就是数据结构选择的价值。

### 3.2 命中率 vs 缓存容量

| 容量 | 命中率 |
|---|---|
| 50 | 7.3% |
| 100 | 14.6% |
| 200 | 28.8% |
| 500 | 67.4% |
| 800 | 97.4% |
| 1000 | 97.4% |

**解读**：
- 容量越大，能缓存越多，命中率越高
- 当容量接近 vocab（1000）时，命中率饱和（几乎所有 key 都在缓存里）
- **容量 500 是拐点**：从 28% 跳到 67%，说明访问有局部性，热数据约 500 个

### 3.3 命中率 vs 访问局部性

| 局部性 | 命中率 |
|---|---|
| 0.0（均匀随机） | 19.6% |
| 0.2 | 42.2% |
| 0.4 | 48.6% |
| 0.6 | 65.1% |
| 0.8 | 99.1% |
| 0.95 | 99.8% |

**解读**：
- 局部性 = 访问集中在热数据上的程度
- 局部性越强，LRU 越有效（热数据一直在缓存里）
- **LLM 推理的 KV Cache 有强局部性**：当前 batch 的请求频繁访问，其他 batch 的请求暂时不访问

## 4. KV Cache 淘汰的实际场景

### 4.1 vLLM 的淘汰策略

vLLM 是当前最流行的 LLM 推理框架，它的 KV Cache 管理结合了两种数据结构：

1. **分页内存**（第 11 章 PagedAttention）：解决碎片化
2. **LRU 淘汰**：当显存满时，淘汰最久没生成的请求的 KV Cache

当一个请求暂停生成（如等待用户输入），它的 KV Cache 可以被换出到 CPU 内存，给活跃请求腾空间。请求恢复时再换入。这个「换出/换入」的决策就是 LRU。

### 4.2 为什么 LRU 适合 KV Cache

- **时间局部性**：最近生成的 token 很可能再被访问（同一请求的后续 token 要对它算注意力）
- **空间局部性**：同一请求的 KV Cache 是连续的，一起访问
- **冷热分明**：当前生成的请求是热的，暂停的请求是冷的

LRU 正好捕捉「时间局部性」：最近访问的留着，最久没访问的淘汰。

### 4.3 其他淘汰策略对比

| 策略 | 决策依据 | 适合场景 |
|---|---|---|
| LRU | 最久未访问 | 通用，时间局部性强 |
| LFU | 访问次数最少 | 长期频率稳定 |
| FIFO | 最先进入 | 实现简单，但不利用局部性 |
| ARC | LRU + LFU 自适应 | 综合好但实现复杂 |

KV Cache 通常用 LRU，因为推理的时间局部性强（当前请求频繁访问，暂停请求不访问）。

## 5. 生产级实现考量

### 5.1 Python OrderedDict

Python 的 `OrderedDict` 就是 C 实现的哈希表 + 双向链表：

```python
od = OrderedDict()
od[key] = value
od.move_to_end(key, last=False)  # 移到头部
od.popitem(last=True)            # 弹出尾部（最久未访问）
```

源码在 CPython 的 `Objects/odictobject.c`，用 C 实现了双向链表，所以比纯 Python 快。

### 5.2 functools.lru_cache

Python 还提供了装饰器版本：

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)
```

底层也是哈希表 + 双向链表，C 实现。

### 5.3 C++ 的 LRU

C++ 标准库没有现成的 LRU，但可以用 `unordered_map` + `list`：

```cpp
class LRUCache {
    list<pair<int,int>> l;
    unordered_map<int, list<pair<int,int>>::iterator> m;
    int cap;
public:
    int get(int key) {
        auto it = m.find(key);
        if (it == m.end()) return -1;
        l.splice(l.begin(), l, it->second);  // O(1) 移到头部
        return it->second->second;
    }
};
```

`list::splice` 是 $O(1)$ 的指针操作，这正是双向链表的优势。

## 6. 链表在 AI 里的其他应用

除了 LRU，链表在 AI 里还有少量应用：

- **自由链表**（内存分配器）：空闲块用链表管理，分配/释放 $O(1)$
- **图的邻接表**（第 09 章）：每个顶点的邻居用链表存（但实践中常用 vector）
- **梯度累积**：某些框架用链表存历史梯度（但常用数组）

**总体而言，链表在 AI 里是辅助角色**，主力是数组（张量）。但 LRU 这个场景，链表不可替代。

## 7. 与其他章节的关联

- **数组**（第 01 章）：KV Cache 底层存储用数组（分块），链表只管淘汰顺序
- **哈希表**（第 05 章）：LRU 的另一半，$O(1)$ 找节点
- **PagedAttention**（第 11 章）：KV Cache 的分页管理，和 LRU 淘汰结合

## 8. 小结

LRU 缓存的核心：
1. **KV Cache 是长序列推理的显存瓶颈**，需要淘汰
2. **朴素 list LRU 是 $O(n)$**，长序列下不可接受
3. **哈希表 + 双向链表实现 $O(1)$**，快 9~15 倍
4. **双向链表的 `prev` 指针**让脱链 $O(1)$，这是关键
5. **命中率随容量和局部性增长**，LRU 捕捉时间局部性

链表在 AI 里不是主力，但在 LRU 这个「频繁插入删除 + 顺序维护」场景不可替代。这就是数据结构选择的价值：用对结构，性能差 10 倍。

下一篇：第 03 章 栈，讲自动微分反向传播栈。