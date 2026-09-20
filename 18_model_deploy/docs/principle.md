# 第 18 章 · 模型部署与推理服务 — 数据结构原理

> 本章涉及 4 种数据结构：**计算图（DAG）**、**堆（优先队列）**、**一致性哈希环**、**页表**。
> 它们在大模型推理服务中各司其职，协作完成「请求调度 → 负载均衡 → 算子融合 → 显存管理」的全链路。

---

## 1. 计算图（DAG）— 算子融合的载体

### 1.1 基本概念

计算图（computational graph）是有向无环图（DAG）：节点是算子（operator），边是张量流动。
大模型（如 Transformer）的前向传播天然表示为计算图。

```
   x ─→ LayerNorm ─→ MatMul(Q) ─┐
        │           MatMul(K) ─┤→ Attention ─→ Scale ─→ Softmax ─→ MatMul ─→ Add(residual)
        └───────────────────────┘                                              │
   ────────────────────────────────────────────────────────────────────────────┘
   → LayerNorm ─→ MatMul(FFN1) ─→ GELU ─→ MatMul(FFN2) ─→ Add(residual) ─→ 输出
```

每个算子单独执行时，需要 launch 一个 GPU kernel。开销来自：
- **kernel launch 本身**：每次 ~5-10 μs（GPU 启动 kernel 的固定开销）
- **中间张量读写**：算子间中间结果写回 HBM（显存），下一个算子再读出

### 1.2 算子融合（Operator Fusion）

算子融合把相邻的算子合并成一个 kernel，减少 kernel 数量和中间张量读写。

**最典型的融合：element-wise 算子链**

element-wise 算子（Add、ReLU、Scale、GELU 等）对每个元素独立计算，天然可融合：
- 融合前：`y = ReLU(Add(a, b))` → 2 个 kernel，1 个中间张量
- 融合后：`y = AddReLU(a, b)` → 1 个 kernel，中间结果留在寄存器

**producer-consumer 融合**

更一般地，如果一个算子（producer）只有一个消费者（consumer），且 consumer 是 element-wise，
可以把 consumer 融合进 producer：
- producer 计算完一个 tile 后，直接在寄存器里做 consumer 的 element-wise 运算
- 中间结果不写回 HBM

### 1.3 实现：邻接表 + 拓扑序贪心融合

本 demo 用邻接表存 DAG，融合策略：

```python
class ComputeGraph:
    def fuse_elementwise(self):
        # 1. 统计每个算子的消费者数
        # 2. 按拓扑序扫描
        # 3. 当前算子是 element-wise 且只有一个前驱
        #    且前驱只有一个消费者 → 融合
        for name in self.topo_order():
            op = self._ops[name]
            if op.is_elementwise and len(in_preds) == 1:
                if consumer_count[pred] == 1:
                    op.fused = True  # 不再单独 launch kernel
```

融合条件：
- 当前算子是 element-wise
- 只有一个图内前驱（避免多输入的复杂融合）
- 前驱只有一个消费者（避免前驱被复制）

### 1.4 Transformer block 的融合效果

一个简化的 Transformer block 有 14 个算子：

| 算子 | 类型 | 可融合？ |
|------|------|----------|
| LayerNorm | reduction | 否 |
| MatMul × 4 (Q/K/V/out) | matmul | 否 |
| Attention | custom | 否 |
| Scale | element-wise | 是（接 Attention） |
| Softmax | reduction | 否 |
| Add × 2 (residual) | element-wise | 否（2 个前驱） |
| GELU | element-wise | 是（接 MatMul） |

本 demo 融合 3 个算子（Scale、GELU、以及一个 element-wise 链），kernel 数从 14 降到 11（减少 21%）。

### 1.5 复杂度

| 操作 | 复杂度 |
|------|--------|
| 构造图 | O(V + E) |
| 拓扑序 | O(V + E) |
| 贪心融合 | O(V + E) |

### 1.6 生产级实现

- **XLA**（TensorFlow/JAX）：自动融合 element-wise + reduction，生成 GPU/TPU 代码
- **MLIR**（PyTorch/TorchInductor）：基于 MLIR 的图融合 + codegen
- **TensorRT**：NVIDIA 的推理优化器，融合 conv+bn+relu、matmul+gelu 等
- **Triton**（OpenAI）：DSL 写 fused kernel，vLLM/TorchInductor 底层用 Triton

---

## 2. 堆 / 优先队列 — 请求调度的核心

### 2.1 基本概念

堆是完全二叉树，满足堆序性质：父节点优先级 ≥ 子节点。用数组实现：
- 根在 `heap[0]`（最大/最小优先级）
- 节点 i 的父在 `(i-1)//2`，左子在 `2i+1`，右子在 `2i+2`

Python `heapq` 是小顶堆。要实现大顶堆（优先级高的先出），存 `-priority`。

### 2.2 为什么推理服务需要优先级调度？

大模型推理服务面对混合请求：

| 请求类型 | 优先级 | SLA 要求 |
|----------|--------|----------|
| VIP 用户在线交互 | 5（最高） | < 100ms |
| 普通用户在线交互 | 3 | < 200ms |
| 后台批处理 | 1 | 无硬性要求 |

如果用 FIFO（先到先服务），一个低优先级的批处理请求可能排在 VIP 请求前面，
导致 VIP 用户等待过久，SLA 违约。

**优先级调度**：每次 pop 优先级最高的请求，VIP 优先服务。

### 2.3 SLA 违约率

SLA（Service Level Agreement）要求请求在 deadline 前完成。违约率 = 违约数 / 总数。

- **FIFO**：不看优先级，所有请求平等竞争 → 高优先级请求容易违约
- **优先级调度**：高优先级先服务 → 高优先级违约率低，低优先级可能违约（可接受）

本 demo 在过载场景下（服务能力略低于请求速率）：
- FIFO 违约率：285/300 = 95%
- 优先级违约率：219/300 = 73%（高优先级请求几乎不违约）

### 2.4 实现：heapq + tie-breaker

```python
import heapq

class RequestScheduler:
    def __init__(self):
        self._heap = []

    def push(self, req):
        # (-priority, arrive_ts, req_id, req)
        # 负优先级 → 大优先级先出
        # arrive_ts → 同优先级 FIFO
        # req_id → 最终 tie-breaker，避免比较 InferenceRequest
        heapq.heappush(self._heap, (-req.priority, req.arrive_ts, req.req_id, req))

    def pop(self):
        return heapq.heappop(self._heap)[3]
```

tie-breaker 设计：
- `-priority`：大优先级先出（小顶堆存负值）
- `arrive_ts`：同优先级先到先服务
- `req_id`：避免比较 `InferenceRequest` 对象（Python 3 不允许比较 dataclass）

### 2.5 复杂度

| 操作 | 复杂度 |
|------|--------|
| push | O(log n) |
| pop  | O(log n) |
| peek | O(1) |

### 2.6 生产级实现

- **vLLM**：连续批处理 + 优先级调度，支持 preemption（高优先级抢占低优先级）
- **Triton Inference Server**：多模型多队列，按优先级和模型实例调度
- **TensorRT-LLM**：in-flight batching，动态优先级调整
- **Redis + Lua**：分布式优先队列，跨节点调度

---

## 3. 一致性哈希环 — 负载均衡的低迁移方案

### 3.1 基本概念

一致性哈希（consistent hashing）把节点和 key 都映射到同一个哈希环上，
key 顺时针找最近的节点。

```
哈希环（0 ~ 2^64）:
        node_A (hash=1000)
       /
      /  key1 → node_A
     /
    ──────────────── node_B (hash=3000)
                    │
                    │  key2 → node_B
                    │
                    node_C (hash=5000)
```

### 3.2 为什么不用取模哈希？

取模哈希：`node = hash(key) % N`。简单，但节点增减时 N 变化，几乎所有 key 重新映射。

**场景**：3 个 GPU worker → 加 1 个 GPU 扩容：
- 取模哈希：`hash(key) % 3 → hash(key) % 4`，约 75% 的 key 要迁移
- 一致性哈希：新节点只接管环上一段区间，约 25% 的 key 迁移

本 demo 实测（3→4 节点，1000 keys）：
- 取模哈希迁移：763/1000 = 76.3%
- 一致性哈希迁移：254/1000 = 25.4%（约 1/N）

### 3.3 虚拟节点（vnode）

节点少时，一致性哈希分布不均（某节点管很大一段环）。虚拟节点解决：
- 每个真实节点在环上放 v 个虚拟节点（如 v=150）
- 虚拟节点均匀撒在环上 → key 分布均匀

```python
class ConsistentHashRing:
    def add_node(self, node):
        for i in range(self._vnodes):  # 150 个虚拟节点
            slot = self._hash(f"{node}#{i}")
            bisect.insort(self._ring, slot)  # 有序数组
            self._slot_to_node[slot] = node

    def get_node(self, key):
        slot = self._hash(key)
        pos = bisect.bisect_right(self._ring, slot)  # 二分找顺时针下一个
        if pos == len(self._ring):
            pos = 0  # 环绕
        return self._slot_to_node[self._ring[pos]]
```

### 3.4 实现：有序数组 + 二分查找

环用有序数组存虚拟节点槽位，`bisect` 二分查找 O(log(V×N))，V=虚拟节点数，N=节点数。

- `add_node`：插入 v 个槽位，每个 O(V×N)（insort 需要移动）
- `remove_node`：删除 v 个槽位
- `get_node`：O(log(V×N)) 二分

### 3.5 复杂度

| 操作 | 复杂度 |
|------|--------|
| add_node | O(V × V×N)（insort 移动） |
| remove_node | O(V × V×N) |
| get_node | O(log(V×N)) |

生产级用红黑树/跳表替代有序数组，add/remove 降到 O(V log(V×N))。

### 3.6 生产级实现

- **vLLM**：多 GPU worker 用 Ray + 一致性哈希分发请求
- **Nginx**：`upstream` 一致性哈希模块（`hash $request_uri consistent`）
- **Redis Cluster**：16384 个 slot（虚拟节点），一致性哈希分配
- **Cassandra/DynamoDB**：一致性哈希 + 虚拟节点（v=256）做数据分片

---

## 4. 页表 — PagedAttention 的显存管理

### 4.1 基本概念

页表（page table）是操作系统虚拟内存的核心：逻辑页 → 物理页的映射。
大模型推理的 KV cache 显存管理借用同样的思想，称为 **PagedAttention**。

### 4.2 KV cache 的显存问题

大模型推理时，每个请求的 KV cache 随生成长度增长：

```
请求 1: 生成 200 token → KV cache 占 200 × 2 × n_layer × d_model × sizeof(fp16) 字节
请求 2: 生成 50 token  → KV cache 占 50 × ... 字节
```

**连续分配**（传统做法）：为每个请求预分配 `max_seq_len` 的连续显存。

问题：
- `max_seq_len` 通常设很大（如 2048），但实际生成长度远小于此
- 请求 2 只生成 50 token，却预占 2048 的空间 → 利用率 50/2048 ≈ 2.4%
- 多个请求并发时，显存很快被预分配占满，无法接新请求

### 4.3 分页分配（PagedAttention）

PagedAttention 把显存分成固定大小的小 block（如 16 token/block），按需分配：

```
连续分配:  请求1 [████████████████████████████████████] 预分配 2048
           请求2 [███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 预分配 2048，只用 50
           → 利用率 ~3%

分页分配:  请求1 [block0][block1][block2]...[block12]  按需 13 个 block
           请求2 [block13]                       按需 1 个 block
           → 利用率 ~88%
```

**页表**：每个序列维护逻辑页 → 物理页的映射，物理页可分散在显存任意位置。

```
序列 1 的页表:
  逻辑页 0 → 物理块 5
  逻辑页 1 → 物理块 12
  逻辑页 2 → 物理块 3
  ...（物理块不连续，但逻辑上连续）
```

### 4.4 实现：物理块池 + 页表 + 空闲链表

```python
class MemoryPaged:
    def __init__(self, num_blocks, block_size):
        self._blocks = {i: Block(i, block_size) for i in range(num_blocks)}
        self._page_table = {}    # seq_id → [物理块 ID 列表]
        self._free_list = set(range(num_blocks))  # 空闲块

    def allocate(self, seq_id, n_tokens):
        n_blocks = ceil(n_tokens / block_size)
        if n_blocks > len(self._free_list):
            return False  # 显存不足
        blocks = [self._free_list.pop() for _ in range(n_blocks)]
        self._page_table[seq_id] = blocks

    def append(self, seq_id, n_new):  # 序列增长时追加块
        ...

    def free(self, seq_id):  # 释放，块归还空闲链表
        for bid in self._page_table[seq_id]:
            self._free_list.add(bid)
```

### 4.5 优势

| 指标 | 连续分配 | 分页分配 |
|------|----------|----------|
| 显存利用率 | ~3-20% | ~80-90% |
| 碎片 | 外碎片严重 | 无外碎片（块固定大小） |
| 并发数 | 受预分配限制 | 按需分配，并发更多 |
| 增长 | 需重新分配整个序列 | 追加一个块即可 |
| 共享 | 不支持 | 可共享 prefix block |

本 demo 实测（256 blocks × 16 tokens，200 请求，生成长度 16-128）：
- 连续分配利用率：12.6%
- 分页分配利用率：88.55%（7x 提升）
- 分页碎片率：20.3%（可接受，块内碎片）

### 4.6 复杂度

| 操作 | 复杂度 |
|------|--------|
| allocate(n) | O(n / block_size) |
| append(n) | O(n / block_size) |
| free | O(页数) |
| 访问 token i | O(1)（页表查物理块 + 块内偏移） |

### 4.7 生产级实现

- **vLLM**：PagedAttention 的提出者，block_size=16，GPU kernel 直接用页表寻址
- **TensorRT-LLM**：类似 PagedAttention 的显存管理，支持 in-flight batching
- **TGI**（Text Generation Inference）：HuggingFace 的推理服务，分页 KV cache
- **操作系统**：虚拟内存的页表 + TLB + 缺页中断，PagedAttention 借鉴此设计

---

## 5. 四种数据结构如何协作

### 5.1 推理服务的完整流程

```
推理请求到达
  │
  ▼
[堆/优先队列]  请求调度：按 SLA/优先级 pop，高优先级先服务
  │             对比 FIFO：SLA 违约率显著降低
  ▼
[一致性哈希环] 负载均衡：分发到多 GPU，节点增减时迁移量小
  │             对比取模哈希：迁移量从 ~75% 降到 ~25%
  ▼
[计算图]       算子融合：合并相邻 element-wise 算子，减少 kernel launch
  │             每 block kernel 数从 14 降到 11
  ▼
[页表]         显存管理：PagedAttention 分块分配 KV cache
  │             对比连续分配：利用率从 ~13% 提升到 ~89%
  ▼
输出 token → 返回客户端
```

### 5.2 协作的必要性

四种数据结构解决推理服务的四个不同层面问题：

| 层面 | 数据结构 | 解决的问题 |
|------|----------|------------|
| 调度层 | 堆 | 谁先服务？（优先级） |
| 分发层 | 一致性哈希 | 发到哪个 GPU？（负载均衡） |
| 计算层 | 计算图 | 怎么算得快？（算子融合） |
| 存储层 | 页表 | 显存怎么用？（分页管理） |

缺少任何一层都会成为瓶颈：
- 无堆：VIP 请求被批处理阻塞，SLA 违约
- 无一致性哈希：扩缩容时大量请求重路由，缓存失效
- 无算子融合：kernel launch 开销大，小 batch 时 GPU 利用率低
- 无页表：显存利用率低，并发请求数少

### 5.3 数据结构间的信息流动

```
请求 → [堆] pop 出最高优先级请求
         │
         ▼ 请求
       [一致性哈希] get_node(req_id) → 分配到某 GPU
         │
         ▼ (GPU, 请求)
       [计算图] 该 GPU 上的模型图做算子融合，生成优化后的 kernel 序列
         │
         ▼ (kernel 序列, 请求)
       [页表] 为请求的 KV cache 分配物理块，kernel 执行时通过页表寻址
         │
         ▼
       输出 token
```

---

## 6. 小结

本章四种数据结构在大模型推理服务中协作：

1. **计算图（DAG）**：算子融合的载体，拓扑序贪心合并 element-wise 算子，减少 kernel launch
2. **堆（优先队列）**：请求调度的核心，O(log n) push/pop，按 SLA 优先级服务
3. **一致性哈希环**：负载均衡的低迁移方案，虚拟节点保证均匀，节点增减时迁移量 O(K/N)
4. **页表**：显存管理的分页方案，PagedAttention 按需分配 block，利用率从 ~13% 提升到 ~89%

它们分别解决调度、分发、计算、存储四个层面的问题，缺少任何一层都会成为推理服务的瓶颈。
生产级的 vLLM、TensorRT-LLM、Triton 都在这四个方向上做了工程化实现。