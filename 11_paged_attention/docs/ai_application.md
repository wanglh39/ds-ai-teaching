# PagedAttention：vLLM 的 KV Cache 分页管理

> 本文档讲清「分页 KV Cache 在 AI 里干什么」。原理见 `principle.md`。核心应用：**vLLM 的 PagedAttention**——把操作系统的分页机制搬到 KV Cache 管理，把显存利用率从 30-60% 提升到 95%+，让 LLM 推理吞吐量翻倍。这是 2023 年 LLM 推理系统最重要的工程创新之一。

## 1. KV Cache 是什么

### 1.1 Transformer 的自回归推理

大模型（GPT、LLaMA、Qwen 等）生成文本是**自回归**的：每次生成一个 token，把它拼到输入上再生成下一个。每生成一个 token，要对之前所有 token 计算 attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

其中 $Q$ 是当前 token 的查询，$K/V$ 是**所有历史 token** 的键值。如果每步都重算所有历史 token 的 K、V，计算量随长度平方增长，极慢。

### 1.2 缓存 K 和 V

**KV Cache** 把每层的 K、V 缓存起来，下一步直接复用：

```
生成 token 5 时:
  Q = token 5 的 query
  K = [K_1, K_2, K_3, K_4, K_5]   ← 前 4 个从缓存读，第 5 个新算
  V = [V_1, V_2, V_3, V_4, V_5]   ← 同上
  attention = softmax(Q · K^T / √d) · V
  缓存追加: K_cache.append(K_5), V_cache.append(V_5)
```

每层每个 token 存一对 (K, V)。模型有 L 层、hidden_dim = d、KV head 数 = h_kv、head_dim = d/h，则**每个 token 的 KV Cache 大小**：

```
2 (K和V) × L (层) × h_kv (KV头数) × head_dim (每头维度) × dtype_size
```

以 LLaMA-7B (fp16) 为例：L=32, h_kv=32, head_dim=128，每 token KV Cache = 2 × 32 × 32 × 128 × 2 = 524288 字节 = 512 KB。**一个 token 半兆**。

### 1.3 显存瓶颈

序列越长、batch 越大，KV Cache 占的显存越多：

| 序列长度 | batch=1 | batch=32 | batch=64 |
|---|---|---|---|
| 512 | 256 MB | 8 GB | 16 GB |
| 2048 | 1 GB | 32 GB | 64 GB |
| 8192 | 4 GB | 128 GB | 256 GB |

LLaMA-7B 模型权重才 13 GB，但 batch=64、序列 2048 的 KV Cache 就要 64 GB——**KV Cache 比模型权重还大**。A100 80GB 也放不下。

**结论**：KV Cache 是长序列、大 batch 推理的显存瓶颈。怎么高效管理它，直接决定吞吐量。

## 2. 朴素连续分配的碎片化

### 2.1 朴素方案

早期 LLM 推理框架（如原始 vLLM 之前）给每个序列**连续分配**一段显存放 KV Cache：

```
显存: [序列1 ████████░░░░░░░░] [序列2 ████░░░░░░░░░░░░] [序列3 ████████████░░]
       预留 max_len=2048        预留 max_len=2048        预留 max_len=2048
       实际用了 512              实际用了 256              实际用了 768
```

每个序列预留 `max_seq_len` 的连续空间（因为生成时长度会增长，连续分配要预留余量）。

### 2.2 内部碎片

序列实际长度通常远小于 `max_seq_len`（用户输入 50 token，max_len 设 2048），预留空间大量浪费：

```
预留: 2048 slot
实际: 50 token
浪费: 1998 slot (97.5%)
利用率: 50/2048 = 2.4%
```

实测朴素预分配的显存利用率只有 **20-40%**——大部分显存预留了但没用。

### 2.3 外部碎片

如果改成动态连续分配（不预留 max_len，按实际长度分配），释放后产生**外部碎片**：

```
初始: [空闲 4096]
分配 seq1 (1000): [seq1 1000][空闲 3096]
分配 seq2 (1500): [seq1 1000][seq2 1500][空闲 1596]
分配 seq3 (800):  [seq1 1000][seq2 1500][seq3 800][空闲 796]
释放 seq1:        [空闲 1000][seq2 1500][seq3 800][空闲 796]
分配 seq4 (1200): 失败！没有连续 1200 的空闲区间（虽有 1000+796=1796 总空闲，但不连续）
```

序列长度可变 + 连续分配 = 必然产生外部碎片。长序列可能因找不到连续区间被拒，即使总空闲足够。

### 2.4 共享前缀的困难

多个序列常有相同前缀（system prompt、few-shot examples）。连续分配下，每个序列各存一份前缀的 KV，无法共享——因为前缀在不同序列的连续区间里，物理位置不同。

```
序列 A: [前缀 KV][A 的独有 KV]    前缀存了一份
序列 B: [前缀 KV][B 的独有 KV]    前缀又存了一份（完全相同的数据！）
```

长前缀（如 2000 token 的 system prompt）× 32 个序列 = 64000 token 的重复 KV，浪费巨大。

### 2.5 扩容的困难

连续分配下，序列生成时长度增长，但连续区间可能不够。要么预留 max_len（浪费），要么搬移到更大的区间（复制开销大、可能找不到）。

## 3. PagedAttention 的核心思路

PagedAttention（vLLM, 2023）把**操作系统的分页机制**搬到 KV Cache 管理，一举解决上述所有问题。三个核心创新：

### 3.1 分页：固定块 + 按需分配

显存切成等大的**块**（block，vLLM 默认 16 token/块）。序列的 KV Cache 按**块**分配，不按连续区间：

```
显存块池: [块0][块1][块2][块3][块4][块5][块6][块7]...

序列 A (10 token, block_size=4): 分配 3 个块
  页表 A: [逻辑页0→块2][逻辑页1→块5][逻辑页2→块1]   (物理块不连续！)
  逻辑:   [token 0-3][token 4-7][token 8-9]
  物理:   块2        块5        块1

序列 B (6 token): 分配 2 个块
  页表 B: [逻辑页0→块3][逻辑页1→块7]
```

- **无外部碎片**：任何空闲块都能用，不用找连续区间
- **按需分配**：序列生成新 token 才申请新块，不预留 max_len
- **内部碎片小**：每序列最后一块可能未满，最多浪费 block_size - 1 = 15 slot

### 3.2 页表：逻辑连续，物理分散

每个序列一张**页表**（block table），逻辑页号 → 物理块号。序列看到的是连续的逻辑 token 位置，物理块可以分散在各处：

```
序列 A 的逻辑视图:  [token 0][token 1]...[token 9]   (连续)
序列 A 的物理布局:  块2[0-3] 块5[4-7] 块1[8-9]       (分散)

attention 计算时:
  for token t in 序列A:
    page = t / block_size
    offset = t % block_size
    physical_block = 页表A[page]        ← 查页表
    K[t] = KV_cache[physical_block][offset].K
    V[t] = KV_cache[physical_block][offset].V
```

GPU kernel 里每次访问 KV 要查一次页表。这是 PagedAttention 的核心修改：把 attention kernel 里的连续内存访问改成**分页访问**——查页表得到物理块，再访问块内 offset。

### 3.3 块复用 + 引用计数：共享前缀

多个序列的页表可以指向**同一物理块**，用引用计数管理：

```
序列 A: "请翻译：Hello"     页表 A: [块0][块1][块2][块3]
序列 B: "请翻译：World"     页表 B: [块0][块1][块4][块5]
                                    ↑↑↑↑ 共享前缀"请翻译："（2 个块）
  ref_count[块0] = 2
  ref_count[块1] = 2
  ref_count[块2] = 1 (A 独有)
  ref_count[块3] = 1 (A 独有)
  ref_count[块4] = 1 (B 独有)
  ref_count[块5] = 1 (B 独有)

前缀 "请翻译：" 的 KV 只存一份（在块0、块1），两个序列共享。
省了 2 块 × 16 token × 512 KB/token = 16 MB 显存。
```

序列释放时，每块引用减 1，归零才回收。共享前缀不复制数据，只增加引用计数和页表项——O(页数) 而非 O(token 数)。

## 4. 显存利用率对比

### 4.1 朴素 vs 分页的理论利用率

| 方案 | 利用率 | 碎片类型 | 并发数 |
|---|---|---|---|
| 朴素预分配（预留 max_len） | 实际长度 / max_len ≈ 20-40% | 内部（预留未用） | total / max_len |
| 朴素动态连续分配 | 60-80% | 外部（空洞） | 受碎片限制 |
| **PagedAttention** | **95%+** | 内部（块尾，≤block_size-1/序列） | total / 平均长度 |

分页的利用率 = 实际 token / (分配块数 × block_size)。每序列最多浪费 block_size - 1 个 slot，序列越长浪费占比越小。block_size=16、序列长 256 时，浪费 ≤ 15/256 = 5.9%，利用率 ≥ 94%。

### 4.2 demo 实测对比

本 demo 模拟 4096 slot 显存、block_size=16、2000 步随机工作负载，对比三种方案：

```
方案                平均利用率    平均碎片化    最大并发数
朴素预分配           23.07%      72.08%       8
朴素动态分配          77.25%      73.00%       37
PagedAttention      85.09%       5.57%       35
```

- **朴素预分配**：每序列预留 512 slot，最多 8 个序列，利用率 23%（序列平均才 100 多 token）
- **朴素动态分配**：利用率 77%，但碎片化 73%（空闲区间分散，长序列被拒）
- **PagedAttention**：利用率 85%，碎片化仅 5.6%（只有块尾内部碎片），并发 35

### 4.3 不同块大小的利用率

```
block_size=  4: 利用率=89.57%, 碎片化=1.14%
block_size=  8: 利用率=88.02%, 碎片化=2.70%
block_size= 16: 利用率=85.09%, 碎片化=5.57%
block_size= 32: 利用率=80.96%, 碎片化=10.15%
block_size= 64: 利用率=73.89%, 碎片化=17.99%
block_size=128: 利用率=63.29%, 碎片化=29.58%
```

块越小利用率越高（块尾浪费越少），但页表越大、管理开销越多。vLLM 默认 block_size=16 是平衡点。

### 4.4 固定长度场景

当所有序列长度相同且是 block_size 的整数倍时，分页和动态分配利用率相同（93%），但分页碎片化为 0：

```
固定长度 128, block_size=16:
  朴素动态分配: 利用率=93.11%, 碎片化=17.19% (外部碎片)
  PagedAttention: 利用率=93.11%, 碎片化=0.00% (无碎片)
```

分页彻底消除了外部碎片。

## 5. demo 结果解读

### 5.1 利用率随时间变化

demo 的 `kv_cache_usage_over_time.png` 显示三种方案的利用率随模拟步数变化：

- **朴素预分配**：利用率始终在 25% 附近，因为每序列预留 512 但平均只用 100 多
- **朴素动态分配**：利用率 70-85% 波动，释放后碎片导致波动
- **PagedAttention**：利用率稳定在 85-93%，几乎不波动（块尾碎片小且稳定）

### 5.2 并发序列数

`kv_cache_concurrency_bar.png` 显示最大并发数：

- 朴素预分配：8（受限于 total/max_len = 4096/512）
- 朴素动态分配：37（受限于碎片）
- PagedAttention：35（受限于总块数和序列平均长度）

分页的并发数和动态分配接近，但**稳定性高**——动态分配的并发数随碎片波动，分页几乎只受总显存限制。

### 5.3 块大小扫描

`paged_block_size_sweep.png` 显示块大小从 4 到 128 的利用率：

- 块 4：利用率 89.6%，碎片 1.1%——几乎无浪费
- 块 128：利用率 63.3%，碎片 29.6%——每序列浪费半块

块越小越好，但 GPU kernel 对小块的访存效率低（页表查询次数多）。vLLM 选 16 是 GPU 友好性和利用率的权衡。

## 6. vLLM 的生产级实现

### 6.1 整体架构

vLLM 的 PagedAttention 实现分三层：

```
┌─────────────────────────────────┐
│ Scheduler（调度器）              │  决定哪些序列参与本轮推理
├─────────────────────────────────┤
│ BlockSpaceManager（块空间管理器） │  管理块池 + 每序列页表
├─────────────────────────────────┤
│ PagedAttention Kernel（GPU 核）  │  分页访存的 attention 计算
└─────────────────────────────────┘
```

- **Scheduler**：每轮从等待队列选序列组成 batch，考虑显存余量
- **BlockSpaceManager**：本 C 实现的 Python 版，管理块池和页表
- **Kernel**：修改过的 attention GPU kernel，按页表分页访存

### 6.2 Block Table

vLLM 给每个序列维护一个 **block table**（就是页表），在 GPU 上是一个整数数组：

```python
# 序列 A 的 block table（逻辑页 → 物理块）
block_table_A = [2, 5, 1, 8, 3]   # 逻辑页 0→块2, 1→块5, 2→块1, ...

# attention kernel 访问 token t 的 K:
page = t // block_size
offset = t % block_size
physical_block = block_table_A[page]
K_t = KV_cache[physical_block, offset]
```

block table 在 GPU 上，kernel 直接读取。每次序列增长（申请新块），更新 block table。

### 6.3 Prefix Caching

vLLM 的 prefix caching 利用引用计数共享前缀：

```
请求 1: "系统提示 + 用户问题 1"
请求 2: "系统提示 + 用户问题 2"

1. 请求 1 来了: 分配块存 "系统提示" + "用户问题 1"
2. 请求 2 来了: 发现 "系统提示" 已有块（hash 匹配）
   → 直接让请求 2 的页表指向这些块（引用计数 +1）
   → 不重新计算 "系统提示" 的 KV
   → 只分配新块存 "用户问题 2" 的 KV
```

vLLM 用 **hash of block content** 标识块，新请求的前缀块 hash 匹配已有块就直接复用。前缀越长省的越多——2000 token 的 system prompt × 32 个请求，省 32 倍前缀计算和显存。

### 6.4 Copy-on-Write（CoW）

共享块后，如果序列要**修改**块（如继续生成写最后一块的空 slot），而块被共享（ref_count > 1），就要 CoW：

```
块 5 被 seq A 和 B 共享 (ref_count=2)
seq A 要在块 5 写新 token:
  1. 分配新块 5'
  2. 复制块 5 的内容到块 5'（CoW）
  3. seq A 的页表: 块5 → 块5'
  4. ref_count[块5]-- (变 1, B 独占)
  5. ref_count[块5'] = 1 (A 独占)
  6. seq A 在块 5' 写新 token
```

CoW 保证共享块不被单方修改破坏。本 C 实现简化了 CoW（假设共享块只读），vLLM 完整实现了 CoW 逻辑。

### 6.5 GPU Kernel 的分页访存

标准 attention kernel 假设 KV 连续：

```cuda
// 标准 attention（连续访存）
for (int t = 0; t < seq_len; t++) {
    K_t = KV_cache[seq_offset + t];   // 连续地址
    ...
}
```

PagedAttention kernel 改成分页访存：

```cuda
// PagedAttention（分页访存）
for (int t = 0; t < seq_len; t++) {
    int page = t / block_size;
    int offset = t % block_size;
    int physical_block = block_table[page];   // 查页表
    K_t = KV_cache[physical_block * block_size + offset];  // 分页地址
    ...
}
```

每次访存多一次页表查询（block_table 是 GPU 上的小数组，cache 友好）。实测这个开销 < 5%，远小于碎片化带来的浪费。

### 6.6 调度与抢占

显存不够时，vLLM 的 scheduler 会**抢占**低优先级序列：

```
显存满，新请求来了:
  1. 选最近最少使用的序列 (LRU)
  2. 把它的 KV Cache 块释放（或换出到 CPU）
  3. 腾出块给新请求
  4. 被抢占的序列之后重新调度时，重算被换出的 KV
```

分页让抢占极快——释放几个块即可，不用搬移连续区间。连续分配下抢占要搬移或复制大段显存，开销巨大。

## 7. 与传统内存管理的类比

PagedAttention 和 OS 分页的对应关系：

| OS 分页 | PagedAttention | 说明 |
|---|---|---|
| 物理页框 | KV Cache 块 | 显存的最小分配单位 |
| 页大小（4KB） | block_size（16 token） | 固定大小，权衡碎片和管理开销 |
| 进程页表 | 序列 block table | 逻辑页号 → 物理块号 |
| 空闲页框链表 | 空闲块池 | O(1) 分配/回收 |
| 缺页中断 | 按需分配新块 | 序列增长时申请新块 |
| 共享内存页 | prefix caching | 多序列共享前缀块 |
| 引用计数 | 块引用计数 | 归零才回收 |
| 写时复制 | CoW | 共享块被修改时复制 |
| 交换（swap） | 换出到 CPU | 显存不够时把 KV 换到 CPU 内存 |
| LRU 淘汰 | 序列抢占 | 显存不够时释放低优先级序列 |
| 虚拟地址 | 逻辑 token 位置 | 序列看到连续的逻辑位置 |
| 物理地址 | 物理块 + 偏移 | 实际显存位置 |

**PagedAttention 本质上就是把 OS 的内存管理子系统搬到 GPU 显存管理 KV Cache**。vLLM 论文的标题就是 "Efficient Memory Management for Large Language Model Serving with PagedAttention"——明确把这是定位为内存管理创新。

## 8. 实际效果

### 8.1 vLLM 的吞吐量提升

vLLM 论文报告的吞吐量对比（vs TGI、vLLM 朴素版）：

| 模型 | 朴素 | PagedAttention | 提升 |
|---|---|---|---|
| LLaMA-7B | 1.0x | 2-4x | 2-4 倍 |
| LLaMA-13B | 1.0x | 2-4x | 2-4 倍 |
| LLaMA-33B | 1.0x | 2-3x | 2-3 倍 |

提升主要来自：

1. **显存利用率提升**：30-60% → 95%+，同样显存能容纳更多序列
2. **并发数提升**：batch size 从几十提升到上百
3. **prefix caching**：共享前缀省的计算和显存
4. **无碎片**：序列不会被因碎片拒绝，调度更灵活

### 8.2 行业影响

PagedAttention 之后，主流 LLM 推理框架都采用了分页 KV Cache：

- **vLLM**：原创实现，block_size=16
- **TensorRT-LLM**：NVIDIA 的实现，类似分页
- **SGLang**：在 PagedAttention 基础上进一步优化 prefix caching
- **LMDeploy**：商汤的实现，分页 + 其他优化
- **TGI**：HuggingFace 的实现，也跟进分页

PagedAttention 成了 LLM 推理系统的**标配**，就像分页成了 OS 的标配。

## 9. 局限与演进

### 9.1 块大小权衡

块太小：页表大、kernel 访存次数多、管理开销大。块太大：内部碎片多、利用率下降。vLLM 默认 16 是经验值，不同模型/负载可能最优值不同。

### 9.2 页表访存开销

每次 attention 访存要多一次页表查询。虽然 block table 在 GPU cache 里，但仍是非零开销。一些优化：

- **block table 预取**：kernel 开始时预取整个 block table 到 shared memory
- **大块**：增大 block_size 减少页表项（但增加碎片）
- **连续块优化**：如果序列的块恰好连续，用连续访存路径

### 9.3 跨序列共享的 hash 开销

prefix caching 要 hash 块内容来匹配前缀。hash 计算和比较有开销，且 hash 冲突可能误共享。vLLM 用 rolling hash 平衡速度和准确性。

### 9.4 演进方向

- **多级分页**：类似 OS 的多级页表，减少页表内存占用
- **块大小自适应**：根据负载动态调整 block_size
- **跨 GPU 分页**：多 GPU 时把块分布在不同 GPU，类似分布式共享内存
- **KV Cache 量化 + 分页**：块内量化压缩，进一步省显存

## 10. 总结

PagedAttention 的核心贡献是把**操作系统的分页机制**搬到 KV Cache 管理：

1. **分页消除外部碎片**：固定块 + 空闲池，任何块都能分配，不用找连续区间
2. **按需分配消除内部碎片**：序列增长才申请新块，不预留 max_len，利用率 95%+
3. **页表解耦逻辑和物理**：序列看连续逻辑位置，物理块分散各处
4. **引用计数支持共享**：多序列共享前缀块，prefix caching 省计算和显存
5. **CoW 保证正确性**：共享块被修改时复制，不破坏其他序列

效果：显存利用率从 30-60% 提升到 95%+，吞吐量提升 2-4 倍。这是把经典数据结构思想（分页内存管理）应用到新领域（LLM 推理）的成功范例——OS 几十年的内存管理智慧，在 GPU 显存管理上同样有效。