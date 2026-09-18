# 哈希表的 AI 应用：词嵌入查找 + MoE 路由

> 本文档讲清「哈希表在 AI 里干什么」。原理见 `principle.md`。核心两个应用：词嵌入查找（为什么 embedding 层需要 $O(1)$）和 MoE 路由（哈希把 token 分发到专家）。

## 1. 词嵌入层：AI 里最大的哈希表

### 1.1 什么是词嵌入

神经网络不能直接吃字符串，要把词变成向量。**词嵌入**（Word Embedding）就是给每个词一个 $d$ 维向量：

```
"猫"   → [0.12, -0.34, 0.56, ..., 0.78]   (d=128 维)
"狗"   → [0.15, -0.30, 0.60, ..., 0.75]
"汽车" → [-0.80, 0.22, -0.11, ..., 0.03]
```

语义相近的词向量也相近（"猫"和"狗"的余弦相似度高，"猫"和"汽车"低）。这是所有 NLP 模型（Word2Vec、GPT、BERT、LLaMA）的基础。

### 1.2 词嵌入层的本质：一张大查找表

词嵌入层在数学上就是一个矩阵 $E \in \mathbb{R}^{V \times d}$：
- $V$ = 词表大小（vocabulary size）
- $d$ = 嵌入维度（embedding dim）

```
词表 V = 100000, 维度 d = 128

E = [
  [0.12, -0.34, ...],   ← 词 0 ("猫") 的向量
  [0.15, -0.30, ...],   ← 词 1 ("狗") 的向量
  ...                   ← 共 100000 行
  [-0.80, 0.22, ...],   ← 词 99999 的向量
]
```

给定一个词，embedding 层做的事就是：**词 → 索引 → 矩阵那一行**。

```
forward("猫") = E[ hash("猫") ]   // 查表，O(1)
```

这就是一个**哈希表查找**。

### 1.3 为什么必须是 O(1)

现代 LLM 的规模：

| 模型 | 词表大小 V | 维度 d | 参数量 |
|---|---|---|---|
| GPT-2 | 50,257 | 768/1280 | 117M/1.5B |
| LLaMA-2 | 32,000 | 4096 | 7B |
| GPT-3 | 50,257 | 12288 | 175B |
| LLaMA-3 | 128,256 | 8192 | 405B |
| GPT-4 | ~100,000 | ~12288 | ~1.8T |

一次 forward pass 要处理一个序列（如 2048 个 token），每个 token 都要查一次 embedding 表。训练时一个 batch（如 32 个序列）就是 $32 \times 2048 = 65536$ 次查找。

如果用 $O(\log n)$ 二分：$65536 \times \log_2(100000) \approx 65536 \times 17 \approx 1.1M$ 次比较。
如果用 $O(1)$ 哈希：$65536$ 次查找。

而且这只是**一次 forward**。训练一个 epoch 有几百万次 forward，推理每生成一个 token 也要查一次。$O(\log n)$ vs $O(1)$ 的差距乘以几百万次，就是小时级 vs 分钟级。

### 1.4 实测：哈希 vs 二分

本目录 `python/demo.py` 实测结果（词表 10 万，每次查 10000 个词）：

```
词表大小 n = 100000
实现              均值(ms)    标准差(ms)   加速比
hash O(1)         1.1782      0.1029       5.61x
binary O(log n)   6.6056      1.1815       1.00x
```

**哈希比二分快 5.6 倍**。而且词表越大优势越大：

```
n=1000:    hash 0.59ms, binary 3.35ms, 5.67x
n=10000:   hash 0.78ms, binary 4.07ms, 5.19x
n=100000:  hash 1.18ms, binary 6.61ms, 5.61x
```

- 哈希查找耗时近似常数（0.59 → 1.18，增长慢，主要是 cache miss 随表变大增加）
- 二分查找随 $\log_2(n)$ 增长（3.35 → 6.61，约 2x，对应 $\log_2(100000)/\log_2(1000) \approx 1.67$）

### 1.5 为什么不用排序数组 + 二分？

理论上排序数组 + 二分也能查，而且空间更紧凑（连续存储，cache 友好）。但实际不用，原因：

1. **速度**：$O(1)$ 就是比 $O(\log n)$ 快，5x 实测差距
2. **插入**：词表会更新（新词、OOV 处理），哈希 $O(1)$ 插入，排序数组 $O(n)$ 插入
3. **哈希天然适配任意键类型**：字符串、子词（subword）、字节对都能哈希
4. **PyTorch/TF 内部就是哈希**：`nn.Embedding` 的 `forward` 实质是 `E[idx]`，而 `idx` 来自 tokenizer 的哈希查找

### 1.6 PyTorch nn.Embedding 内部实现

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=100000, embedding_dim=128)
token_ids = torch.tensor([42, 7, 99999])  # tokenizer 输出的索引
output = embedding(token_ids)  # 内部: E[token_ids], 即 gather 操作
```

`nn.Embedding` 的 forward 本质是 `torch.gather`，从大矩阵按索引取行。而**索引从哪来**？tokenizer 把字符串变成 int 索引，tokenizer 内部就是一个哈希表（Python `dict` 或 C++ `unordered_map`）：

```
"猫" → tokenizer.vocab["猫"] → 42 → embedding.weight[42]
         ↑ 哈希表 O(1) 查找        ↑ 矩阵行取，O(1)
```

整条链路全靠哈希表保证 $O(1)$。

## 2. MoE 路由：哈希分发 token 到专家

### 2.1 什么是 MoE

**混合专家**（Mixture of Experts, MoE）是稀疏激活技术：不把每个 token 送进所有参数，而是**只激活一部分专家**。

```
传统稠密模型:  每个 token → 所有参数（全算）
MoE 模型:     每个 token → top-k 个专家（只算这几个）
```

例如 Mixtral 8×7B：8 个专家，每个 token 路由到 top-2 专家。总参数 47B，但每次只激活 13B（2/8），推理成本接近 13B 稠密模型。

### 2.2 MoE 路由的核心问题

每个 token 要决定**送哪几个专家**。方案：

| 路由策略 | 做法 | 问题 |
|---|---|---|
| 学到的 gating | `gate(token) = softmax(W·token)`，取 top-k | 训练时专家负载不均，需要 aux loss |
| **哈希路由** | `expert = hash(token) % n_experts` | 确定性、均匀、无需训练 |

**哈希路由**是 MoE 的简化方案（Switch Transformer、GShard 都讨论过）：
- 优点：天然均匀（哈希函数均匀分布），无需辅助损失，无需负载均衡
- 缺点：不感知语义（同义词可能路由到不同专家），表达能力弱于学到的 gating
- 适用：推理加速、负载均衡要求高、训练成本敏感场景

### 2.3 哈希路由实现

本目录 `demo.py` 的 `moe_route`：

```python
def moe_route(token: str, n_experts: int = 8, top_k: int = 2) -> list[int]:
    h = hash(token) & 0xFFFFFFFFFFFFFFFF  # 64 位哈希
    experts = []
    shift = 0
    while len(experts) < top_k:
        e = (h >> shift) % n_experts      # 取不同比特段 mod 专家数
        if e not in experts:
            experts.append(e)
        shift += 8
    return experts
```

策略：对 token 哈希一次，取哈希值的不同 8 比特段分别 mod 8，得到 top-2 个不同专家。

```
tok_000000 → hash=... → 取 bit[0:8] % 8 = 5, bit[8:16] % 8 = 2 → 专家 [5, 2]
tok_000001 → hash=... → 取 bit[0:8] % 8 = 1, bit[8:16] % 8 = 7 → 专家 [1, 7]
```

### 2.4 负载均匀性验证

10 万个 token 路由到 8 个专家（top-2），实测负载分布：

```
专家         负载      偏差%
expert_0    25304      1.22%
expert_1    24715     -1.14%
expert_2    25106      0.42%
expert_3    24998     -0.01%
expert_4    24996     -0.02%
expert_5    25139      0.56%
expert_6    24784     -0.86%
expert_7    24958     -0.17%

最大偏差 = 1.22%
```

理想均匀负载 = 100000 × 2 / 8 = 25000。实际偏差 < 1.3%，**哈希路由天然均匀**。

对比学到的 gating：不加辅助损失时，常出现"赢家通吃"（几个专家被路由 80%+ 的 token，其余闲置），需要额外的 load balancing loss 纠正。哈希路由免去了这个麻烦。

### 2.5 为什么用哈希而不是随机？

路由必须是**确定性的**：
- 训练时：同一个 token 每次路由到同一个专家，否则梯度无法回传
- 推理时：同一个 token 每次结果一致，否则输出不稳定

随机路由每次结果不同，不可用。哈希路由 `hash(token) % n` 是确定性的（同 token 同结果），又均匀（不同 token 哈希值均匀散布），完美满足需求。

### 2.6 生产级 MoE 的路由

实际 MoE（Mixtral、GPT-4 MoE 层）用**学到的 gating network**：

```python
gate_logits = W_gate @ token        # [n_experts]
weights, experts = topk(softmax(gate_logits), k=2)  # top-2
output = sum(weights[i] * experts[i](token) for i in range(2))
```

学到的 gating 比哈希路由表达力强（能让语义相关的 token 路由到同一专家），但：
1. 需要训练 gating 参数
2. 需要辅助损失防负载不均
3. 推理时多一次矩阵乘（gating 计算）

哈希路由是 MoE 的**最简形式**，适合理解原理和某些推理加速场景（如固定路由的 MoE 推理引擎）。

## 3. demo 结果解读

运行 `python 05_hash_table/python/demo.py` 的完整输出：

### 3.1 查找性能对比

```
[1] 不同词表大小下的查找性能
词表大小 n = 100000
hash O(1)         1.18 ms    5.61x
binary O(log n)   6.61 ms    1.00x
```

读法：查 10000 个词，哈希 1.18ms，二分 6.61ms，哈希快 5.61 倍。

### 3.2 复杂度趋势

```
[2] 复杂度趋势
      n     hash_ms  binary_ms   ratio   log2(n)
   1000      0.59      3.35      5.67x    9.97
  10000      0.78      4.07      5.19x   13.29
 100000      1.18      6.61      5.61x   16.61
```

- `hash_ms` 增长慢：$O(1)$ 理论常数，实际因 cache miss 随表变大略增
- `binary_ms` 随 `log2(n)` 增长：3.35 → 6.61，比值 1.97x，对应 $\log_2(100000)/\log_2(1000) = 16.61/9.97 = 1.67$（实测略高于理论，因 cache 效应）
- `ratio` 稳定在 5x 左右：哈希的常数优势

### 3.3 MoE 负载分布

```
[3] MoE 路由：8 专家，top-2，10 万 token
最大偏差 = 1.22%
```

10 万 token 哈希路由到 8 专家，每专家分到约 25000 次，偏差 < 1.3%。证明哈希路由均匀。

### 3.4 图

- `figures/lookup_compare_100k.png`：词表 10 万时哈希 vs 二分柱状图
- `figures/lookup_vs_vocab_size.png`：词表 1k/10k/100k 查找耗时折线图
- `figures/moe_expert_load.png`：8 专家负载分布柱状图

## 4. 生产级实现：PyTorch / CUDA 里的哈希

### 4.1 PyTorch nn.Embedding

```python
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input):
        return F.embedding(input, self.weight)  # 本质: self.weight[input]
```

`F.embedding` 底层是 `aten::embedding`，CUDA kernel 里每个线程取一个 `input[i]` 索引，从 `weight` 矩阵 gather 一行。索引到行的映射是**数组直接下标**（$O(1)$），而索引本身来自 tokenizer 的哈希表。

### 4.2 Tokenizer 的哈希表

HuggingFace tokenizers（Rust 实现）用 `std::collections::HashMap`（拉链法变体）存 `vocab: HashMap<String, u32>`：

```rust
let vocab: HashMap<String, u32> = HashMap::new();
vocab.insert("猫".to_string(), 42);
let token_id = vocab["猫"];  // O(1) 哈希查找
```

整条链路：`"猫" → HashMap["猫"] = 42 → embedding.weight[42]`，两步都是 $O(1)$。

### 4.3 CUDA 里的 MoE 路由

生产级 MoE（如 vLLM、TensorRT-LLM 的 MoE kernel）用学到的 gating，但底层分发仍依赖**确定性映射**（token → expert），本质是哈希思想：

```cuda
int expert_id = gating_scores[token_id].argmax();  // 学到的路由
// 或固定路由:
int expert_id = hash(token_id) % n_experts;         // 哈希路由
expert_buffers[expert_id].append(token_id);         // 分发到专家 buffer
```

每个专家有自己的 buffer（队列），token 按路由结果入队，专家并行处理各队列。这里哈希表/哈希函数保证分发是 $O(1)$ 且均匀。

### 4.4 KV Cache 里的哈希

LLM 推理的 KV Cache（见第 11 章 paged_attention）也用哈希：`block_table: HashMap<seq_id, List<Block>>`，按序列 ID $O(1)$ 查到该序列的 KV 块列表。多请求并发时，哈希表让调度器 $O(1)$ 定位每个请求的缓存。

## 5. 哈希表在 AI 里的其他应用

| 应用 | 键 | 值 | 为什么用哈希 |
|---|---|---|---|
| **词嵌入查找** | token 字符串 | embedding 索引 | $O(1)$ 查表，10 万词表 |
| **MoE 路由** | token ID | 专家 ID | 确定性均匀分发 |
| **KV Cache 管理** | 序列 ID | 缓存块列表 | 多请求 $O(1)$ 定位 |
| **参数索引** | 参数名 | 参数张量 | 模型 `state_dict` 就是哈希表 |
| **梯度累积** | 参数名 | 梯度 | 反向传播按名查梯度 |
| **数据去重** | 样本哈希 | 是否见过 | 训练数据去重 $O(1)$ 判重 |
| **特征哈希** | 原始特征 | 哈希桶 | 大规模稀疏特征（FM/FFM） |
| **分布式路由** | 请求 key | 节点 ID | 一致性哈希，分布式推理 |

### 5.1 特征哈希（Feature Hashing）

推荐系统里高维稀疏特征（如用户 ID、商品 ID，维度上亿）直接存不现实。**特征哈希**把高维特征哈希到固定维度：

```python
def feature_hash(raw_feature: str, n_buckets: int = 1_000_000) -> int:
    return hash(raw_feature) % n_buckets
```

原维度 1 亿 → 哈希到 100 万桶，空间降 100x，代价是冲突（多个特征共享桶，引入噪声）。这是哈希表思想在降维上的应用。

### 5.2 一致性哈希（分布式推理）

多 GPU/多节点部署 LLM 时，请求按 key 路由到节点。普通 `hash(key) % n_nodes` 在节点增减时大量 key 重映射，缓存全失效。**一致性哈希**把节点和 key 都映射到环上，节点增减只影响相邻段，最小化重映射。这是哈希表在分布式系统的扩展。

## 6. 性能优化方向

### 6.1 开放地址法替代拉链法

CPU 密集场景（如 CPU 推理的 embedding 查找），开放地址法缓存友好，比拉链法快 2-3x。Google `absl::flat_hash_map` 用开放地址法，比 `std::unordered_map`（拉链）快数倍。

### 6.2 完美哈希

如果键集合**固定且已知**（如词表训练好不再变），可用**完美哈希**（perfect hash）：构造一个无冲突的哈希函数，$O(1)$ 查找且无冲突开销。工具如 `gperf`、`frozen` 生成完美哈希。词表推理可用。

### 6.3 布隆过滤器前置

查 embedding 前先用布隆过滤器（多个哈希函数的位数组）判 key 是否存在，不存在直接返回 OOV，避免查大表。布隆过滤器本身也是哈希的衍生结构。

## 7. 小结

哈希表在 AI 里的价值一句话：**把 $O(\log n)$ 的查找变成 $O(1)$，在词表 10 万、token 数百万次的规模下，省下的是数量级的时间**。

两个核心应用：
1. **词嵌入查找**：tokenizer 哈希表 + embedding 矩阵 gather，全链路 $O(1)$
2. **MoE 路由**：哈希把 token 确定性均匀分发到专家，免去学到的 gating 的负载均衡难题

理解了哈希表的 $O(1)$ 和均匀分布特性，就理解了为什么 AI 框架从 tokenizer 到 MoE 到 KV Cache 全程都在用哈希。