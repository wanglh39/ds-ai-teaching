# 第 18 章 · 模型部署与推理服务 — AI 应用

> AI 场景：大模型（LLM）部署推理服务的全链路优化。
> 数据结构：计算图（算子融合）+ 堆（请求调度）+ 一致性哈希（负载均衡）+ 页表（显存管理）协作。

---

## 1. 大模型推理服务的挑战

### 1.1 问题：推理比训练更难服务

大模型训练是**离线批处理**：固定 batch、固定序列长度、不在乎延迟。
推理服务是**在线交互**：流式请求、变长生成、毫秒级延迟要求。

```
训练:  一个大 batch → 算几十秒 → 没人等
推理:  用户发请求 → 期望 100ms 内出第一个 token → 等不起
```

推理服务的四大挑战：

| 挑战 | 说明 | 本章数据结构 |
|------|------|-------------|
| 算子开销 | 几百个算子，kernel launch 开销累积 | 计算图（算子融合） |
| 请求调度 | 混合优先级，SLA 保障 | 堆（优先队列） |
| 负载均衡 | 多 GPU 分发，扩缩容 | 一致性哈希 |
| 显存管理 | KV cache 按需分配 | 页表（PagedAttention） |

### 1.2 为什么 GPU 利用率低？

大模型推理时 GPU 利用率常只有 30-50%，原因：

- **kernel launch 开销**：每个算子 launch 一次 kernel，固定开销 ~5-10 μs。
  小 batch 时，计算时间可能还没 launch 开销长。
- **显存带宽瓶颈**：中间张量写回 HBM 再读出，带宽成为瓶颈。
- **显存浪费**：连续分配预留 max_seq_len，实际利用率低，并发数上不去。
- **请求堆积**：FIFO 调度不区分优先级，VIP 请求被阻塞。

本章四种数据结构分别解决这四个问题。

---

## 2. 算子融合：减少 kernel launch

### 2.1 问题：kernel launch 开销

大模型的一个 Transformer block 有 ~14 个算子，12 层就是 ~168 个算子。
每个算子 launch 一个 GPU kernel：

```
168 个 kernel × 7 μs (launch 开销) = 1.18 ms 纯 launch 开销
```

如果 batch=1（单请求），每个 kernel 的计算时间可能只有几十 μs，
launch 开销占比可达 20-30%。

### 2.2 解决：计算图上合并相邻算子

算子融合在计算图上把相邻的 element-wise 算子合并成一个 kernel：

```
融合前:  MatMul → GELU → MatMul    (3 个 kernel, 2 个中间张量)
融合后:  MatMul → GELU             (2 个 kernel, GELU 融合进 MatMul)
         (GELU 在 MatMul kernel 末尾直接算，中间结果不写回 HBM)
```

融合的收益：
- **减少 kernel 数**：从 168 降到 132（减少 21%）
- **减少中间张量读写**：element-wise 结果留在寄存器，不写回 HBM
- **降低延迟**：尤其小 batch 时，launch 开销占比大，融合收益明显

### 2.3 可融合的算子类型

| 算子类型 | 例子 | 可融合？ |
|----------|------|----------|
| element-wise | Add, ReLU, Scale, GELU | 是（互相可融合） |
| matmul | MatMul, Linear | 与后续 element-wise 可融合 |
| reduction | Softmax, LayerNorm | 与后续 element-wise 可融合 |
| custom | Attention | 与后续 Scale 可融合 |

本 demo 的融合策略（贪心）：
1. 按拓扑序扫描
2. 当前算子是 element-wise 且只有一个前驱
3. 前驱只有一个消费者 → 融合

每个 Transformer block 融合 3 个算子（Scale、GELU 等），kernel 从 14 降到 11。

### 2.4 生产级融合器

- **XLA**（Google）：TensorFlow/JAX 的编译器，自动融合 element-wise + reduction
- **MLIR/TorchInductor**（PyTorch 2.0+）：基于 MLIR 的图融合 + Triton codegen
- **TensorRT**（NVIDIA）：融合 conv+bn+relu、matmul+gelu、attention+softmax 等
- **Triton**（OpenAI）：DSL 手写 fused kernel，vLLM 的 attention kernel 用 Triton

---

## 3. 请求调度：SLA 保障

### 3.1 问题：混合优先级请求

推理服务面对不同优先级的请求：

```
VIP 用户在线聊天  → 优先级 5, SLA < 100ms
普通用户在线聊天  → 优先级 3, SLA < 200ms
后台批处理        → 优先级 1, 无硬性 SLA
```

FIFO 调度（先到先服务）不区分优先级：一个批处理请求排在 VIP 前面，
VIP 用户等批处理完才能响应，SLA 违约。

### 3.2 解决：堆/优先队列按优先级调度

优先级调度用堆实现：每次 pop 优先级最高的请求。

```
FIFO:       req1(P1) → req2(P5) → req3(P3)  (按到达顺序)
            VIP 的 req2 要等 req1 算完 → 违约

优先级调度:  req2(P5) → req3(P3) → req1(P1)  (按优先级)
            VIP 的 req2 先服务 → 不违约
```

### 3.3 SLA 违约率对比

本 demo 在过载场景下（服务能力略低于请求速率）：

| 调度策略 | SLA 违约数 | 违约率 |
|----------|-----------|--------|
| FIFO | 285/300 | 95.0% |
| 优先级调度 | 219/300 | 73.0% |

优先级调度的违约率更低，且**违约集中在低优先级请求**（可接受），
高优先级请求几乎不违约。

### 3.4 连续批处理（Continuous Batching）

现代推理服务（vLLM）用**连续批处理**：不等一个 batch 全部生成完才接新请求，
而是在每步生成时动态加入/移除请求。

```
传统批处理:  [req1, req2, req3] 全部生成完 → 才接 req4
            req2 生成 200 token, req1 只生成 50 → req1 等 req2 完才释放

连续批处理:  step 1: [req1, req2, req3]
            step 2: [req1, req2, req3]  (req1 还在生成)
            step 3: [req2, req3, req4]  (req1 完了，req4 加入)
            → GPU 不空等，吞吐大幅提升
```

连续批处理 + 优先级调度 = 高吞吐 + 低延迟。

### 3.5 生产级实现

- **vLLM**：连续批处理 + 优先级调度 + preemption（高优先级抢占低优先级的显存）
- **Triton Inference Server**：多模型多队列，按优先级和模型实例调度
- **TensorRT-LLM**：in-flight batching，动态优先级调整
- **SGLang**：基于 RadixAttention 的请求调度，共享 prefix 优化

---

## 4. 负载均衡：多 GPU 分发

### 4.1 问题：多 GPU 分发与扩缩容

大模型推理常需要多 GPU（张量并行 / 流水并行 / 多副本）。请求如何分发到 GPU？

```
请求 → ? → GPU 0 / GPU 1 / GPU 2 / GPU 3
```

简单方案：取模哈希 `hash(req_id) % N`。但扩缩容时 N 变化，几乎所有请求要重路由：

```
3 GPU → 4 GPU:  hash % 3 → hash % 4
  约 75% 的请求映射到不同 GPU → KV cache 全部失效 → 重新计算
```

### 4.2 解决：一致性哈希

一致性哈希把 GPU 和请求都映射到环上，请求顺时针找最近 GPU。
扩缩容时只有相邻区间的请求迁移：

```
3 GPU → 4 GPU:  新 GPU 只接管环上一段区间
  约 25% 的请求迁移（1/4），其余 75% 不动 → KV cache 保留
```

本 demo 实测（3→4 节点，1000 keys）：

| 方案 | 迁移 key 数 | 迁移率 |
|------|------------|--------|
| 取模哈希 | 763/1000 | 76.3% |
| 一致性哈希 | 254/1000 | 25.4% |

一致性哈希迁移量是取模的 1/3。

### 4.3 虚拟节点：解决分布不均

节点少时，一致性哈希分布不均。虚拟节点（vnode）解决：
- 每个真实 GPU 在环上放 150 个虚拟节点
- 虚拟节点均匀撒在环上 → 请求分布均匀

本 demo 4 个 GPU 分发 500 请求，分布 {142, 90, 140, 128}（最大/最小 = 1.58），
虚拟节点数足够时分布更均匀。

### 4.4 生产级实现

- **vLLM**：多 GPU worker 用 Ray + 一致性哈希分发请求
- **Nginx**：`upstream` 一致性哈希模块（`hash $request_uri consistent`）
- **Redis Cluster**：16384 个 slot 做一致性哈希分片
- **Kubernetes**：Service + Endpoints，负载均衡 + 一致性哈希

---

## 5. 显存管理：PagedAttention

### 5.1 问题：KV cache 的显存浪费

大模型推理时，每个请求的 KV cache 随生成长度增长。传统做法：预分配 max_seq_len 的连续显存。

```
max_seq_len = 2048, 实际生成 50 token → 利用率 50/2048 = 2.4%
```

多个请求并发时，显存被预分配占满，无法接新请求：
```
显存 8GB, 每请求预分配 1GB → 最多 8 个并发
即使每请求实际只用 0.1GB → 还是只能 8 个并发
```

### 5.2 解决：分页分配 KV cache

PagedAttention（vLLM 提出）把显存分成固定大小的 block，按需分配：

```
block_size = 16 token
请求生成 50 token → 需要 4 个 block (ceil(50/16)) → 占 64 token 空间
利用率 = 50/64 = 78%
```

页表维护逻辑页 → 物理页映射，物理页可分散在显存任意位置：

```
序列 1 的页表:  逻辑页 0 → 物理块 5
                逻辑页 1 → 物理块 12
                逻辑页 2 → 物理块 3
                (物理块不连续，但 attention kernel 通过页表寻址)
```

### 5.3 利用率对比

本 demo（256 blocks × 16 tokens，200 请求，生成长度 16-128）：

| 方案 | 利用率 | 碎片率 |
|------|--------|--------|
| 连续分配（max=512） | 12.6% | 0%（外碎片严重） |
| 分页分配（block=16） | 88.55% | 20.3%（块内碎片） |

分页分配利用率是连续分配的 **7 倍**。碎片率 20% 是块内碎片（最后一个块没填满），
可接受。

### 5.4 PagedAttention 的额外优势

- **并发数提升**：按需分配，同样显存能服务更多并发请求
- **prefix 共享**：相同 system prompt 的请求共享 prefix block，省显存
- **无需重分配**：序列增长时追加一个 block，不需要重新分配整个序列
- **preemption**：显存不足时，低优先级请求的 block 被换出，高优先级请求继续

### 5.5 生产级实现

- **vLLM**：PagedAttention 的提出者，block_size=16，GPU kernel 直接用页表寻址
  - 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)
- **TensorRT-LLM**：类似 PagedAttention 的显存管理，支持 in-flight batching
- **TGI**（HuggingFace）：分页 KV cache
- **SGLang**：RadixAttention，用 Radix Tree 共享 prefix，更进一步

---

## 6. demo 结果解读

### 6.1 算子融合

```
总算子数        : 168 (12 个 Transformer block × 14 算子)
融合前 kernel 数: 168
融合后 kernel 数: 132
融合算子数      : 36 (Scale + GELU + 部分 element-wise 链)
kernel 减少比例 : 21.4%
```

**解读**：每个 block 融合 3 个算子（Scale、GELU 等），kernel 数减少 21%。
实际生产中 XLA/TensorRT 融合更激进，可减少 30-50%。

### 6.2 请求调度

```
SLA 违约对比：FIFO=285/300, 优先级=219/300
```

**解读**：过载场景下，优先级调度的违约率更低（73% vs 95%），
且违约集中在低优先级请求。生产中 vLLM 的 preemption 进一步降低高优先级违约。

### 6.3 负载均衡

```
一致性哈希迁移 : 254/1000 (25.4%)
取模哈希迁移   : 763/1000 (76.3%)
取模迁移是一致性的 3.0x
```

**解读**：3→4 节点扩容时，一致性哈希只迁移 25% 的 key（约 1/N），
取模哈希迁移 76%。KV cache 保留率直接影响扩缩容时的性能。

### 6.4 显存管理

```
分页利用率 : 88.55%
连续利用率 : 12.60%
分页/连续利用率比: 7.0x
```

**解读**：生成长度远小于 max_seq_len 时，分页分配利用率是连续分配的 7 倍。
这意味着同样显存能服务 7 倍的并发请求。

---

## 7. 生产级推理服务系统

### 7.1 vLLM

vLLM 是当前最流行的开源 LLM 推理服务框架，本章四种数据结构都有体现：

| 数据结构 | vLLM 实现 |
|----------|-----------|
| 计算图 | Triton 写 fused attention kernel（算子融合） |
| 堆 | Engine + Scheduler，按优先级调度请求 |
| 一致性哈希 | 多 GPU worker 用 Ray 分发 |
| 页表 | PagedAttention，block_size=16 |

核心创新：PagedAttention + 连续批处理，吞吐比传统方案高 2-4 倍。

### 7.2 TensorRT-LLM

NVIDIA 的推理优化框架：
- 算子融合：TensorRT 的图优化，融合 LayerNorm+MatMul+GELU 等
- in-flight batching：动态批处理，请求随时加入/移除
- 显存：类似 PagedAttention 的分页管理
- 量化：FP8/INT8 量化，进一步降显存

### 7.3 Triton Inference Server

NVIDIA 的通用推理服务框架：
- 多模型多队列：按模型和优先级调度
- 动态 batching：自动组 batch，平衡吞吐和延迟
- 多 backend：支持 TensorRT、PyTorch、ONNX、vLLM 等

### 7.4 SGLang

新兴推理框架，核心创新 RadixAttention：
- 用 Radix Tree 共享 prefix 的 KV cache
- 同 system prompt 的请求共享 prefix block，更进一步省显存
- 结构化生成加速（regex、JSON 约束）

---

## 8. 小结

大模型推理服务的四个层面问题，分别由四种数据结构解决：

1. **算子融合（计算图）**：减少 kernel launch，降低延迟。14→11 kernel/block，减少 21%。
2. **请求调度（堆）**：按 SLA 优先级服务，VIP 优先。违约率 95%→73%。
3. **负载均衡（一致性哈希）**：扩缩容时低迁移。迁移 76%→25%。
4. **显存管理（页表）**：按需分配 KV cache，高利用率。利用率 13%→89%。

生产级的 vLLM、TensorRT-LLM 在这四个方向上做了工程化实现，
使得大模型推理服务的吞吐和延迟达到可用水平。