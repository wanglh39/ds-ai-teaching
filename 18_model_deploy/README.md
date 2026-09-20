# 第 18 章 · 模型部署与推理服务（扩展专题）

> AI 场景：大模型（LLM）部署推理服务的全链路优化
> 数据结构：计算图（算子融合）+ 堆（请求调度）+ 一致性哈希（负载均衡）+ 页表（显存管理）

## 本章要回答的问题

- 几百个算子的计算图如何减少 kernel launch？（算子融合，14→11 kernel/block）
- 混合优先级的推理请求如何调度？（堆/优先队列，SLA 违约率 95%→73%）
- 多 GPU 分发请求，扩缩容时如何减少迁移？（一致性哈希，迁移 76%→25%）
- KV cache 显存如何按需分配不浪费？（页表/PagedAttention，利用率 13%→89%）
- 4 种数据结构如何协奏完成推理服务？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示计算图、堆、一致性哈希、页表如何协作解决大模型推理服务的全链路问题。

## 目录

```
18_model_deploy/
├── python/
│   └── demo.py              # 推理服务 demo（4 种数据结构协作）
├── docs/
│   ├── principle.md         # 4 种数据结构原理 + 协作机制
│   └── ai_application.md    # 模型部署应用（vLLM / TensorRT-LLM / Triton）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟完整大模型推理服务，4 种数据结构协作：

```
推理请求到达
  ├─→ [堆/优先队列]   请求调度：按 SLA/优先级 pop，高优先级先服务
  ├─→ [一致性哈希环]  负载均衡：分发到多 GPU，节点增减时迁移量小
  ├─→ [计算图]        算子融合：合并相邻 element-wise 算子，减少 kernel launch
  └─→ [页表]          显存管理：PagedAttention 分块分配 KV cache
```

- `OperatorFusion`：计算图 DAG + 拓扑序贪心融合 element-wise 算子
- `RequestScheduler`：heapq 优先队列，(-priority, arrive_ts, req_id) tie-breaker
- `ConsistentHashRing`：有序数组 + bisect 二分，虚拟节点解决倾斜
- `MemoryPaged`：物理块池 + 页表 + 空闲链表，按需分配 block

性能对比：
1. 算子融合前后 kernel 数（168→132，减少 21.4%）
2. 优先级调度 vs FIFO（SLA 违约 219 vs 285）
3. 一致性哈希 vs 取模哈希（迁移 254 vs 763，3x 更优）
4. 分页显存 vs 连续分配（利用率 88.55% vs 12.60%，7x 更优）

## 跑法

```bash
python 18_model_deploy/python/demo.py
```

## 状态

已完成：demo + 文档 + 性能对比图，验证通过。