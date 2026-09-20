# 第 17 章 · 数据管道与流式处理（扩展专题）

> AI 场景：AI 训练数据的流式加载、滑动窗口统计、近似去重
> 数据结构：队列 + 环形缓冲 + 树状数组（BIT）+ 布隆过滤器（4 种协作）

## 本章要回答的问题

- 多阶段数据管道用什么数据结构连接？（有界队列 + 背压）
- 流式数据如何固定内存保留最近 N 个？（环形缓冲，O(1) 覆盖最老）
- 滑动窗口内区间统计如何加速？（树状数组 BIT，O(log n) vs 朴素 O(n)）
- 流式数据去重如何省内存？（布隆过滤器，位数组 vs 哈希表）
- 4 种数据结构如何协奏完成流式训练管道？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示队列、环形缓冲、树状数组、布隆过滤器如何协作解决 AI 训练数据流式处理的全链路问题。

## 目录

```
17_data_pipeline/
├── python/
│   └── demo.py              # 数据管道 demo（4 种数据结构协作）
├── docs/
│   ├── principle.md         # 4 种数据结构原理 + 协作机制
│   └── ai_application.md    # 流式处理应用（PyTorch DataLoader / WebDataset / Kafka）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟完整 AI 数据管道，4 种数据结构协作：

```
原始样本流
  ├─→ [PipelineQueue] 多阶段管道：producer → processor → augmenter → batcher
  ├─→ [RingBuffer]    滑动窗口：保留最近 N 个样本的 loss，O(1) 写入/覆盖
  ├─→ [BIT]           窗口统计：O(log n) 区间和 / 前缀和
  └─→ [BloomFilter]   流式去重：O(1) 判重，内存远小于 set
```

- `PipelineQueue`：有界队列连接各阶段，满了丢最老的（背压）
- `RingBuffer`：定长数组 + 写指针取模，O(1) 覆盖最老
- `BIT`：树状数组，lowbit 分块前缀和，O(log n) 更新/查询
- `BloomFilter`：双哈希线性组合 + 位数组，可控误判率

性能对比：
1. BIT 区间和 vs 朴素累加（O(log n) vs O(n)，加速 ~4.4x）
2. 布隆去重 vs 集合去重（内存省 58x，Python 中速度 set 占优）
3. 管道背压 vs 无界缓存（内存峰值 256 vs 7500，固定 vs 增长）
4. RingBuffer vs list.pop(0)（O(1) vs O(n)，加速 ~20x）

## 跑法

```bash
python 17_data_pipeline/python/demo.py
```

## 状态

已完成：demo + 文档 + 性能对比图，验证通过。