# 第 16 章 · RAG 与知识检索（扩展专题）

> AI 场景：检索增强生成（RAG）：语义检索 + 关键词检索 + 知识图谱 + 去重
> 数据结构：HNSW + 倒排索引 + 图 + 布隆过滤器（4 种协作）

## 本章要回答的问题

- 语义检索用什么数据结构？（HNSW 向量近邻搜索）
- 关键词检索用什么数据结构？（倒排索引，词 → 文档列表）
- 多跳推理用什么数据结构？（知识图谱，实体关系，BFS 多跳）
- 多路结果去重用什么数据结构？（布隆过滤器，O(1) 判重）
- 4 种数据结构如何协奏？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示 HNSW、倒排索引、知识图谱、布隆过滤器如何协作解决 RAG 的全链路检索问题。

## 目录

```
16_rag_retrieval/
├── python/
│   └── demo.py              # RAG pipeline demo（4 种数据结构协作）
├── docs/
│   ├── principle.md         # 4 种数据结构原理 + 协作机制
│   └── ai_application.md    # RAG 应用（LangChain / LlamaIndex / GraphRAG）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟完整 RAG pipeline，4 种数据结构协作：

```
用户 query
  ├─→ [HNSW]      语义检索（query 向量化 → top-k 相似文档）
  ├─→ [倒排索引]  关键词检索（query 分词 → AND/OR 查询）
  ├─→ [知识图谱]  多跳推理（抽实体 → 邻居 → 相关文档）
  └─→ [布隆]      融合去重（多路结果合并，O(1) 判重）
```

- `SimpleHNSW`：简化 HNSW 向量检索（暴力 KNN，展示协作接口）
- `InvertedIndex`：倒排索引（词 → 文档 ID 列表，AND/OR/phrase 查询）
- `KnowledgeGraph`：知识图谱（实体 + 关系，邻接表，BFS 多跳）
- `BloomFilter`：布隆过滤器（k 个哈希 + m 位位数组，可控误判率）

性能对比：
1. 多路检索 vs 单路（召回率 100% vs 62%，提升 38 个百分点）
2. 布隆去重 vs 集合去重（内存省 23x，代价是 0.4% 误判）
3. 倒排索引 vs 全文扫描（快 5.4x）

## 跑法

```bash
python 16_rag_retrieval/python/demo.py
```

## 状态

✅ 已完成