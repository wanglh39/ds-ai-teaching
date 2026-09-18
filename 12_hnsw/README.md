# 第 12 章 · HNSW 专题

> 数据结构：跳表（多层索引链表）+ 近邻图
> AI 优化：HNSW 向量检索

## 本章要回答的问题

- 为什么纯图不够、纯跳表也不够？
- 跳表给图提供了什么？图给跳表提供了什么？
- HNSW vs 暴力 KNN 在 1M 向量下的延迟差多少？

## 目录

```
12_hnsw/
├── c/                   # 跳表 + 图的 C 实现
├── python/              # HNSW 向量检索 demo
├── docs/
│   ├── principle.md
│   └── ai_application.md
└── figures/
```

## 状态

🚧 待填充