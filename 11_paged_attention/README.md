# 第 11 章 · PagedAttention 专题

> 数据结构：页表 + 固定大小块 + 空闲块池
> AI 优化：PagedAttention（vLLM 的 KV Cache 管理）

## 本章要回答的问题

- KV Cache 为什么会碎片化？
- 操作系统的分页机制怎么搬到 KV Cache？
- 朴素分配 vs PagedAttention 的显存利用率差多少？

## 目录

```
11_paged_attention/
├── c/                   # 页表 + 分块内存的 C 实现
├── python/              # PagedAttention 模拟 demo
├── docs/
│   ├── principle.md
│   └── ai_application.md
└── figures/
```

## 状态

🚧 待填充