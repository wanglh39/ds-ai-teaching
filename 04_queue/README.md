# 第 04 章 · 循环队列

> 数据结构：循环队列（固定大小 + 头尾指针）
> AI 优化：Ring-AllReduce 分布式通信 + 批处理调度

## 本章要回答的问题

- Ring-AllReduce 为什么是带宽最优的通信拓扑？
- 环形队列怎么对应到环形通信拓扑？
- 树形通信 vs 环形通信的通信量差多少？

## 目录

```
04_queue/
├── c/                   # 循环队列的 C 实现
├── python/              # Ring-AllReduce 模拟 demo
├── docs/
│   ├── principle.md
│   └── ai_application.md
└── figures/
```

## 状态

🚧 待填充