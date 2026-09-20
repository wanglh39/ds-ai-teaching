# 第 13 章 · 多智能体编排（扩展专题）

> AI 场景：多个 AI agent 协作完成复杂任务（LangGraph / AutoGen / CrewAI）
> 数据结构：DAG + 队列 + 堆 + 并查集（4 种协作）

## 本章要回答的问题

- 多个 AI agent 协作时，谁先执行？（DAG 拓扑排序）
- agent 之间怎么传消息？（队列）
- 多个 agent 就绪时，谁优先？（堆）
- agent 怎么分组共享上下文？（并查集）
- 4 种数据结构如何协奏？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示 DAG、队列、堆、并查集如何协作解决多智能体编排问题。

## 目录

```
13_multi_agent/
├── python/
│   └── demo.py              # 多 agent 编排 demo（4 种数据结构协作）
├── docs/
│   ├── principle.md         # 4 种数据结构原理 + 协作机制
│   └── ai_application.md    # 多智能体编排应用（LangGraph/AutoGen/CrewAI）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟"写技术博客"的 8 agent 工作流：

```
search → research → outline → draft ┬→ citation ┐
                                  └→ polish    ┴→ review → publish
```

4 种数据结构协作：
- `AgentDAG`：邻接表 + 拓扑排序，决定执行顺序
- `MessageQueue`：队列，agent 间消息传递
- `PriorityScheduler`：堆，优先级调度
- `AgentGroups`：并查集，agent 分组管理

性能对比：
1. 有依赖排序 vs 无依赖乱序（2.59x 尝试次数差异）
2. 优先级调度 vs FIFO（23.5x 关键任务延迟差异）
3. 4 种数据结构随规模增长的耗时

## 跑法

```bash
python 13_multi_agent/python/demo.py
```

## 状态

✅ 已完成