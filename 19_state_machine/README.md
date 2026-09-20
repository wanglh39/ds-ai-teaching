# 第 19 章 · 状态机与工作流编排（扩展专题）

> AI 场景：AI 智能体的状态机驱动工作流（LangGraph 式 StateGraph、ReAct、Plan-and-Execute、Tree-of-Thought）
> 数据结构：图（状态转移）+ 栈（ReAct）+ 队列（Plan-Execute）+ 树（ToT）+ 哈希表（state dict）

## 本章要回答的问题

- agent 工作流为什么用状态机而不是 if-else？（可维护、可可视化、可并行、可中断）
- ReAct 推理链为什么用栈？（回溯能力，出错 pop 而非重跑）
- Plan-Execute 任务列表为什么用队列？（FIFO 保依赖顺序，动态 replan 入队）
- Tree-of-Thought 为什么用树？（多分支搜索 + 剪枝，指数级缩小搜索空间）
- 4 种数据结构如何协奏完成 agent 工作流？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示图、栈、队列、树、哈希表如何协作解决 AI 智能体的工作流编排问题。

## 目录

```
19_state_machine/
├── python/
│   └── demo.py              # 4 种 agent 工作流模式 demo
├── docs/
│   ├── principle.md         # 4 种数据结构原理 + 协作机制
│   └── ai_application.md    # 状态机与工作流应用（LangGraph / AutoGen / CrewAI）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟 4 种 agent 工作流模式，4 种数据结构协作：

```
用户提问到达
  ├─→ [图/状态转移图]  StateMachine：状态机驱动状态转移
  ├─→ [栈]             ReActLoop：think→act→observe 推理链 + 回溯
  ├─→ [队列]           PlanAndExecute：planner 生成任务队列，executor FIFO 执行
  ├─→ [树]             TreeOfThought：多分支思维搜索 + 剪枝
  └─→ [哈希表]         状态存储：state dict，所有节点共享读写
```

- `StateMachine`：邻接表存状态转移图 + 条件转移守卫函数 + state dict
- `ReActLoop`：栈记录 think→act→observe 链，`backtrack()` 回溯
- `PlanAndExecute`：`deque` 任务队列，`plan` / `execute_one` / `replan`
- `TreeOfThought`：`ThoughtNode` 树，BFS/DFS 搜索 + 剪枝标记

性能对比：
1. 状态机 vs 硬编码 if-else（新增状态成本 1 行 vs 3 行）
2. ReAct 栈 vs 无栈（出错恢复 2 步 vs 12 步）
3. Plan-Execute 队列 vs 栈（执行顺序正确 1 vs 0）
4. ToT 树搜索 vs 线性枚举（剪枝 58%，访问 25 vs 80 节点）

## 跑法

```bash
python 19_state_machine/python/demo.py
```

## 状态