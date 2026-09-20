# 第 19 章 · 状态机与工作流编排 — AI 应用

> AI 场景：AI 智能体（agent）的状态机驱动工作流编排。
> 数据结构：图（状态转移）+ 栈（ReAct）+ 队列（Plan-Execute）+ 树（ToT）+ 哈希表（state dict）协作。

---

## 1. AI 智能体工作流的挑战

### 1.1 问题：agent 不是单次 LLM 调用

早期的 LLM 应用是「单次调用」：输入 prompt → 输出回答。但真实任务需要**多步推理 + 工具调用 + 状态管理**：

```
简单任务：用户问 "1+1=?" → LLM 直接答 "2"
复杂任务：用户问 "分析 A 公司财报并对比 B 公司"
  → 需要多次调用工具（搜索 / 检索 / 计算）
  → 需要中间状态管理（收集了什么 / 还缺什么）
  → 需要错误恢复（工具调用失败 / 结果不符预期）
  → 需要动态规划（发现新信息要调整计划）
```

这就是 **AI 智能体（agent）**：能自主推理、调用工具、管理状态、动态调整的 LLM 系统。

### 1.2 agent 工作流的四大挑战

| 挑战 | 说明 | 本章数据结构 | 对应模式 |
|------|------|-------------|----------|
| 状态管理 | 多步推理的中间状态 | 图 + 哈希表 | StateGraph |
| 推理回溯 | 出错后恢复 | 栈 | ReAct |
| 任务调度 | 复杂任务拆解 | 队列 | Plan-Execute |
| 思维搜索 | 多方向探索 | 树 | Tree-of-Thought |

### 1.3 为什么需要状态机驱动

硬编码的 agent 工作流（一串 if-else）有几个痛点：

```python
# 硬编码：难以维护
def agent(question):
    if has_input(question):
        plan = generate_plan(question)
        if plan is not None:
            for task in plan:
                result = execute(task)
                if result.error:
                    # 错误处理嵌套深...
                    if should_retry(result):
                        plan = replan(question, result)
                        # 又一层嵌套...
```

痛点：
- **状态隐式**：当前在哪一步、为什么在这里，全靠函数调用栈隐式表达，难以可视化
- **扩展困难**：新增一个状态（如 "reflecting"）要改 if-else 嵌套，容易引入 bug
- **无法中断**：跑到一半想暂停、恢复、回滚，几乎不可能
- **无法并行**：独立的状态分支想并行执行，if-else 结构不支持

状态机驱动解决这些痛点：状态显式、扩展加一行、可中断、可并行。

---

## 2. LangGraph 的 StateGraph 原理

### 2.1 LangGraph 是什么

[LangGraph](https://github.com/langchain-ai/langgraph) 是 LangChain 团队推出的 agent 工作流引擎，核心抽象是 **StateGraph**：把 agent 工作流建模为状态机。

```python
from langgraph.graph import StateGraph

graph = StateGraph(StateDict)
graph.add_node("planning", planning_node)
graph.add_node("acting", acting_node)
graph.add_node("observing", observing_node)
graph.add_edge("planning", "acting")
graph.add_conditional_edges("observing", router_fn, {
    "continue": "acting",
    "done": END,
    "error": "error_handler",
})
graph.set_entry_point("planning")
app = graph.compile()
```

### 2.2 StateGraph 的数据结构

LangGraph 内部用两类数据结构：

**状态转移图（图）**：
- `nodes: dict[str, callable]` — 节点表，每个节点是一个处理函数
- `edges: dict[str, list[Edge]]` — 邻接表，每个节点的出边
- `conditional_edges: dict[str, tuple[router_fn, dict[str, str]]]` — 条件转移

**状态存储（哈希表）**：
- `state: TypedDict` — 所有节点共享读写的状态字典
- 每个节点接收 state、返回 state 的更新部分
- LangGraph 自动合并更新（可配置 reducer）

### 2.3 状态转移的执行

```python
def run(self, input_state):
    state = input_state
    current = self.entry_point
    while current != END:
        # 执行当前节点
        update = self.nodes[current](state)
        state = merge(state, update)
        # 决定下一个节点
        if current in self.conditional_edges:
            router_fn, mapping = self.conditional_edges[current]
            decision = router_fn(state)
            current = mapping[decision]
        else:
            current = self.edges[current][0].target
    return state
```

这就是本章 `StateMachine.step()` 的生产级版本。核心是**图驱动状态转移 + 哈希表存状态**。

### 2.4 StateGraph 的优势

| 特性 | 硬编码 if-else | StateGraph |
|------|----------------|------------|
| 可视化 | 难（要手画） | ✓ `graph.draw()` 自动画 |
| 新增状态 | 改 if-else 嵌套 | `add_node` + `add_edge` 两行 |
| 中断 / 恢复 | 不支持 | ✓ state 可序列化，任意点恢复 |
| 并行 | 难 | ✓ `add_branch` 自动并行 |
| 错误处理 | try-except 嵌套 | 条件转移到 "error_handler" 节点 |
| 持久化 | 难 | ✓ state dict 存数据库 |

---

## 3. ReAct 模式（Reasoning + Acting）

### 3.1 ReAct 论文与思想

ReAct 是 Yao et al. (2022) 提出的 agent 推理模式，核心思想：**让 LLM 交替推理（Reasoning）和行动（Acting）**，而不是一次性生成答案。

```
Question: 科罗拉多山脉的海拔比阿巴拉契亚山脉高多少？

Thought 1: 我需要先查两个山脉的海拔
Action 1: Search[科罗拉多山脉海拔]
Observation 1: 最高峰 Mt Elbert, 4401 米
Thought 2: 现在查阿巴拉契亚
Action 2: Search[阿巴拉契亚山脉海拔]
Observation 2: 最高峰 Mt Mitchell, 2037 米
Thought 3: 4401 - 2037 = 2364 米
Action 3: Finish[2364 米]
```

### 3.2 ReAct 的栈结构

ReAct 的推理链是**栈**结构，本章 `ReActLoop` 实现：

```python
class ReActLoop:
    def __init__(self):
        self.stack: list[ReActStep] = []  # 推理栈

    def think(self, thought):  # push
    def act(self, action):     # push
    def observe(self, result): # push
    def backtrack(self):       # pop，回溯
```

栈的关键作用是**回溯**：当 `observe` 发现错误时，`pop` 掉错误的 `act + observe`，从上一个 `think` 重新推理。无栈的线性流程只能从头重跑。

### 3.3 ReAct 的生产实现

生产级 ReAct（如 LangChain 的 `AgentExecutor`）在栈基础上加了：
- **最大迭代数限制**：防止死循环
- **工具调用超时**：单个 act 不超过 N 秒
- **中间结果持久化**：栈内容可序列化到数据库，支持中断恢复
- **多轮对话上下文**：栈 + 对话历史合并管理

### 3.4 ReAct 的局限

ReAct 是**单链推理**——每次只走一条思维路径。遇到分叉决策（"先查 A 还是先查 B？"）只能选一个，错了再回溯。这引出了 Tree-of-Thought。

---

## 4. Plan-and-Execute 模式

### 4.1 Plan-Execute 的思想

Plan-and-Execute 是 LangChain 官方推荐的 agent 架构（2024），核心：**先规划再执行**，区别于 ReAct 的「边想边做」。

```
ReAct:        think→act→observe→think→act→...（每步现想）
Plan-Execute: plan([t1,t2,t3,...]) → execute(t1) → execute(t2) → ...
              （先全规划好，再逐个执行）
```

### 4.2 为什么用队列

Plan-Execute 的任务列表是**队列**，本章 `PlanAndExecute` 实现：

```python
class PlanAndExecute:
    def __init__(self):
        self.queue: deque[Task] = deque()  # 任务队列

    def plan(self, task_names):  # planner 生成，入队
        for name in task_names:
            self.queue.append(Task(name))

    def execute_one(self):  # executor，队头出队
        return self.queue.popleft()

    def replan(self, new_tasks):  # 动态重规划，追加到队尾
        for name in new_tasks:
            self.queue.append(Task(name))
```

队列 vs 栈的关键区别：
- **队列 FIFO**：先规划的任务先执行，保依赖顺序（`理解→收集→分析→生成→验证`）
- **栈 LIFO**：反序执行，先验证再理解，依赖全乱
- **动态重规划**：队列 append 到队尾，新任务排后面不打断；栈 push 到栈顶，新任务立即执行打断当前

### 4.3 Plan-Execute vs ReAct

| 维度 | ReAct | Plan-Execute |
|------|-------|--------------|
| 规划时机 | 每步现想 | 一次性规划 |
| 执行结构 | 栈（推理链） | 队列（任务队列） |
| 错误恢复 | 回溯 pop | replan 重新规划 |
| 适合任务 | 探索性强、难预估 | 可拆解、步骤明确 |
| LLM 调用 | 多（每步都调） | 少（planner 调一次） |

Plan-Execute 的优势是 **planner 只调一次 LLM**（生成全计划），executor 可以用更便宜的模型执行单步，成本低。劣势是**计划可能不准**，需要 replan。

### 4.4 生产级 Plan-Execute

LangChain 的 `PlanAndExecute` 架构：
- **Planner**：用强模型（GPT-4）生成结构化任务列表
- **Executor**：用便宜模型（GPT-3.5）逐个执行
- **Replanner**：执行后评估，需要时调强模型重新规划
- **State**：用 state dict 跨步骤传递中间结果

---

## 5. Tree-of-Thought 模式

### 5.1 ToT 论文与思想

Tree-of-Thought（Yao et al. 2023）是 ReAct 的升级：**不只走一条思维链，而是分支多个方向，搜索最优思维路径**。

```
ReAct:  think → act → observe → think → ... （一条链）
ToT:    think ─┬─ 分支1 → 分支1.1 → ...
              ├─ 分支2 → 分支2.1 → ...   （多分支树）
              └─ 分支3 → 分支3.1 → ...
```

### 5.2 ToT 的四个步骤

每个思维节点：

1. **Thought Decomposition**：把当前思维拆成多个候选下一步
2. **State Evaluation**：给每个候选打分（用 LLM 评估潜力）
3. **Search Algorithm**：BFS / DFS / beam search 遍历思维树
4. **Pruning**：低分分支直接剪掉，不展开

### 5.3 ToT 的树结构

本章 `TreeOfThought` 实现：

```python
class ThoughtNode:
    thought: str
    score: float
    children: list["ThoughtNode"]
    pruned: bool  # 剪枝标记

class TreeOfThought:
    def build(self, root_thought):  # 构建思维树
    def bfs_best(self):  # BFS 搜索最优叶
    def dfs_best(self):  # DFS 搜索最优叶
```

树结构让**剪枝是 O(1) 的标记**——把 `child.pruned = True`，搜索时跳过整棵子树。

### 5.4 ToT 的搜索策略

```
BFS（广度优先）：逐层搜索，找浅层最优解
  适合：最优解在浅层
  内存：O(宽度^深度)，深树爆炸

DFS（深度优先）：一条路走到底再回溯
  适合：深树、内存敏感
  内存：O(深度)

Beam Search：每层保留 top-k，其余剪枝
  适合：分支多、需要平衡广度深度
  内存：O(k × 深度)
```

ToT 通常用 **BFS + beam search + 剪枝**，在探索和利用间平衡。

### 5.5 ToT vs ReAct vs CoT

| 模式 | 思维结构 | 搜索 | 适合任务 |
|------|----------|------|----------|
| CoT | 一条链 | 无 | 简单推理 |
| ReAct | 一条链 + 回溯 | 栈回溯 | 工具调用 |
| ToT | 多分支树 | BFS/DFS + 剪枝 | 复杂决策（24 点、创意写作、战略规划） |

ToT 在需要**探索多个方向**的任务上显著优于 ReAct（论文中 24 点游戏从 4% 提升到 74%），但代价是 **LLM 调用次数多**（每个分支都要评估）。

---

## 6. 状态机 vs 硬编码 if-else 的优势

### 6.1 可维护性

```python
# 硬编码：新增一个 "reflecting" 状态要改 3 处
def agent(question):
    plan = generate_plan(question)
    for task in plan:
        result = execute(task)
        if result.error:
            # 改 1：这里加 if should_reflect
            if should_reflect(result):
                reflection = reflect(result)  # 改 2：新逻辑
                if reflection.needs_replan:   # 改 3：嵌套
                    plan = replan(...)

# 状态机：新增 "reflecting" 状态只需 2 行
sm.add_node("reflecting", reflect_node)
sm.add_conditional_edges("observing", router, {"reflect": "reflecting", ...})
```

### 6.2 可可视化

状态机可以自动画出状态转移图，便于调试和沟通：

```python
# LangGraph
graph.draw_mermaid_png()  # 生成 Mermaid 图

# 本章 demo
print(sm.graph)  # 邻接表一目了然
```

硬编码的 if-else 流程要手画，且容易和代码脱节。

### 6.3 可并行

状态机可以识别**独立分支**自动并行：

```python
# LangGraph：branch 自动并行
graph.add_branch("planning", ["research_a", "research_b", "research_c"])
# research_a / b / c 并行执行，全部完成后汇合
```

硬编码的串行 if-else 无法自动并行，要手动开线程 / asyncio。

### 6.4 可中断 / 恢复

状态机的 state dict 可序列化，任意点可暂停恢复：

```python
# LangGraph：checkpoint 到数据库
app = graph.compile(checkpointer=SqliteSaver(...))
# 中断后从任意 checkpoint 恢复
config = {"configurable": {"thread_id": "thread-1"}}
app.invoke(None, config)  # 从上次中断处继续
```

硬编码跑到一半想暂停，函数调用栈全丢，无法恢复。

---

## 7. demo 结果解读

### 7.1 状态机 vs 硬编码

```
状态机状态数: 6, 转移数: 7
新增状态成本: 状态机 1 行 vs 硬编码 3 行
```

状态机耗时略高于硬编码（多了图遍历），但**新增状态成本 1 行 vs 3 行**，可维护性显著更优。这是用一点运行时开销换可维护性的典型权衡。

### 7.2 ReAct 栈 vs 无栈

```
回溯能力: 栈=1, 无栈=0
出错恢复: 栈 2 步 vs 无栈 12 步
```

栈支持回溯，出错只需 pop 2 步重新推理；无栈只能从头重跑 12 步。对于易出错的 agent 任务（工具调用常失败），栈的回溯能力是关键。

### 7.3 Plan-Execute 队列 vs 栈

```
执行顺序正确: 队列=1, 栈=0
动态重规划: 队列=1, 栈=0
```

队列 FIFO 保依赖顺序（`理解→收集→分析→生成→验证`），栈 LIFO 反序执行破坏依赖。动态重规划时队列 append 到队尾不打断，栈 push 到栈顶立即打断。

### 7.4 ToT 树搜索 vs 线性

```
展开节点: 8, 剪枝: 11 (58%)
访问数: BFS=25, DFS=25, 线性=80
```

ToT 剪枝掉 58% 的节点，BFS/DFS 只访问 25 节点 vs 线性 80 节点。树 + 剪枝让搜索空间从线性枚举的 80 降到 25，节省 ~69%。

---

## 8. 生产级实现

### 8.1 LangGraph（LangChain）

LangGraph 是状态机驱动 agent 的代表：
- **StateGraph**：图 + 哈希表（本章 StateMachine 的生产版）
- **条件转移**：`add_conditional_edges` + router_fn
- **持久化**：SqliteSaver / PostgresSaver checkpoint
- **并行**：`add_branch` 自动并行分支
- **人机协作**：`interrupt_before` / `interrupt_after` 暂停等人输入
- **流式**：state 更新流式输出

### 8.2 AutoGen（Microsoft）

AutoGen 用**对话图**编排多 agent：
- 每个 agent 是一个节点
- agent 间的消息传递是边
- 支持嵌套对话（agent 内部再开子对话，对应栈）
- `GroupChat` 用图定义 agent 发言顺序

### 8.3 CrewAI

CrewAI 用**任务队列**编排 agent 团队：
- `Crew` = 一组 agent + 一组 task
- task 间可定义依赖（队列保顺序）
- 支持 `process="hierarchical"`（manager agent 分配任务，对应 Plan-Execute）
- 每个 task 内部可以跑 ReAct

### 8.4 LlamaIndex Workflows

LlamaIndex 的 `Workflow` 用**事件驱动**编排：
- 每步 emit 事件，下一步监听事件触发
- 事件队列（本章 PlanAndExecute 队列的变体）
- 支持并发事件、条件事件

### 8.5 各引擎的数据结构对照

| 引擎 | 工作流结构 | 推理结构 | 任务结构 | 状态存储 |
|------|-----------|----------|----------|----------|
| LangGraph | 图（StateGraph） | 栈（ReAct） | 队列（Plan-Execute） | 哈希表（state dict） |
| AutoGen | 图（对话图） | 栈（嵌套对话） | — | 哈希表（context） |
| CrewAI | — | 栈（ReAct） | 队列（task list） | 哈希表（context） |
| LlamaIndex | 图（事件图） | — | 队列（事件队列） | 哈希表（context） |

它们都用了**图 + 栈 + 队列 + 哈希表**的组合，只是侧重不同。LangGraph 最完整地实现了状态机驱动，是本章 demo 的主要参照。

---

## 9. 小结

AI 智能体的工作流编排是「一个场景 → 多种数据结构协作」的典型例子：

1. **状态机（图 + 哈希表）**：工作流的骨架，状态显式、可可视化、可中断、可并行
2. **ReAct（栈）**：单步推理的回溯结构，出错 pop 而非重跑，支持嵌套子任务
3. **Plan-Execute（队列）**：任务拆解的调度结构，FIFO 保依赖，支持动态 replan
4. **Tree-of-Thought（树）**：多分支思维搜索，BFS/DFS + 剪枝，指数级缩小搜索空间

四种数据结构在不同层次各司其职，通过 state dict 传递信息，嵌套组合完成复杂 agent 工作流。LangGraph、AutoGen、CrewAI 等生产级引擎都基于这套数据结构组合，证明了它的工程有效性。