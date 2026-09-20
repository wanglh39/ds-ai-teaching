# 多智能体编排的 AI 应用：LangGraph / AutoGen / CrewAI

> 本文档讲清「多 agent 编排在 AI 里干什么」。原理见 `principle.md`。核心：多个 AI agent 协作完成复杂任务（如写一篇技术博客），需要 4 种数据结构协作——DAG 决定执行顺序、队列传消息、堆做优先级调度、并查集管分组。

## 1. 多智能体系统的背景

### 1.1 从单 agent 到多 agent

单个 LLM agent 能力有限：一个 agent 既要搜索资料、又要写文章、又要审核，上下文塞不下、角色混乱、错误难定位。**多智能体系统**（Multi-Agent System）把复杂任务拆成多个 agent，每个 agent 专注一个职能：

```
单 agent（什么都做）：          多 agent（各司其职）：

  [一个巨大的 prompt]            [search agent] → 搜索资料
  → 上下文爆炸                    [research agent] → 整理研究
  → 角色混乱                      [outline agent] → 写大纲
  → 错误难定位                    [draft agent] → 写初稿
                                  [review agent] → 审核
                                  → 上下文聚焦、角色清晰、可调试
```

### 1.2 主流框架

| 框架 | 出品 | 编排核心 | 典型用法 |
|---|---|---|---|
| **LangGraph** | LangChain | 图编排（StateGraph） | 把 agent 编排成状态机/图 |
| **AutoGen** | Microsoft | 消息路由（GroupChat） | agent 间对话式协作 |
| **CrewAI** | 开源 | 角色分组（Crew） | 角色分工 + 任务委派 |
| **MetaGPT** | 开源 | SOP 标准作业流程 | 模拟软件公司团队 |

这些框架底层都用到了本章讲的 4 种数据结构，只是封装和侧重点不同。

### 1.3 为什么需要数据结构

agent 多了之后，**编排**（orchestration）成为核心问题：

- **谁先执行？** agent 有依赖（写作要等研究），不能乱序
- **怎么传消息？** agent 之间要传中间结果
- **谁优先？** 多个就绪 agent，关键任务先执行
- **怎么分组？** 同职能 agent 共享上下文，跨组通信要序列化

这 4 个问题分别对应 DAG、队列、堆、并查集。

## 2. agent 依赖图为什么是 DAG

### 2.1 依赖关系天然形成有向图

多 agent 工作流里，agent 之间有明确的依赖：

```
写技术博客：
  search → research → outline → draft → {citation, polish} → review → publish

数据分析报告：
  fetch → clean → analyze → {visualize, summarize} → report → send

代码审查：
  parse → {lint, test, security_scan} → aggregate → comment → merge
```

把 agent 看成顶点，依赖看成有向边，这就是**有向图**。

### 2.2 循环依赖 = 死锁

如果依赖图有环 `A → B → C → A`：

- A 等 C 完成
- C 等 B 完成
- B 等 A 完成
- → **死锁**，三个 agent 互相等待，没人能执行

所以 agent 依赖图**必须是无环的**，即 **DAG**（Directed Acyclic Graph）。

### 2.3 现实中的循环依赖怎么处理

有时 agent 之间确实需要"循环"（如审核 agent 发现问题要退回写作 agent 修改）。这不是真正的循环依赖，而是**迭代**：

```
线性 DAG（无审核退回）：       带退回的迭代（展开成 DAG）：

  draft → review                draft → review ──→ 通过 → publish
                                   ↑      │
                                   └── 不通过（重做 draft）
                                   
  审核退回 = 把 draft → review → draft 的"环"展开成多次迭代
  每次迭代是一个 DAG，迭代次数有上限（避免无限循环）
```

LangGraph 的 `conditional edges` 支持这种"条件退回"：审核 agent 根据结果决定走 `publish` 还是回到 `draft`，每次决策是一个新的 DAG 执行。

## 3. 拓扑排序决定执行顺序

### 3.1 为什么不能随机顺序执行

随机顺序执行会导致 **agent 在依赖未满足时执行**：

- 写作 agent 在研究 agent 之前执行 → 没有资料，写空内容
- 审核 agent 在写作 agent 之前执行 → 没有初稿，审什么？

demo 的性能对比显示：**乱序执行导致 2.59x 的尝试次数**（200 次试验，8 个 agent，拓扑排序 1600 次尝试，乱序 4151 次尝试，多出 2551 次无效尝试）。

### 3.2 拓扑排序给出合法顺序

**Kahn 算法**（BFS 拓扑排序）：

1. 找所有入度为 0 的 agent（无前置，可立即执行）
2. 执行它们，把后继的入度 -1
3. 新的入度 0 的 agent 进入下一轮
4. 重复直到所有 agent 执行完

demo 的博客工作流拓扑排序结果：

```
合法执行顺序 = [search, research, outline, draft, citation, polish, review, publish]
```

### 3.3 拓扑层级：同层并行

更细粒度的是**拓扑层级**——同层 agent 无依赖，可并行：

```
层 0: [search]           ← 串行
层 1: [research]
层 2: [outline]
层 3: [draft]
层 4: [citation, polish]  ← 可并行（2 个 agent 同时跑）
层 5: [review]
层 6: [publish]
```

关键路径长度 = 7 步。即使无限并行，最少也要 7 步（因为 `search → research → outline → draft → review → publish` 是必须串行的关键路径）。`citation` 和 `polish` 在层 4 可并行，省 1 步。

### 3.4 LangGraph 的图编排

LangGraph 的 `StateGraph` 本质就是 DAG + 拓扑排序：

```python
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("search", search_agent)
graph.add_node("research", research_agent)
graph.add_node("draft", draft_agent)
graph.add_edge("search", "research")  # DAG 边
graph.add_edge("research", "draft")
graph.compile()  # 编译时做拓扑排序
```

`compile()` 时 LangGraph 检查是否有环（必须是 DAG），并计算拓扑序用于执行。

## 4. agent 间通信的消息队列

### 4.1 消息传递的必要性

agent 不是孤立执行的：研究 agent 的输出要传给写作 agent。这是**生产者-消费者**模式：

```
search agent（生产者）          research agent（消费者）
  │                                │
  └─── 消息队列 ──────────────────→│
       "搜索结果: [...]"           │ 读取消息，整理研究
```

### 4.2 FIFO 队列的公平性

每个 agent 维护一个 FIFO 队列接收消息。FIFO 保证：

- **先发的先处理**：search 先于 research 发消息，research 先处理 search 的消息
- **无饥饿**：每条消息最终都被处理
- **保序**：便于调试和追溯

demo 的消息队列实现：

```python
mq = MessageQueue()
mq.send("search", "research", "搜索结果: [论文A, 论文B]")
mq.send("research", "outline", "研究综述: ...")
# research agent 接收
msg = mq.recv("research")  # → search 的消息（先发的先收）
```

### 4.3 AutoGen 的消息路由

AutoGen 的 `GroupChat` 用消息队列做 agent 间对话：

```python
from autogen import GroupChat

chat = GroupChat(
    agents=[search_agent, research_agent, draft_agent],
    messages=[],
)
# search_agent 发消息 → 进队列 → 路由给 research_agent
# research_agent 回复 → 进队列 → 路由给下一个 agent
```

AutoGen 的路由比简单 FIFO 复杂：有 `round_robin`（轮询）、`manual`（手动指定下一个）、`random`（随机）、`auto`（LLM 决定）等策略。但底层都是消息队列 + 路由函数。

### 4.4 消息的两种模式

- **同步消息**：发送方等接收方处理完才继续（如 RPC）。适合强依赖。
- **异步消息**：发送方不等，继续执行。适合解耦。

多 agent 编排通常用**异步消息**：search agent 发完消息就结束，不等 research agent 处理。research agent 从队列取消息时 search 已经完成了。这解耦了 agent，提高并行度。

## 5. 任务优先级调度

### 5.1 为什么需要优先级

不是所有 agent 同等重要：

| 优先级 | agent | 理由 |
|---|---|---|
| 高（1） | review, publish | 用户在等最终结果，关键路径 |
| 中（2） | search, research, outline, draft | 主干任务，必须完成但用户不直接看 |
| 低（3） | citation, polish | 辅助任务，可延迟 |

多个就绪 agent 时，**先执行哪个**？FIFO 不区分优先级，关键任务可能排在低优后面。

### 5.2 堆做优先级调度

最小堆（`heapq`）保证每次 $O(\log n)$ 取出优先级最高的任务：

```python
scheduler = PriorityScheduler()
scheduler.push("search", priority=2)
scheduler.push("review", priority=1)   # 高优
scheduler.push("polish", priority=3)   # 低优

next_task = scheduler.pop()  # → review（priority=1 最小，最高优）
```

### 5.3 demo 的对比结果

构造场景：50 个独立任务同时就绪，5 个高优后到达（push 顺序：低优先、高优后）：

| 调度器 | 关键任务执行位置 | 延迟总和 |
|---|---|---|
| 优先级调度（堆） | [0, 1, 2, 3, 4] | 10 步 |
| FIFO 调度（队列） | [45, 46, 47, 48, 49] | 235 步 |

**优先级调度让后到达的高优任务插队到最前面**，延迟仅 10 步；FIFO 让高优任务排在 45 个低优后面，延迟 235 步——**23.5x 差距**。

### 5.4 优先级反转问题

纯优先级调度有**优先级反转**问题：低优任务持有高优任务需要的资源，导致高优任务被低优任务间接阻塞。解决方案：

- **优先级继承**：低优任务临时继承高优任务的优先级
- **优先级天花板**：持有资源的任务优先级提升到可能等待它的最高优先级

多 agent 编排里，资源主要是共享上下文。同组 agent（并查集同组）共享上下文，跨组通信时要注意优先级反转。

### 5.5 Celery 的优先级队列

生产级任务队列 Celery 用堆做优先级调度：

```python
from celery import Celery
app = Celery()

@app.task(priority=1)  # 高优
def review_task(doc): ...

@app.task(priority=3)  # 低优
def polish_task(doc): ...
```

Celery 的 `priority` 参数底层映射到 RabbitMQ 的优先级队列，用堆实现。

## 6. agent 分组与并查集

### 6.1 为什么分组

agent 按职能分组，同组 agent 可以**共享上下文/缓存**：

```
研究组：[search, research, citation]
  → 共享资料库（search 找到的论文 research 直接用，不用重新搜）

写作组：[outline, draft, polish]
  → 共享写作风格（outline 定的风格 draft 沿用）

审核组：[review, publish]
  → 共享审核标准
```

跨组通信需要**序列化**（如研究组把结果传给写作组，要转成消息），同组通信可以**共享内存**（快得多）。

### 6.2 并查集的动态合并

分组不是静态的，运行时可能**动态合并**：

- 初始：研究组、写作组、审核组（3 组）
- 发现研究和写作需要紧密协作 → 合并成"内容组"（2 组）
- 再合并审核 → 全在一个组（1 组）

并查集的 `union` 操作 $O(\alpha(n))$ ≈ $O(1)$，让动态合并几乎免费。如果用 `dict[str, set[str]]` 存分组，合并两个组要遍历其中一个 set $O(n)$。

### 6.3 demo 的分组结果

```
组 search:  [citation, research, search]   ← 研究组
组 outline: [draft, outline, polish]       ← 写作组
组 review:  [publish, review]              ← 审核组

跨组检查：
  search 与 research 同组？True   ← 同组，共享资料库
  search 与 draft 同组？False     ← 跨组，要序列化
  review 与 publish 同组？True    ← 同组，共享审核标准
```

### 6.4 CrewAI 的角色分组

CrewAI 的 `Crew` 本质是并查集分组 + 任务委派：

```python
from crewai import Crew, Agent

researcher = Agent(role="研究员", ...)
writer = Agent(role="作家", ...)
reviewer = Agent(role="审核员", ...)

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, write_task, review_task],
)
# 同 role 的 agent 自动同组，共享角色上下文
```

CrewAI 按 `role` 字段分组，同 role 的 agent 共享角色设定（system prompt、工具集、记忆）。

## 7. demo 结果解读

### 7.1 工作流结构

demo 模拟"写技术博客"的 8 agent 工作流：

```
search → research → outline → draft ┬→ citation ┐
                                  └→ polish    ┴→ review → publish
```

- 8 个 agent，8 条依赖边，3 个分组，3 个优先级
- 拓扑层级 7 层（关键路径 7 步），层 4 的 `citation` 和 `polish` 可并行

### 7.2 性能对比结果

**对比 1：拓扑排序 vs 乱序**（200 次试验）：

| 方案 | 总尝试次数 | 重试次数 |
|---|---|---|
| 有依赖排序 | 1600 | 0 |
| 无依赖乱序 | 4151 | 2551 |

乱序导致 2.59x 尝试次数，拓扑排序消除 2551 次无效尝试。

**对比 2：优先级 vs FIFO**（50 个独立任务，5 个高优后到达）：

| 调度器 | 关键任务执行位置 | 延迟总和 |
|---|---|---|
| 优先级调度 | [0,1,2,3,4] | 10 步 |
| FIFO 调度 | [45,46,47,48,49] | 235 步 |

FIFO 让关键任务多等 23.5x。

**对比 3：4 种数据结构随规模增长**（n=100 到 10000）：

| 规模 | DAG 拓扑排序 | 队列 | 堆 | 并查集 |
|---|---|---|---|---|
| 100 | 0.056ms | 0.007ms | 0.025ms | 0.151ms |
| 1000 | 1.10ms | 0.116ms | 0.381ms | 2.56ms |
| 10000 | 10.55ms | 1.30ms | 4.18ms | 24.27ms |

- DAG 拓扑排序 $O(V+E)$：线性增长
- 队列 $O(1)$：增长最慢
- 堆 $O(\log n)$：比队列稍慢
- 并查集 $O(\alpha(n))$：几乎 $O(1)$，但 Python 实现常数较大

### 7.3 协作的体现

demo 的 `[6] 完整工作流模拟` 展示了 4 种数据结构协作：

```
[search] 组=search 优先级=2 收到 0 条消息 → 发给 ['research']
[research] 组=search 优先级=2 收到 1 条消息 → 发给 ['outline']
[outline] 组=outline 优先级=2 收到 1 条消息 → 发给 ['draft']
[draft] 组=outline 优先级=2 收到 1 条消息 → 发给 ['citation', 'polish']
[citation] 组=search 优先级=3 收到 1 条消息 → 发给 ['review']
[polish] 组=outline 优先级=3 收到 1 条消息 → 发给 ['review']
[review] 组=review 优先级=1 收到 2 条消息 → 发给 ['publish']
[publish] 组=review 优先级=1 收到 1 条消息 → 发给 （无后继，完成）
```

每行都涉及 4 种数据结构：
- **执行顺序**由 DAG 拓扑排序决定
- **"收到 N 条消息"**来自消息队列
- **优先级**来自堆调度
- **"组=X"**来自并查集

## 8. 生产级框架的实现

### 8.1 LangGraph：图编排

LangGraph 的核心是 `StateGraph`，本质是 DAG + 条件边：

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)
graph.add_node("search", search_agent)
graph.add_node("draft", draft_agent)
graph.add_node("review", review_agent)

graph.add_edge("search", "draft")
graph.add_conditional_edges(
    "review",
    lambda state: "pass" if state["approved"] else "redo",
    {"pass": END, "redo": "draft"},  # 条件边：审核退回
)
app = graph.compile()  # 编译时拓扑排序
```

- `add_edge`：DAG 边
- `add_conditional_edges`：条件边（审核退回 = 迭代展开成 DAG）
- `compile()`：检查无环 + 拓扑排序

### 8.2 AutoGen：消息路由

AutoGen 的 `GroupChat` 用消息队列 + 路由函数：

```python
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[search_agent, research_agent, draft_agent],
    messages=[],
    speaker_selection_method="auto",  # LLM 决定下一个发言者
)
manager = GroupChatManager(groupchat=groupchat)
# agent 间通过消息队列对话，manager 路由消息
```

- `messages`：消息队列
- `speaker_selection_method`：路由策略（`auto`/`round_robin`/`manual`/`random`）
- 底层：每个 agent 一个消息队列，manager 路由消息

### 8.3 CrewAI：角色分组

CrewAI 的 `Crew` 用并查集按角色分组 + 任务队列：

```python
from crewai import Crew, Agent, Task

researcher = Agent(role="研究员", goal="找到资料", backstory="...")
writer = Agent(role="作家", goal="写文章", backstory="...")

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        Task(description="研究 X", agent=researcher),
        Task(description="写关于 X 的文章", agent=writer),
    ],
    process="sequential",  # 按任务顺序执行（拓扑排序）
)
result = crew.kickoff()
```

- `role`：分组依据（同 role 同组）
- `tasks`：任务列表（按顺序执行 = 拓扑排序）
- `process`：`sequential`（串行）/ `hierarchical"`（有 manager agent 调度）

### 8.4 数据结构映射

| 框架 | DAG | 队列 | 堆 | 并查集 |
|---|---|---|---|---|
| LangGraph | ✅ StateGraph | ✅ 状态传递 | ❌ | ❌ |
| AutoGen | ❌ | ✅ GroupChat | ❌ | ❌ |
| CrewAI | ✅ 任务顺序 | ✅ 任务结果 | ❌ | ✅ 角色分组 |
| 本章 demo | ✅ AgentDAG | ✅ MessageQueue | ✅ PriorityScheduler | ✅ AgentGroups |

本章 demo 是唯一同时用 4 种数据结构的完整实现，展示它们如何协作。

## 9. 扩展：更复杂的编排场景

### 9.1 动态 agent 加入

运行时可能动态加入 agent（如审核 agent 发现需要补充资料，启动一个新的 search agent）。这要求：

- **DAG** 支持动态加边（邻接表 $O(1)$ 加边）
- **堆** 支持动态 push（$O(\log n)$）
- **并查集** 支持动态 union（$O(\alpha(n))$）
- 消息队列天然支持动态

demo 的数据结构都支持动态操作，这是邻接表（而非邻接矩阵）、并查集（而非分组 dict）的优势。

### 9.2 agent 失败与重试

agent 可能失败（LLM 超时、工具报错）。处理方式：

- **重试**：同一个 agent 重新执行（不改变 DAG 结构）
- **降级**：换一个更简单 agent 替代（动态改 DAG）
- **跳过**：标记 agent 失败，后继用默认值继续（DAG 里标记节点状态）

这需要 DAG 维护**节点状态**（pending/running/done/failed），拓扑排序时跳过 failed 节点。

### 9.3 agent 并行执行

同层 agent 可并行。生产级用 `asyncio` 或线程池：

```python
import asyncio

async def execute_level(level_agents):
    tasks = [asyncio.create_task(run_agent(a)) for a in level_agents]
    results = await asyncio.gather(*tasks)
    return results
```

拓扑层级（`topological_levels`）天然给出并行分组，每层内并行，层间串行。

## 10. 小结

多智能体编排是 4 种数据结构的协奏：

- **DAG + 拓扑排序** → 合法执行顺序（消除死锁，2.59x 少尝试）
- **消息队列** → agent 间通信（FIFO 公平，无饥饿）
- **堆** → 优先级调度（关键任务先执行，23.5x 少延迟）
- **并查集** → 动态分组（近 $O(1)$ 合并，共享上下文）

生产级框架 LangGraph（图编排）、AutoGen（消息路由）、CrewAI（角色分组）各自侧重不同数据结构，但本质都是这 4 种的组合。本章 demo 完整展示了 4 种数据结构如何协作解决一个 AI 应用问题——这正是扩展专题的精神：**一个 AI 场景 → 多种数据结构协作**。