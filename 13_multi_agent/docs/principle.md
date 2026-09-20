# 多智能体编排的数据结构原理：DAG + 队列 + 堆 + 并查集

> 本文档讲清「多 agent 编排为什么需要 4 种数据结构协作」。AI 应用（LangGraph/AutoGen/CrewAI 多智能体系统）见 `ai_application.md`。前 12 章是「一个数据结构 → 一个 AI 优化」，本章是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。

## 0. 为什么是 4 种数据结构

多智能体系统（Multi-Agent System）的核心问题是：**多个 AI agent 如何协作完成一个复杂任务**。这一个问题分解出 4 个子问题，每个子问题对应一种数据结构：

| 子问题 | 数据结构 | 作用 |
|---|---|---|
| agent 之间有依赖，谁先执行？ | **DAG + 拓扑排序** | 决定合法执行顺序 |
| agent 之间要传消息，怎么传？ | **队列** | FIFO 消息传递 |
| agent 有优先级，谁先调度？ | **堆** | 优先级调度 |
| agent 要分组，谁和谁一组？ | **并查集** | 动态分组管理 |

这 4 种数据结构不是独立工作，而是**协奏**：DAG 决定哪些 agent 就绪 → 堆从就绪 agent 里选优先级最高的 → 执行后通过队列把结果传给后继 → 并查集维护 agent 分组用于上下文共享。

## 1. DAG 与拓扑排序：agent 依赖必须用 DAG

### 1.1 agent 依赖关系是图

多 agent 工作流里，agent 之间常有依赖：写作 agent 要等研究 agent 完成，审核 agent 要等写作 agent 完成。把 agent 看成顶点，依赖关系看成有向边，整个工作流是一个**有向图**。

```
写技术博客的工作流：

    search → research → outline → draft ┬→ citation ┐
                                      └→ polish    ┴→ review → publish

  search   = 搜索资料
  research = 整理研究
  outline  = 写大纲
  draft    = 写初稿
  citation = 加引用
  polish   = 润色
  review   = 审核
  publish  = 发布
```

### 1.2 为什么必须是 DAG

**DAG**（Directed Acyclic Graph，有向无环图）= 没有环的有向图。agent 依赖图**必须**是 DAG，因为：

- 如果有环 `A → B → C → A`，那么 A 等 C 完成、C 等 B 完成、B 等 A 完成 → **死锁**，没人能执行
- DAG 保证至少有一个无前置的 agent（源点），可以从源点开始执行
- DAG 的拓扑排序给出一个**合法的全局执行顺序**：每个 agent 的所有前置都在它之前

```
有环（死锁）：          DAG（可执行）：

    A → B → C            A → B → C
    ↑       │            ↑       │
    └───────┘            └→ D ───┘

  A 等 C，C 等 A        A 无前置，先执行 A
  → 死锁                → B、D 就绪 → C → 拓扑排序可行
```

### 1.3 拓扑排序：Kahn 算法

**拓扑排序**（Topological Sort）把 DAG 的顶点排成线性序列，使所有有向边 $(u,v)$ 都满足 $u$ 在 $v$ 前面。**Kahn 算法**用 BFS 实现：

```
1. 计算每个顶点的入度（前置数）
2. 把入度为 0 的顶点入队（无前置，可立即执行）
3. while 队列非空：
       u = 出队
       加入排序结果
       for u 的每个后继 v：
           v 的入度 -= 1（u 完成了，v 少一个前置）
           if v 的入度 == 0：
               v 入队（v 的前置都完成了，可执行）
4. 如果结果长度 < 顶点数 → 有环，无法排序
```

**复杂度** $O(|V| + |E|)$：每个顶点入队/出队一次，每条边在删边时检查一次。

### 1.4 拓扑层级：同层 agent 可并行

拓扑排序给出**一个**合法顺序，但不是唯一的。更细粒度的结构是**拓扑层级**：

- 第 0 层 = 无前置的 agent（源点）
- 第 $k$ 层 = 所有前置都在 $0..k-1$ 层的 agent

**同一层的 agent 无依赖关系，可以并行执行**。层级数 = 关键路径长度 = 最少串行步数。

```
博客工作流的拓扑层级：

  层 0: [search]           ← 最先执行
  层 1: [research]
  层 2: [outline]
  层 3: [draft]
  层 4: [citation, polish]  ← 同层，可并行
  层 5: [review]
  层 6: [publish]          ← 最后执行

  关键路径长度 = 7 步（即使无限并行，最少也要 7 步）
```

### 1.5 邻接表 vs 邻接矩阵

DAG 用**邻接表**存储（`dict[str, list[str]]`），原因：

- agent 依赖图是**稀疏**的：每个 agent 通常只依赖几个 agent，远小于总 agent 数
- 拓扑排序只遍历实际存在的边 $O(|V|+|E|)$，邻接矩阵要扫整个 $O(|V|^2)$
- 动态增删 agent/依赖：邻接表 $O(1)$，邻接矩阵要重新分配 $O(|V|^2)$

```python
class AgentDAG:
    def __init__(self):
        self._succ: dict[str, list[str]] = {}  # u → [后继...]
        self._pred: dict[str, list[str]] = {}  # v → [前置...]
```

存两种邻接表：`_succ` 用于拓扑排序时遍历后继，`_pred` 用于查前置和算入度。

## 2. 队列与消息传递：FIFO 的公平性

### 2.1 agent 间为什么需要消息传递

agent 不是孤立执行的：研究 agent 的输出要传给写作 agent，写作 agent 的输出要传给审核 agent。这是**生产者-消费者**模式：

- 生产者 agent 完成后，把结果**发送**给消费者 agent
- 消费者 agent **等待**所有前置的消息到达后才执行

### 2.2 为什么用队列

每个 agent 维护一个**FIFO 队列**接收消息。FIFO（First In First Out）保证：

- **公平性**：先发的消息先被处理，不会饿死早期消息
- **保序性**：消息的处理顺序和发送顺序一致，便于调试
- **无饥饿**：每条消息最终都会被处理，不会有消息永远等在队列里

```
agent A 的消息队列（FIFO）：

  发送顺序：search → A, research → A, outline → A
  队列：    [search 的结果, research 的结果, outline 的结果]
  接收顺序：search 的结果 → research 的结果 → outline 的结果（同发送顺序）
```

### 2.3 实现：deque + 全局时钟

```python
class MessageQueue:
    def __init__(self):
        self._queues: dict[str, deque[Message]] = {}  # 每 agent 一个 deque
        self._clock = 0  # 全局时钟，给消息编号

    def send(self, sender, receiver, content):
        msg = Message(sender, receiver, content, self._clock)
        self._clock += 1
        self._queues[receiver].append(msg)  # 投递到 receiver 的队列

    def recv(self, agent):
        return self._queues[agent].popleft()  # FIFO 取最早的消息
```

`collections.deque` 的 `append` 和 `popleft` 都是 $O(1)$，适合队列。全局时钟给消息一个全序编号，便于追溯。

### 2.4 队列在编排里的两个角色

1. **消息传递**：agent 间传中间结果（上面的例子）
2. **就绪队列**：拓扑排序的 Kahn 算法里，入度为 0 的 agent 入队等待调度。这是 DAG 和队列的协作点。

## 3. 堆与优先级调度：关键任务先执行

### 3.1 为什么需要优先级

不是所有 agent 同等重要。在博客工作流里：

- `review`（审核）、`publish`（发布）是**关键任务**：用户在等最终结果
- `search`、`research`、`outline`、`draft` 是**主干任务**：必须完成但用户不直接看到
- `citation`、`polish` 是**辅助任务**：锦上添花，可延迟

如果多个 agent 同时就绪，**先执行哪个**？FIFO 队列不区分优先级，关键任务可能排在大量低优任务后面。**堆**（Heap）解决这个问题。

### 3.2 最小堆做优先级调度

用**最小堆**（`heapq`），priority 值小的先执行（1=高 > 2=中 > 3=低）：

```python
@dataclass(order=True)
class Task:
    priority: int   # 1=高, 2=中, 3=低（先比这个）
    seq: int        # 同优先级内 FIFO 的插入序号（再比这个）
    name: str = field(compare=False)

class PriorityScheduler:
    def __init__(self):
        self._heap: list[Task] = []

    def push(self, name, priority):
        heapq.heappush(self._heap, Task(priority, self._seq, name))

    def pop(self):
        return heapq.heappop(self._heap)  # 返回优先级最高（priority 值最小）的任务
```

**复杂度**：`push` 和 `pop` 都是 $O(\log n)$。比 FIFO 的 $O(1)$ 慢，但保证了关键任务先执行。

### 3.3 `seq` 字段：同优先级 FIFO

`Task` 的比较顺序是 `(priority, seq)`：先比优先级，同优先级再比 `seq`（插入序号）。这保证**同优先级的任务 FIFO**——先 push 的先执行。没有 `seq` 的话，同优先级任务的执行顺序不确定（堆不稳定）。

### 3.4 优先级 vs FIFO：关键任务延迟

构造场景：50 个独立任务同时就绪，5 个高优后到达（push 顺序：低优先、高优后）：

| 调度器 | 关键任务执行位置 | 延迟总和 |
|---|---|---|
| 优先级调度（堆） | [0, 1, 2, 3, 4] | 10 步 |
| FIFO 调度（队列） | [45, 46, 47, 48, 49] | 235 步 |

优先级调度让后到达的高优任务**插队**到最前面，延迟仅 10 步；FIFO 让高优任务排在 45 个低优后面，延迟 235 步——**23.5x 差距**。

### 3.5 堆在编排里的位置

堆不是替代队列，而是在"多个就绪 agent"时**选择先执行哪个**。工作流：

```
1. DAG 维护入度，决定哪些 agent 就绪
2. 就绪 agent 进入堆（按优先级排序）   ← 堆在这里
3. 堆 pop 出优先级最高的 agent 执行
4. 执行后通过队列把结果传给后继        ← 队列在这里
5. 后继入度 -1，新的就绪 agent 进堆
```

## 4. 并查集与 agent 分组：近 O(1) 动态合并

### 4.1 agent 为什么要分组

多 agent 系统里，agent 按职能分组：

- **研究组**：search、research、citation（共享资料库）
- **写作组**：outline、draft、polish（共享写作风格）
- **审核组**：review、publish（共享审核标准）

同组 agent 可以**共享上下文/缓存**（如研究组共享搜索结果），跨组通信需要序列化（如研究组把结果传给写作组）。分组还用于：权限控制（同组 agent 互信）、负载均衡（同组 agent 可替换）、计费（按组计费）。

### 4.2 为什么用并查集

分组的核心操作：

- `union(a, b)`：把 a 和 b 分到同一组
- `find(a)`：a 属于哪个组（返回组代表）
- `same_group(a, b)`：a 和 b 是否同组

这些操作**高频**且需要**动态合并**（运行时可能合并两个组）。并查集（Union-Find）是这些操作的最优数据结构：

- `union` 和 `find` 都是 $O(\alpha(n))$ ≈ $O(1)$（$\alpha$ 是反阿克曼函数，增长极慢，$n=10^{80}$ 时 $\alpha < 5$）
- 比把分组存成 `dict[str, set[str]]` 然后 union 时合并两个 set $O(n)$ 快得多
- 支持动态合并：随时 union 两个 agent，不需要重建分组

### 4.3 实现：路径压缩 + 按秩合并

```python
class AgentGroups:
    def __init__(self):
        self._parent: dict[str, str] = {}  # agent → 父节点
        self._rank: dict[str, int] = {}    # 树的高度（秩）

    def find(self, agent):
        # 路径压缩：把查找路径上的节点直接指向根
        root = agent
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[agent] != root:
            self._parent[agent], agent = root, self._parent[agent]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return  # 已同组
        # 按秩合并：矮树挂到高树下
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
```

两个优化让并查集达到 $O(\alpha(n))$：

- **路径压缩**：`find` 时把路径上的节点直接指向根，下次 `find` 是 $O(1)$
- **按秩合并**：`union` 时矮树挂到高树下，避免树变深

### 4.4 并查集在编排里的位置

并查集不参与执行调度，而是**管理 agent 的组织结构**：

- 启动时按职能 `union` 出初始分组
- 运行时可能动态合并组（如研究组和写作组合并成一个"内容组"）
- 查询 `same_group` 决定通信方式（同组共享内存，跨组序列化）

## 5. 四种数据结构如何协作

### 5.1 协作流程

一个完整的多 agent 工作流执行，4 种数据结构协同工作：

```
                    ┌─────────────────────────────────────────┐
                    │           AgentGroups（并查集）           │
                    │   管理 agent 分组，决定通信方式           │
                    │   同组共享上下文，跨组序列化              │
                    └─────────────────────────────────────────┘
                                      │
                                      │ 查询分组
                                      ▼
┌──────────────┐    就绪    ┌──────────────────┐    执行    ┌──────────────┐
│  AgentDAG    │ ─────────→ │ PriorityScheduler│ ─────────→ │  执行 agent  │
│ （邻接表）    │            │ （堆）            │            │              │
│ 维护入度      │            │ 选优先级最高      │            │  产出结果    │
│ 决定就绪      │            │ 的就绪 agent      │            │              │
└──────────────┘            └──────────────────┘            └──────────────┘
                                      ▲                          │
                                      │ 入度 -1                   │ 发消息
                                      │                          ▼
                                      │                 ┌──────────────┐
                                      └─────────────────│ MessageQueue │
                                                        │ （队列）      │
                                                        │ 传结果给后继  │
                                                        └──────────────┘
```

### 5.2 一次执行的步骤

以博客工作流为例，执行 `draft` agent：

1. **DAG**：`draft` 的前置 `[outline]` 都执行完了，入度变 0，`draft` 就绪
2. **堆**：`draft` 进入 `PriorityScheduler`，priority=2（中）
3. **堆**：如果 `citation`（priority=3）和 `polish`（priority=3）也在堆里，`draft` 先出堆（priority 2 < 3）
4. **执行**：`draft` 运行，产出"初稿"
5. **队列**：`draft` 通过 `MessageQueue` 把"初稿"发给后继 `citation` 和 `polish`
6. **DAG**：`citation` 和 `polish` 的入度各 -1，变 0，进入堆
7. **并查集**：`draft` 属于"写作组"，和 `outline`、`polish` 同组，共享写作风格缓存

### 5.3 各数据结构的复杂度

| 数据结构 | 核心操作 | 复杂度 | 在编排里的角色 |
|---|---|---|---|
| DAG（邻接表） | 拓扑排序 | $O(\|V\|+\|E\|)$ | 决定执行顺序 |
| 队列（deque） | push/pop | $O(1)$ | 消息传递 |
| 堆（heapq） | push/pop | $O(\log n)$ | 优先级调度 |
| 并查集 | union/find | $O(\alpha(n))$≈$O(1)$ | 动态分组 |

### 5.4 为什么不能只用一种

- **只用 DAG**：知道顺序但无法传消息、无法按优先级、无法分组
- **只用队列**：能传消息但不知道谁先执行（可能死锁）、无优先级、无分组
- **只用堆**：有优先级但不知道依赖关系（可能前置未完成就执行）、无消息传递、无分组
- **只用并查集**：能分组但无执行顺序、无消息传递、无优先级

4 种数据结构各管一个维度，**组合起来**才完整描述多 agent 编排：DAG 管"什么时候能执行"，堆管"先执行谁"，队列管"执行完怎么传"，并查集管"谁和谁一组"。

## 6. 生产级框架的数据结构选择

| 框架 | 编排核心 | 用到的数据结构 |
|---|---|---|
| **LangGraph** | 图编排（StateGraph） | DAG + 拓扑排序（条件边） |
| **AutoGen** | 消息路由（GroupChat） | 队列（消息传递）+ 路由表 |
| **CrewAI** | 角色分组（Crew） | 并查集（角色分组）+ 任务队列 |
| **Celery** | 任务队列 | 堆（优先级）+ 队列（FIFO） |

生产级框架通常用 2-3 种数据结构的组合，本章 demo 展示了 4 种全协作的完整图景。

## 7. 小结

多智能体编排是 4 种数据结构的**协奏**：

- **DAG + 拓扑排序** → 合法执行顺序（消除死锁和重做）
- **队列** → agent 间消息传递（FIFO 公平）
- **堆** → 优先级调度（保障关键任务先执行）
- **并查集** → 动态分组（近 $O(1)$ 合并）

每种数据结构解决编排的一个维度，组合起来完整描述"多个 AI agent 如何协作完成复杂任务"。这正是扩展专题的精神：**不是一种数据结构优化一个 AI 算法，而是多种数据结构协作解决一个 AI 应用问题**。