"""第 13 章 demo：多智能体编排 × 数据结构协作（DAG + 队列 + 堆 + 并查集）

扩展专题：一个 AI 应用场景（多 agent 协作）→ 多种数据结构协作
- AgentDAG        ：邻接表 + 拓扑排序，决定 agent 执行顺序
- MessageQueue    ：队列，agent 间消息传递（FIFO 公平）
- PriorityScheduler：堆，任务优先级调度
- AgentGroups     ：并查集，agent 分组管理

模拟工作流：写一篇技术博客
  search → research → outline → draft → {citation, polish} → review → publish

性能对比：
1. 有依赖拓扑排序 vs 无依赖乱序（错误率、重做次数）
2. 优先级调度 vs FIFO（关键任务延迟）

跑法：
    python 13_multi_agent/python/demo.py
"""

from __future__ import annotations

import heapq
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. AgentDAG：邻接表 + 拓扑排序
# ============================================================================


class AgentDAG:
    """用邻接表表示 agent 依赖图，拓扑排序决定执行顺序。

    add_edge(u, v) 表示 u 必须先于 v 执行（u 是 v 的前置）。
    内部存两种邻接表：
    - _succ: u → [v...]，u 的所有后继（u 完成后可解锁的 agent）
    - _pred: v → [u...]，v 的所有前置（v 执行前必须完成的 agent）
    拓扑排序用 Kahn 算法（BFS）：每轮取入度为 0 的顶点。
    """

    def __init__(self) -> None:
        self._succ: dict[str, list[str]] = {}
        self._pred: dict[str, list[str]] = {}
        self._nodes: set[str] = set()

    def add_node(self, name: str) -> None:
        self._nodes.add(name)
        self._succ.setdefault(name, [])
        self._pred.setdefault(name, [])

    def add_edge(self, u: str, v: str) -> None:
        """u → v：u 必须先执行。"""
        self.add_node(u)
        self.add_node(v)
        if v not in self._succ[u]:
            self._succ[u].append(v)
        if u not in self._pred[v]:
            self._pred[v].append(u)

    @property
    def nodes(self) -> set[str]:
        return self._nodes

    @property
    def edges(self) -> list[tuple[str, str]]:
        return [(u, v) for u, succs in self._succ.items() for v in succs]

    def successors(self, u: str) -> list[str]:
        return self._succ.get(u, [])

    def predecessors(self, v: str) -> list[str]:
        return self._pred.get(v, [])

    def topological_sort(self) -> list[str]:
        """Kahn 算法拓扑排序。返回一个合法的执行顺序。

        O(V + E)：每个顶点入队一次、出队一次，每条边在删边时检查一次。
        如果存在环，剩余顶点入度永远 > 0，结果长度 < |V|。
        """
        in_degree = {v: len(self._pred[v]) for v in self._nodes}
        queue = deque([v for v in sorted(self._nodes) if in_degree[v] == 0])
        order: list[str] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self._succ[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        if len(order) != len(self._nodes):
            remaining = self._nodes - set(order)
            raise ValueError(f"依赖图有环，无法拓扑排序，涉及节点: {remaining}")
        return order

    def topological_levels(self) -> list[list[str]]:
        """按拓扑层级分组，同层 agent 无依赖关系可并行。

        第 0 层 = 无前置的 agent；第 k 层 = 所有前置都在 0..k-1 层的 agent。
        """
        in_degree = {v: len(self._pred[v]) for v in self._nodes}
        levels: list[list[str]] = []
        current = [v for v in sorted(self._nodes) if in_degree[v] == 0]
        remaining = set(self._nodes)
        while current:
            levels.append(current)
            for u in current:
                remaining.discard(u)
            next_level: list[str] = []
            for v in sorted(remaining):
                if all(u not in remaining for u in self._pred[v]):
                    next_level.append(v)
            current = next_level
        return levels

    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True


# ============================================================================
# 2. MessageQueue：队列，agent 间消息传递
# ============================================================================


@dataclass
class Message:
    """agent 间传递的消息。"""

    sender: str
    receiver: str
    content: str
    timestamp: int  # 全局时钟，用于排序


class MessageQueue:
    """agent 间消息传递，每个 agent 一个 FIFO 队列。

    - send(msg): 把消息投递到 receiver 的队列
    - recv(agent): 取出 agent 队列里最早的消息（FIFO 公平）
    - pending(agent): agent 队列里还有多少消息

    FIFO 保证：先发的消息先被处理，不会饿死早期消息。
    """

    def __init__(self) -> None:
        self._queues: dict[str, deque[Message]] = {}
        self._clock = 0

    def register(self, agent: str) -> None:
        self._queues.setdefault(agent, deque())

    def send(self, sender: str, receiver: str, content: str) -> Message:
        self.register(sender)
        self.register(receiver)
        msg = Message(sender, receiver, content, self._clock)
        self._clock += 1
        self._queues[receiver].append(msg)
        return msg

    def recv(self, agent: str) -> Message | None:
        q = self._queues.get(agent)
        if not q:
            return None
        return q.popleft()

    def pending(self, agent: str) -> int:
        return len(self._queues.get(agent, ()))

    def total_messages(self) -> int:
        return sum(len(q) for q in self._queues.values())


# ============================================================================
# 3. PriorityScheduler：堆，任务优先级调度
# ============================================================================


@dataclass(order=True)
class Task:
    """待调度的任务。priority 小的先执行（最小堆）。"""

    priority: int  # 1=高, 2=中, 3=低
    seq: int  # 同优先级内 FIFO 的插入序号
    name: str = field(compare=False)


class PriorityScheduler:
    """任务优先级调度，用最小堆。

    - push(task): O(log n)
    - pop(): 取出优先级最高（priority 值最小）的任务 O(log n)
    - peek(): 看堆顶 O(1)

    对比 FIFO 队列：FIFO 不区分优先级，关键任务可能排在大量低优任务后面。
    """

    def __init__(self) -> None:
        self._heap: list[Task] = []
        self._seq = 0

    def push(self, name: str, priority: int) -> Task:
        task = Task(priority=priority, seq=self._seq, name=name)
        self._seq += 1
        heapq.heappush(self._heap, task)
        return task

    def pop(self) -> Task | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Task | None:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)


class FIFOScheduler:
    """FIFO 调度器，对比基准。不区分优先级，先入先出。"""

    def __init__(self) -> None:
        self._queue: deque[Task] = deque()
        self._seq = 0

    def push(self, name: str, priority: int) -> Task:
        task = Task(priority=priority, seq=self._seq, name=name)
        self._seq += 1
        self._queue.append(task)
        return task

    def pop(self) -> Task | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)


# ============================================================================
# 4. AgentGroups：并查集，agent 分组管理
# ============================================================================


class AgentGroups:
    """agent 分组管理，用并查集（Union-Find）。

    - union(a, b): 把 a 和 b 分到同一组 O(α(n))
    - find(a): 返回 a 的组代表 O(α(n))
    - same_group(a, b): a 和 b 是否同组
    - groups(): 返回所有分组

    应用：同组 agent 可以共享上下文/缓存，跨组通信需要序列化。
    并查集的近 O(1) 操作让动态分组（运行时合并组）几乎免费。
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, agent: str) -> None:
        if agent not in self._parent:
            self._parent[agent] = agent
            self._rank[agent] = 0

    def find(self, agent: str) -> str:
        self.add(agent)
        # 路径压缩
        root = agent
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[agent] != root:
            self._parent[agent], agent = root, self._parent[agent]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 按秩合并
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def same_group(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> dict[str, list[str]]:
        """返回 {组代表: [成员...]}。"""
        result: dict[str, list[str]] = {}
        for agent in self._parent:
            root = self.find(agent)
            result.setdefault(root, []).append(agent)
        return result


# ============================================================================
# 5. 完整工作流模拟
# ============================================================================


def build_blog_workflow() -> tuple[AgentDAG, dict[str, int], AgentGroups]:
    """构建"写技术博客"的多 agent 工作流。

    8 个 agent，依赖关系形成 DAG，分 3 组，3 个优先级。

    DAG 结构：
        search → research → outline → draft ┬→ citation ┐
                                          └→ polish    ┴→ review → publish
    """
    dag = AgentDAG()
    # 依赖关系（u → v 表示 u 必须先于 v 执行）
    dag.add_edge("search", "research")
    dag.add_edge("research", "outline")
    dag.add_edge("outline", "draft")
    dag.add_edge("draft", "citation")
    dag.add_edge("draft", "polish")
    dag.add_edge("citation", "review")
    dag.add_edge("polish", "review")
    dag.add_edge("review", "publish")

    # 优先级：1=高（关键路径）, 2=中（主干）, 3=低（辅助）
    priorities = {
        "search": 2, "research": 2, "outline": 2, "draft": 2,
        "citation": 3, "polish": 3,
        "review": 1, "publish": 1,
    }

    # 分组：研究组 / 写作组 / 审核组
    groups = AgentGroups()
    for a in priorities:
        groups.add(a)
    groups.union("search", "research")
    groups.union("research", "citation")  # 研究组：search, research, citation
    groups.union("outline", "draft")
    groups.union("draft", "polish")       # 写作组：outline, draft, polish
    groups.union("review", "publish")     # 审核组：review, publish

    return dag, priorities, groups


def execute_with_coordination(
    dag: AgentDAG,
    priorities: dict[str, int],
    use_priority: bool = True,
) -> tuple[list[str], int, int]:
    """模拟多 agent 协作执行，4 种数据结构协同工作。

    调度逻辑（事件驱动）：
    1. DAG 维护入度，决定哪些 agent 就绪
    2. 就绪 agent 进入调度器（堆 or FIFO）
    3. 调度器选出下一个执行的 agent
    4. 执行后通过消息队列通知后继（入度 -1）
    5. 新就绪的 agent 进入调度器

    Args:
        use_priority: True 用堆（优先级调度），False 用 FIFO

    Returns:
        (执行顺序, 关键任务总延迟步数, 总执行步数)
    """
    in_degree = {v: len(dag.predecessors(v)) for v in dag.nodes}
    scheduler = PriorityScheduler() if use_priority else FIFOScheduler()
    # 计算关键任务延迟：就绪时刻 → 执行时刻
    ready_at: dict[str, int] = {}
    clock = 0

    # 初始就绪 agent
    for v in sorted(dag.nodes):
        if in_degree[v] == 0:
            scheduler.push(v, priorities[v])
            ready_at[v] = clock

    order: list[str] = []
    critical_delay = 0

    while len(scheduler) > 0:
        t = scheduler.pop()
        assert t is not None
        agent = t.name
        clock += 1
        order.append(agent)

        # 关键任务延迟 = 执行时刻 - 就绪时刻
        if priorities[agent] == 1:
            critical_delay += clock - ready_at.get(agent, clock)

        # 依赖已满足（入度 0 才进调度器），执行成功，通知后继
        for succ in dag.successors(agent):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                scheduler.push(succ, priorities[succ])
                ready_at[succ] = clock

    return order, critical_delay, clock


def execute_chaos(dag: AgentDAG, seed: int = 0) -> tuple[list[str], int]:
    """乱序执行：随机打乱所有 agent，不管依赖逐个尝试。

    如果 agent 的依赖未满足，放到末尾重试。
    这模拟"没有 DAG 拓扑排序，随机调度"的灾难。

    Returns:
        (执行顺序, 总尝试次数)
    """
    rng = random.Random(seed)
    queue = list(dag.nodes)
    rng.shuffle(queue)

    executed: set[str] = set()
    order: list[str] = []
    attempts = 0

    while queue:
        agent = queue.pop(0)
        attempts += 1
        preds = dag.predecessors(agent)
        if all(p in executed for p in preds):
            executed.add(agent)
            order.append(agent)
        else:
            # 依赖未满足，放到末尾重试
            queue.append(agent)

    return order, attempts


def run_full_workflow_trace(
    dag: AgentDAG, priorities: dict[str, int], groups: AgentGroups,
) -> list[str]:
    """完整工作流追踪：4 种数据结构协作的详细日志。"""
    log: list[str] = []
    mq = MessageQueue()

    log.append("  步骤 1：拓扑排序决定执行顺序（DAG + Kahn 算法）")
    order = dag.topological_sort()
    log.append(f"    执行顺序 = {order}")

    log.append("  步骤 2：按拓扑序执行，消息队列传递中间结果")
    for agent in order:
        preds = dag.predecessors(agent)
        received: list[Message] = []
        for _ in range(len(preds)):
            msg = mq.recv(agent)
            if msg:
                received.append(msg)
        succs = dag.successors(agent)
        group = groups.find(agent)
        log.append(
            f"    [{agent}] 组={group} 优先级={priorities[agent]} "
            f"收到 {len(received)} 条消息 → 发给 {succs if succs else '（无后继，完成）'}"
        )
        for succ in succs:
            mq.send(agent, succ, f"{agent} 的输出结果")

    log.append("  步骤 3：并查集验证分组（同组 agent 可共享上下文）")
    for root, members in groups.groups().items():
        log.append(f"    组 {root}: {sorted(members)}")

    return log


# ============================================================================
# 6. 性能对比
# ============================================================================


def benchmark_order_vs_chaos(dag: AgentDAG, priorities: dict[str, int]) -> dict:
    """对比：有依赖排序 vs 无依赖乱序。

    乱序执行会让 agent 在依赖未满足时尝试，必须重试。
    拓扑排序保证一次成功，乱序需要多次尝试。
    """
    n_trials = 200
    attempts_ordered = 0
    attempts_chaos = 0
    n_agents = len(dag.nodes)

    for seed in range(n_trials):
        # 拓扑排序：一次成功，尝试次数 = agent 数
        order, _, _ = execute_with_coordination(dag, priorities, use_priority=True)
        attempts_ordered += len(order)

        # 乱序：可能需要多次尝试
        _, attempts = execute_chaos(dag, seed=seed)
        attempts_chaos += attempts

    return {
        "n_trials": n_trials,
        "n_agents": n_agents,
        "attempts_ordered": attempts_ordered,
        "attempts_chaos": attempts_chaos,
        "redo_chaos": attempts_chaos - attempts_ordered,
    }


def benchmark_priority_vs_fifo(
    dag: AgentDAG, priorities: dict[str, int], n_agents: int = 50,
) -> dict:
    """对比：优先级调度 vs FIFO。

    构造一个有多个就绪 agent 的场景：
    - 50 个独立任务（无依赖），同时就绪
    - 5 个高优（priority=1），45 个低优（priority=3）
    - push 顺序：低优先 push，高优后 push（模拟高优任务后到达）
    - 优先级调度：高优先执行（尽管后到）
    - FIFO 调度：先 push 的先执行（高优被低优阻塞）
    """
    # 构造独立任务：低优先 push，高优后 push
    push_order: list[tuple[str, int]] = []
    for i in range(n_agents - 5):
        push_order.append((f"low_{i}", 3))  # 低优先到
    for i in range(5):
        push_order.append((f"high_{i}", 1))  # 高优后到

    # 优先级调度
    ps = PriorityScheduler()
    for name, pri in push_order:
        ps.push(name, pri)
    order_p: list[str] = []
    while len(ps) > 0:
        t = ps.pop()
        assert t is not None
        order_p.append(t.name)

    # FIFO 调度
    fs = FIFOScheduler()
    for name, pri in push_order:
        fs.push(name, pri)
    order_f: list[str] = []
    while len(fs) > 0:
        t = fs.pop()
        assert t is not None
        order_f.append(t.name)

    # 关键任务（高优）的执行位置：位置越大延迟越多
    high_pos_p = [i for i, a in enumerate(order_p) if a.startswith("high")]
    high_pos_f = [i for i, a in enumerate(order_f) if a.startswith("high")]

    return {
        "n_agents": n_agents,
        "n_critical": 5,
        "order_priority": order_p,
        "order_fifo": order_f,
        "high_pos_priority": high_pos_p,
        "high_pos_fifo": high_pos_f,
        "delay_priority": sum(high_pos_p),
        "delay_fifo": sum(high_pos_f),
    }


def benchmark_ds_scaling() -> dict:
    """4 种数据结构随规模增长的操作耗时。

    验证：
    - DAG 拓扑排序 O(V+E)
    - 队列 push/pop O(1)
    - 堆 push/pop O(log n)
    - 并查集 union/find O(α(n)) ≈ O(1)
    """
    sizes = [100, 500, 1000, 5000, 10000]
    results: dict[str, list[float]] = {
        "DAG 拓扑排序": [],
        "队列 push/pop": [],
        "堆 push/pop": [],
        "并查集 union": [],
    }

    for n in sizes:
        # DAG：链式依赖 0→1→2→...→n-1
        dag = AgentDAG()
        for i in range(n - 1):
            dag.add_edge(f"a{i}", f"a{i + 1}")
        with Timer() as t:
            dag.topological_sort()
        results["DAG 拓扑排序"].append(t.elapsed_ms)

        # 队列
        q: deque[int] = deque()
        with Timer() as t:
            for i in range(n):
                q.append(i)
            for _ in range(n):
                q.popleft()
        results["队列 push/pop"].append(t.elapsed_ms)

        # 堆
        h: list[int] = []
        with Timer() as t:
            for i in range(n):
                heapq.heappush(h, i)
            for _ in range(n):
                heapq.heappop(h)
        results["堆 push/pop"].append(t.elapsed_ms)

        # 并查集
        uf = AgentGroups()
        with Timer() as t:
            for i in range(n):
                uf.add(f"u{i}")
            for i in range(n - 1):
                uf.union(f"u{i}", f"u{i + 1}")
            for i in range(n):
                uf.find(f"u{i}")
        results["并查集 union"].append(t.elapsed_ms)

    return {"sizes": sizes, "results": results}


# ============================================================================
# 7. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 13 章 demo：多智能体编排 × 数据结构协作")
    print("（DAG + 队列 + 堆 + 并查集）")
    print("=" * 72)

    print("\n[1] 构建 agent 依赖 DAG：写技术博客工作流")
    print("-" * 72)
    dag, priorities, groups = build_blog_workflow()
    print(f"  agent 数 = {len(dag.nodes)}: {sorted(dag.nodes)}")
    print(f"  依赖边数 = {len(dag.edges)}")
    print("  依赖关系：")
    for u, v in dag.edges:
        print(f"    {u:>10} → {v}")
    print(f"  有环？{dag.has_cycle()}（必须是 DAG 才能拓扑排序）")

    print("\n[2] 拓扑排序决定执行顺序（Kahn 算法 O(V+E)）")
    print("-" * 72)
    order = dag.topological_sort()
    print(f"  合法执行顺序 = {order}")
    levels = dag.topological_levels()
    print(f"  拓扑层级（同层可并行）共 {len(levels)} 层：")
    for i, lvl in enumerate(levels):
        print(f"    层 {i}: {lvl}")
    print(f"  → 关键路径长度 = {len(levels)} 步（最少串行步数）")

    print("\n[3] 消息队列：agent 间通信（FIFO 公平）")
    print("-" * 72)
    mq = MessageQueue()
    # 模拟前 3 个 agent 执行并发消息
    for agent in order[:3]:
        for succ in dag.successors(agent):
            msg = mq.send(agent, succ, f"{agent} 的研究结果")
            print(f"    {agent} → {succ}: \"{msg.content}\" (t={msg.timestamp})")
    print(f"  各 agent 待处理消息数：")
    for a in sorted(dag.nodes):
        p = mq.pending(a)
        if p > 0:
            print(f"    {a:>10}: {p} 条")
    print(f"  总消息数 = {mq.total_messages()}")

    print("\n[4] 优先级调度：堆 vs FIFO")
    print("-" * 72)
    print(f"  优先级分配：1=高（关键）, 2=中（主干）, 3=低（辅助）")
    for a in sorted(priorities):
        label = {1: "高", 2: "中", 3: "低"}[priorities[a]]
        print(f"    {a:>10}: 优先级 {priorities[a]} ({label})")
    # 用优先级调度执行
    order_p, delay_p, steps_p = execute_with_coordination(
        dag, priorities, use_priority=True
    )
    print(f"  优先级调度执行顺序 = {order_p}")
    print(f"  关键任务延迟 = {delay_p} 步（总执行 {steps_p} 步）")

    print("\n[5] 并查集：agent 分组管理（近 O(1) 动态合并）")
    print("-" * 72)
    print(f"  分组结果（同组 agent 可共享上下文/缓存）：")
    for root, members in groups.groups().items():
        print(f"    组 {root}: {sorted(members)}")
    print(f"  跨组通信检查：")
    test_pairs = [("search", "research"), ("search", "draft"), ("review", "publish"),
                  ("outline", "review")]
    for a, b in test_pairs:
        same = groups.same_group(a, b)
        print(f"    {a} 与 {b} 同组？{same}")

    print("\n[6] 完整工作流模拟：4 种数据结构协作")
    print("-" * 72)
    trace = run_full_workflow_trace(dag, priorities, groups)
    for line in trace:
        print(line)

    print("\n[7] 性能对比：有依赖排序 vs 无依赖乱序")
    print("-" * 72)
    res_order = benchmark_order_vs_chaos(dag, priorities)
    print(f"  试验次数 = {res_order['n_trials']}，agent 数 = {res_order['n_agents']}")
    print(f"  有依赖排序：总尝试次数 = {res_order['attempts_ordered']} "
          f"（每轮恰好 {res_order['n_agents']} 次，0 次重试）")
    print(f"  无依赖乱序：总尝试次数 = {res_order['attempts_chaos']} "
          f"（重试 {res_order['redo_chaos']} 次）")
    if res_order["redo_chaos"] > 0:
        ratio = res_order["attempts_chaos"] / res_order["attempts_ordered"]
        print(f"  → 乱序尝试次数是拓扑排序的 {ratio:.2f}x，"
              f"拓扑排序消除 {res_order['redo_chaos']} 次无效尝试")

    print("\n[8] 性能对比：优先级调度 vs FIFO（高优任务后到达场景）")
    print("-" * 72)
    res_sched = benchmark_priority_vs_fifo(dag, priorities, n_agents=50)
    print(f"  场景：{res_sched['n_agents']} 个独立任务同时就绪，"
          f"其中 {res_sched['n_critical']} 个关键任务（priority=1）后到达")
    print(f"  优先级调度：关键任务执行位置 = {res_sched['high_pos_priority']}")
    print(f"  FIFO 调度  ：关键任务执行位置 = {res_sched['high_pos_fifo']}")
    print(f"  优先级调度：关键任务延迟总和 = {res_sched['delay_priority']} 步")
    print(f"  FIFO 调度  ：关键任务延迟总和 = {res_sched['delay_fifo']} 步")
    if res_sched["delay_fifo"] > res_sched["delay_priority"]:
        ratio = res_sched["delay_fifo"] / max(res_sched["delay_priority"], 1)
        print(f"  → FIFO 让关键任务多等 {ratio:.1f}x，"
              f"优先级调度让关键任务插队执行")

    print("\n[9] 4 种数据结构随规模增长的耗时")
    print("-" * 72)
    res_scale = benchmark_ds_scaling()
    print(f"  {'规模':>8} | " + " | ".join(f"{k:>14}" for k in res_scale["results"]))
    print("  " + "-" * 70)
    for i, n in enumerate(res_scale["sizes"]):
        row = f"  {n:>8} | " + " | ".join(
            f"{res_scale['results'][k][i]:>12.4f}ms" for k in res_scale["results"]
        )
        print(row)
    print("  解读：")
    print("  - DAG 拓扑排序 O(V+E)：随规模线性增长")
    print("  - 队列 O(1)：增长最慢，纯内存操作")
    print("  - 堆 O(log n)：比队列稍慢，但优先级保障关键任务")
    print("  - 并查集 O(α(n))≈O(1)：几乎不随规模增长")

    print("\n[10] 保存对比图到 figures/")
    print("-" * 72)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    # 图 1：尝试次数对比
    save_bar(
        {
            "有依赖排序": float(res_order["attempts_ordered"]),
            "无依赖乱序": float(res_order["attempts_chaos"]),
        },
        fig_dir / "attempts_ordered_vs_chaos.png",
        title=f"总尝试次数对比（{res_order['n_trials']} 次试验，{res_order['n_agents']} agent）",
        ylabel="总尝试次数",
        baseline="有依赖排序",
    )
    print(f"  ✓ 保存 figures/attempts_ordered_vs_chaos.png")

    # 图 2：关键任务延迟对比
    save_bar(
        {
            "优先级调度": float(res_sched["delay_priority"]),
            "FIFO 调度": float(res_sched["delay_fifo"]),
        },
        fig_dir / "delay_priority_vs_fifo.png",
        title=f"关键任务延迟对比（{res_sched['n_agents']} agent 宽 DAG）",
        ylabel="延迟步数",
        baseline="优先级调度",
    )
    print(f"  ✓ 保存 figures/delay_priority_vs_fifo.png")

    # 图 3：4 种数据结构随规模增长
    save_line(
        res_scale["results"],
        fig_dir / "ds_scaling.png",
        title="4 种数据结构操作耗时 vs 规模",
        ylabel="耗时 (ms)",
        xlabel="规模 (100 / 500 / 1k / 5k / 10k)",
    )
    print(f"  ✓ 保存 figures/ds_scaling.png")

    # 图 4：拓扑层级（每个 agent 的层级）
    level_map: dict[str, int] = {}
    for i, lvl in enumerate(levels):
        for a in lvl:
            level_map[a] = i
    save_bar(
        {a: float(level_map[a]) for a in sorted(level_map)},
        fig_dir / "topo_levels.png",
        title="agent 拓扑层级（同层可并行）",
        ylabel="层级",
        xlabel="agent",
    )
    print(f"  ✓ 保存 figures/topo_levels.png")

    print("\n" + "=" * 72)
    print("结论：多智能体编排是 4 种数据结构的协奏——")
    print("  DAG 拓扑排序 → 合法执行顺序（消除重做）")
    print("  消息队列     → agent 间通信（FIFO 公平）")
    print("  堆           → 优先级调度（保障关键路径）")
    print("  并查集       → 动态分组（近 O(1) 合并）")
    print("生产级框架 LangGraph 用图编排 agent，AutoGen 用消息路由，")
    print("CrewAI 用角色分组——本质都是这 4 种数据结构的不同组合。")
    print("=" * 72)


if __name__ == "__main__":
    main()