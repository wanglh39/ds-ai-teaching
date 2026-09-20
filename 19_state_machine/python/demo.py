"""第 19 章 demo：状态机与工作流编排 × 数据结构协作（图 + 栈 + 队列 + 树）

扩展专题：一个 AI 应用场景（AI 智能体的状态机驱动工作流）
→ 多种数据结构协作
- StateMachine  : 有限状态机（状态转移图 + 状态哈希，模拟 LangGraph StateGraph）
- ReActLoop     : ReAct 循环（栈记录 think→act→observe 推理链）
- PlanAndExecute: Plan-and-Execute（队列存任务列表，支持动态重规划）
- TreeOfThought : Tree-of-Thought（树存思维分支，BFS/DFS 搜索 + 剪枝）

模拟 4 种 agent 工作流模式：
  用户提问 → [图] 状态机驱动状态转移
           → [栈] ReAct 循环记录推理链
           → [队列] Plan-Execute 任务队列
           → [树] ToT 多分支搜索最优思维

性能对比：
1. 状态机 vs 硬编码 if-else（可维护性、扩展性）
2. ReAct 栈 vs 无栈（回溯能力）
3. Plan-Execute 队列 vs 栈（FIFO 任务执行顺序）
4. ToT 树搜索 vs 线性（找到最优解的探索数）

跑法：
    python 19_state_machine/python/demo.py
"""

from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. StateMachine：有限状态机（状态转移图 + 状态哈希）
# ============================================================================


@dataclass
class Transition:
    """状态转移边：从 source 到 target，带条件函数。

    对应 LangGraph 的 conditional edge：
        graph.add_conditional_edges("node", router_fn, {"yes": "A", "no": "B"})
    """

    source: str
    target: str
    condition: str  # 条件名（可读性）
    guard: callable  # 条件函数：state -> bool


class StateMachine:
    """有限状态机（FSM）：状态转移图 + 状态哈希。

    AI 场景：LangGraph 的 StateGraph 把 agent 工作流建模为状态机。
    - 节点 = 状态（idle / planning / acting / observing / done / error）
    - 边   = 状态转移（含条件转移）
    - 状态存储 = 哈希表（state dict）

    数据结构协作：
    - 图（邻接表）：存状态转移拓扑
    - 哈希表：存当前状态变量
    """

    def __init__(self, initial: str = "idle") -> None:
        # 状态转移图：邻接表 {source: [Transition, ...]}
        self.graph: dict[str, list[Transition]] = {}
        # 状态哈希：当前状态变量（LangGraph 的 state dict）
        self.state: dict[str, object] = {"current": initial}
        # 执行历史（用于可视化）
        self.history: list[str] = []

    def add_transition(self, src: str, tgt: str, condition: str, guard: callable) -> None:
        self.graph.setdefault(src, []).append(Transition(src, tgt, condition, guard))

    def step(self) -> str:
        """执行一步状态转移，返回新状态。

        遍历当前状态的所有出边，第一个 guard 满足的就转移。
        这就是 LangGraph 的 router 逻辑。
        """
        cur = self.state["current"]
        self.history.append(cur)
        for trans in self.graph.get(cur, []):
            if trans.guard(self.state):
                self.state["current"] = trans.target
                return trans.target
        # 无转移可用，停留在当前状态
        return cur

    def run(self, max_steps: int = 50) -> list[str]:
        """运行到终止状态（done / error）或 max_steps。"""
        for _ in range(max_steps):
            cur = self.state["current"]
            if cur in ("done", "error"):
                break
            self.step()
        return self.history


def build_agent_state_machine() -> StateMachine:
    """构建一个 agent 工作流状态机。

    状态：idle → planning → acting → observing → done / error

    条件转移：
    - idle → planning : 有用户输入
    - planning → acting : 生成了计划
    - acting → observing : 动作执行完
    - observing → acting : 还需要更多步骤（迭代）
    - observing → done : 任务完成
    - observing → error : 出错
    - error → idle : 重试
    """
    sm = StateMachine(initial="idle")

    # 模拟 agent 的状态变量
    sm.state["has_input"] = True
    sm.state["plan_ready"] = False
    sm.state["action_done"] = False
    sm.state["steps_done"] = 0
    sm.state["task_complete"] = False
    sm.state["has_error"] = False
    sm.state["max_steps"] = 5

    def has_input(s): return s["has_input"]
    def plan_generated(s): return s["plan_ready"]
    def action_executed(s): return s["action_done"]
    def need_more_steps(s): return not s["task_complete"] and not s["has_error"] and s["steps_done"] < s["max_steps"]
    def task_complete(s): return s["task_complete"]
    def has_error(s): return s["has_error"]

    sm.add_transition("idle", "planning", "有用户输入", has_input)
    sm.add_transition("planning", "acting", "计划已生成", plan_generated)
    sm.add_transition("acting", "observing", "动作执行完", action_executed)
    sm.add_transition("observing", "acting", "需要更多步骤", need_more_steps)
    sm.add_transition("observing", "done", "任务完成", task_complete)
    sm.add_transition("observing", "error", "出错", has_error)
    sm.add_transition("error", "idle", "重试", lambda s: True)

    return sm


def simulate_state_machine(sm: StateMachine) -> list[str]:
    """模拟状态机执行，每步更新状态变量。"""
    sm.state["current"] = "idle"
    sm.history.clear()

    for _ in range(20):
        cur = sm.state["current"]
        if cur in ("done", "error"):
            break

        # 在进入某状态时，模拟 agent 的"动作"更新状态变量
        if cur == "idle":
            sm.state["has_input"] = True
        elif cur == "planning":
            sm.state["plan_ready"] = True
        elif cur == "acting":
            sm.state["action_done"] = True
            sm.state["steps_done"] = sm.state["steps_done"] + 1
        elif cur == "observing":
            sm.state["action_done"] = False
            # 模拟：做完 max_steps 步就完成
            if sm.state["steps_done"] >= sm.state["max_steps"]:
                sm.state["task_complete"] = True

        sm.step()

    return sm.history


def hardcoded_agent_workflow(steps: int = 5) -> list[str]:
    """硬编码 if-else 的 agent 工作流（对比基准）。

    没有状态机抽象，所有逻辑写死在函数里。
    难以扩展、难以可视化、难以并行。
    """
    history = []
    has_input = True
    plan_ready = False
    action_done = False
    steps_done = 0
    task_complete = False

    # 硬编码：idle
    if has_input:
        history.append("idle")
        # planning
        plan_ready = True
        history.append("planning")
        # acting + observing 循环
        while not task_complete and steps_done < steps:
            action_done = True
            steps_done += 1
            history.append("acting")
            action_done = False
            history.append("observing")
            if steps_done >= steps:
                task_complete = True
        if task_complete:
            history.append("done")

    return history


# ============================================================================
# 2. ReActLoop：ReAct 循环（栈记录推理链）
# ============================================================================


@dataclass
class ReActStep:
    """ReAct 的一步：think / act / observe。"""

    kind: str  # "think" / "act" / "observe"
    content: str
    step_id: int


class ReActLoop:
    """ReAct 循环：Reasoning + Acting。

    AI 场景：ReAct 是最经典的 agent 推理模式。
    - think  : LLM 推理，决定下一步动作
    - act    : 执行动作（调用工具 / 检索 / 计算）
    - observe: 观察动作结果
    - 循环直到得出答案

    数据结构：栈
    - 推理链是后进先出的回溯结构
    - 出错时可以 pop 回上一个 think 重新推理
    - 嵌套子任务时压栈，子任务完成弹栈
    """

    def __init__(self) -> None:
        # 推理栈：记录 think→act→observe 链
        self.stack: list[ReActStep] = []
        self.step_counter: int = 0
        self.max_iters: int = 10

    def think(self, thought: str) -> ReActStep:
        """推理：生成下一步动作的想法。"""
        self.step_counter += 1
        step = ReActStep("think", thought, self.step_counter)
        self.stack.append(step)
        return step

    def act(self, action: str) -> ReActStep:
        """执行动作。"""
        self.step_counter += 1
        step = ReActStep("act", action, self.step_counter)
        self.stack.append(step)
        return step

    def observe(self, result: str) -> ReActStep:
        """观察结果。"""
        self.step_counter += 1
        step = ReActStep("observe", result, self.step_counter)
        self.stack.append(step)
        return step

    def backtrack(self) -> ReActStep | None:
        """回溯：弹出最近的一步，用于错误恢复。

        这是栈的关键能力——硬编码的线性流程无法回溯。
        """
        if self.stack:
            return self.stack.pop()
        return None

    def run(self, task: str) -> list[ReActStep]:
        """模拟 ReAct 循环解决一个任务。"""
        self.stack.clear()
        self.step_counter = 0

        # 模拟一个简单的 ReAct 循环：计算 "巴黎的人口 + 100万"
        self.think(f"任务：{task}，需要先查巴黎人口")
        self.act("search(巴黎人口)")
        self.observe("巴黎人口约 210 万")

        self.think("210 万 + 100 万 = 310 万，但需要验证单位")
        self.act("calculate(210 + 100)")
        self.observe("310")

        self.think("单位是万，所以答案是 310 万")
        # 模拟一次错误 + 回溯
        self.act("format_answer(310万)")
        self.observe("格式错误：缺少单位说明")

        # 回溯：弹出错误的 act + observe，重新推理
        self.backtrack()  # pop observe
        self.backtrack()  # pop act

        self.think("重新格式化，加上单位说明")
        self.act("format_answer(310 万人)")
        self.observe("格式正确：310 万人")

        return list(self.stack)


def react_no_stack(task: str) -> list[str]:
    """无栈的 ReAct（对比基准）：无法回溯，出错只能从头来。

    硬编码的线性流程，没有栈结构，遇到错误只能重跑。
    """
    steps = []
    steps.append(f"think: 任务={task}")
    steps.append("act: search")
    steps.append("observe: 210 万")
    steps.append("think: 210+100=310")
    steps.append("act: format")
    steps.append("observe: 格式错误")  # 出错
    # 无栈：只能从头重跑
    steps.append("think: 重跑整个流程")
    steps.append("act: search")
    steps.append("observe: 210 万")
    steps.append("think: 310")
    steps.append("act: format 正确")
    steps.append("observe: 310 万人")
    return steps


# ============================================================================
# 3. PlanAndExecute：Plan-and-Execute（队列存任务列表）
# ============================================================================


@dataclass
class Task:
    """计划中的一个任务。"""

    id: int
    name: str
    status: str = "pending"  # pending / running / done / failed
    result: str = ""


class PlanAndExecute:
    """Plan-and-Execute：planner 生成任务队列，executor 逐个执行。

    AI 场景：复杂任务先拆解成子任务，再逐个执行。
    - planner : LLM 生成任务列表（分解）
    - executor: 逐个执行子任务
    - replan  : 执行结果不理想时动态重新规划

    数据结构：队列
    - FIFO：先规划的任务先执行（保持依赖顺序）
    - 动态入队：执行中发现新任务可以追加
    - 对比栈：LIFO 会先执行最后规划的，破坏依赖
    """

    def __init__(self) -> None:
        # 任务队列
        self.queue: deque[Task] = deque()
        self.completed: list[Task] = []
        self.task_counter: int = 0

    def plan(self, task_names: list[str]) -> None:
        """planner：生成任务队列。"""
        for name in task_names:
            self.task_counter += 1
            self.queue.append(Task(self.task_counter, name))

    def execute_one(self) -> Task | None:
        """executor：执行队列头部的任务（FIFO）。"""
        if not self.queue:
            return None
        task = self.queue.popleft()
        task.status = "running"
        # 模拟执行
        task.result = f"done:{task.name}"
        task.status = "done"
        self.completed.append(task)
        return task

    def replan(self, new_tasks: list[str]) -> None:
        """动态重新规划：把新任务追加到队列尾部。"""
        for name in new_tasks:
            self.task_counter += 1
            self.queue.append(Task(self.task_counter, name))

    def run(self, initial_plan: list[str], replan_at: int | None = None,
            replan_tasks: list[str] | None = None) -> list[Task]:
        """运行 Plan-and-Execute。"""
        self.plan(initial_plan)
        executed = 0
        while self.queue:
            task = self.execute_one()
            if task is None:
                break
            executed += 1
            # 在执行到第 replan_at 个任务时，动态重新规划
            if replan_at is not None and executed == replan_at and replan_tasks:
                self.replan(replan_tasks)
        return self.completed


def plan_execute_with_stack(initial_plan: list[str]) -> list[Task]:
    """用栈代替队列执行 Plan-Execute（对比基准）。

    栈是 LIFO，会先执行最后规划的任务，破坏依赖顺序。
    """
    stack: list[Task] = []
    completed: list[Task] = []
    counter = 0
    for name in initial_plan:
        counter += 1
        stack.append(Task(counter, name))
    while stack:
        task = stack.pop()  # LIFO：最后入栈的先执行
        task.status = "done"
        task.result = f"done:{task.name}"
        completed.append(task)
    return completed


# ============================================================================
# 4. TreeOfThought：Tree-of-Thought（树存思维分支）
# ============================================================================


@dataclass
class ThoughtNode:
    """思维树的一个节点：一个思维状态 + 评分 + 子分支。"""

    thought: str
    score: float = 0.0
    children: list["ThoughtNode"] = field(default_factory=list)
    visited: bool = False
    pruned: bool = False

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class TreeOfThought:
    """Tree-of-Thought：多分支思维搜索。

    AI 场景：复杂推理时，不只走一条思维链，而是分支多个可能的方向，
    评估每个方向的潜力，搜索最优思维路径。
    - 生成：每个节点展开 k 个候选下一步思维
    - 评估：给每个思维打分
    - 搜索：BFS / DFS 遍历思维树
    - 剪枝：低分分支直接砍掉，减少搜索空间

    数据结构：树
    - 层次结构：思维从粗到细，自然形成树
    - 分支：每个节点多个子思维
    - 剪枝：标记子树不展开
    """

    def __init__(self, branching: int = 3, max_depth: int = 4,
                 prune_threshold: float = 0.3) -> None:
        self.root: ThoughtNode | None = None
        self.branching = branching  # 每步展开几个分支
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold
        self.nodes_expanded: int = 0
        self.nodes_pruned: int = 0

    def expand(self, node: ThoughtNode, depth: int) -> None:
        """展开一层的子思维。"""
        if depth >= self.max_depth:
            return
        self.nodes_expanded += 1
        # 模拟 LLM 生成 branching 个候选思维，随机评分
        for i in range(self.branching):
            score = random.random()
            thought = f"{node.thought} → 分支{i+1}(d={depth+1})"
            child = ThoughtNode(thought, score=score)
            node.children.append(child)

        # 剪枝：低分子树不展开
        for child in node.children:
            if child.score < self.prune_threshold:
                child.pruned = True
                self.nodes_pruned += 1
            else:
                self.expand(child, depth + 1)

    def build(self, root_thought: str) -> ThoughtNode:
        """构建思维树。

        根节点 score 设为 0.0，让叶节点中评分最高的被 BFS/DFS 找到，
        体现搜索的价值。
        """
        random.seed(42)
        self.root = ThoughtNode(root_thought, score=0.0)
        self.nodes_expanded = 0
        self.nodes_pruned = 0
        self.expand(self.root, 0)
        return self.root

    def bfs_best(self) -> tuple[ThoughtNode, int]:
        """BFS 搜索评分最高的叶节点。

        返回 (最佳节点, 访问节点数)。
        BFS 按层搜索，适合找浅层最优解。
        """
        if self.root is None:
            raise RuntimeError("树未构建")
        visited = 0
        best = self.root
        queue: deque[ThoughtNode] = deque([self.root])
        while queue:
            node = queue.popleft()
            node.visited = True
            visited += 1
            if node.pruned:
                continue
            if node.is_leaf() and node.score > best.score:
                best = node
            for child in node.children:
                queue.append(child)
        return best, visited

    def dfs_best(self) -> tuple[ThoughtNode, int]:
        """DFS 搜索评分最高的叶节点。"""
        if self.root is None:
            raise RuntimeError("树未构建")
        visited = 0
        best = self.root
        stack: list[ThoughtNode] = [self.root]
        while stack:
            node = stack.pop()
            node.visited = True
            visited += 1
            if node.pruned:
                continue
            if node.is_leaf() and node.score > best.score:
                best = node
            # 逆序入栈，保证左到右遍历
            for child in reversed(node.children):
                stack.append(child)
        return best, visited


def linear_thought_search(num_thoughts: int = 50) -> tuple[float, int]:
    """线性搜索（对比基准）：不分支，逐个尝试。

    没有 tree 结构，只能线性枚举所有可能。
    """
    random.seed(42)
    best_score = 0.0
    visited = 0
    for _ in range(num_thoughts):
        score = random.random()
        visited += 1
        if score > best_score:
            best_score = score
    return best_score, visited


# ============================================================================
# 5. 性能对比
# ============================================================================


def bench_state_machine_vs_hardcode() -> dict[str, dict[str, float]]:
    """对比：状态机 vs 硬编码 if-else。"""
    sm = build_agent_state_machine()

    def run_sm():
        simulate_state_machine(sm)

    def run_hardcode():
        hardcoded_agent_workflow(steps=5)

    return compare({"状态机": run_sm, "硬编码 if-else": run_hardcode}, repeat=20)


def bench_react_stack_vs_nostack() -> dict[str, dict[str, float]]:
    """对比：ReAct 栈 vs 无栈。"""
    react = ReActLoop()

    def run_stack():
        react.run("计算巴黎人口+100万")

    def run_nostack():
        react_no_stack("计算巴黎人口+100万")

    return compare({"ReAct 栈": run_stack, "无栈线性": run_nostack}, repeat=20)


def bench_plan_queue_vs_stack() -> dict[str, dict[str, float]]:
    """对比：Plan-Execute 队列 vs 栈。"""
    plan = list(range(20))

    def run_queue():
        pe = PlanAndExecute()
        pe.run([f"task_{i}" for i in plan])

    def run_stack():
        plan_execute_with_stack([f"task_{i}" for i in plan])

    return compare({"队列(FIFO)": run_queue, "栈(LIFO)": run_stack}, repeat=20)


def bench_tot_tree_vs_linear() -> dict[str, dict[str, float]]:
    """对比：ToT 树搜索 vs 线性搜索。"""
    tot = TreeOfThought(branching=3, max_depth=4, prune_threshold=0.3)

    def run_tree():
        tot.build("问题：如何分配 100 万预算到 3 个项目？")
        tot.bfs_best()

    def run_linear():
        linear_thought_search(num_thoughts=80)

    return compare({"ToT 树搜索": run_tree, "线性枚举": run_linear}, repeat=10)


def measure_state_machine_metrics() -> dict[str, int]:
    """状态机 vs 硬编码的可维护性指标。"""
    sm = build_agent_state_machine()
    hist_sm = simulate_state_machine(sm)
    hist_hc = hardcoded_agent_workflow(steps=5)

    # 状态机：新增一个状态只需 add_transition，不改 run 逻辑
    # 硬编码：新增一个状态需要改 if-else 嵌套
    return {
        "状态机状态数": len(set(sm.graph.keys()) | {t.target for ts in sm.graph.values() for t in ts}),
        "状态机转移数": sum(len(ts) for ts in sm.graph.values()),
        "状态机历史长度": len(hist_sm),
        "硬编码历史长度": len(hist_hc),
        "状态机新增状态成本": 1,  # 只需 add_transition 一行
        "硬编码新增状态成本": 3,  # 需要改 if-else 嵌套 ~3 行
    }


def measure_react_metrics() -> dict[str, int]:
    """ReAct 栈 vs 无栈的能力指标。"""
    react = ReActLoop()
    steps = react.run("计算巴黎人口+100万")
    nostack = react_no_stack("计算巴黎人口+100万")

    return {
        "栈推理步数": len(steps),
        "无栈推理步数": len(nostack),
        "栈回溯能力": 1,  # 可以 backtrack()
        "无栈回溯能力": 0,  # 无法回溯，只能重跑
        "栈出错恢复步数": 2,  # pop 2 步后重新推理
        "无栈出错恢复步数": len(nostack),  # 重跑整个流程
    }


def measure_plan_metrics() -> dict[str, int]:
    """Plan-Execute 队列 vs 栈的顺序指标。"""
    pe = PlanAndExecute()
    plan = ["1.理解问题", "2.收集数据", "3.分析", "4.生成方案", "5.验证"]
    completed_queue = pe.run(plan)

    completed_stack = plan_execute_with_stack(plan)

    # 队列：按计划顺序执行（1→2→3→4→5）
    # 栈：反序执行（5→4→3→2→1），破坏依赖
    queue_order_ok = all(
        completed_queue[i].name == plan[i] for i in range(len(plan))
    )
    stack_order_ok = all(
        completed_stack[i].name == plan[i] for i in range(len(plan))
    )

    return {
        "队列执行顺序正确": int(queue_order_ok),
        "栈执行顺序正确": int(stack_order_ok),
        "队列第一个执行": int(completed_queue[0].name == plan[0]),
        "栈第一个执行": int(completed_stack[0].name == plan[0]),
        "队列支持动态重规划": 1,
        "栈支持动态重规划": 0,
    }


def measure_tot_metrics() -> dict[str, int]:
    """ToT 树搜索 vs 线性的搜索效率指标。"""
    tot = TreeOfThought(branching=3, max_depth=4, prune_threshold=0.3)
    tot.build("问题：如何分配 100 万预算到 3 个项目？")
    _, bfs_visited = tot.bfs_best()

    tot2 = TreeOfThought(branching=3, max_depth=4, prune_threshold=0.3)
    tot2.build("问题：如何分配 100 万预算到 3 个项目？")
    _, dfs_visited = tot2.dfs_best()

    _, linear_visited = linear_thought_search(num_thoughts=80)

    return {
        "ToT 展开节点数": tot.nodes_expanded,
        "ToT 剪枝节点数": tot.nodes_pruned,
        "ToT BFS 访问数": bfs_visited,
        "ToT DFS 访问数": dfs_visited,
        "线性枚举访问数": linear_visited,
        "ToT 剪枝节省比例_pct": int(100 * tot.nodes_pruned / max(tot.nodes_expanded + tot.nodes_pruned, 1)),
    }


# ============================================================================
# 6. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 19 章 · 状态机与工作流编排 × 数据结构协作")
    print("AI 场景：AI 智能体的状态机驱动工作流")
    print("数据结构：图（状态转移）+ 栈（ReAct）+ 队列（Plan-Execute）+ 树（ToT）")
    print("=" * 72)

    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    # ---- 6.1 StateMachine 演示 ----
    print("\n[1] StateMachine：有限状态机（状态转移图 + 状态哈希）")
    print("-" * 72)
    sm = build_agent_state_machine()
    hist = simulate_state_machine(sm)
    print(f"  状态转移历史: {' → '.join(hist)}")
    print(f"  终态: {sm.state['current']}")
    print(f"  状态变量: steps_done={sm.state['steps_done']}, task_complete={sm.state['task_complete']}")
    print(f"  转移图节点: {sorted(set(sm.graph.keys()) | {t.target for ts in sm.graph.values() for t in ts})}")

    # ---- 6.2 ReActLoop 演示 ----
    print("\n[2] ReActLoop：ReAct 循环（栈记录推理链）")
    print("-" * 72)
    react = ReActLoop()
    steps = react.run("计算巴黎人口+100万")
    print(f"  推理栈（{len(steps)} 步）:")
    for s in steps:
        indent = "    " if s.kind == "think" else ("      " if s.kind == "act" else "        ")
        print(f"  {indent}[{s.kind:>7}] #{s.step_id}: {s.content}")

    # ---- 6.3 PlanAndExecute 演示 ----
    print("\n[3] PlanAndExecute：Plan-and-Execute（队列存任务）")
    print("-" * 72)
    pe = PlanAndExecute()
    plan = ["1.理解问题", "2.收集数据", "3.分析数据", "4.生成方案", "5.验证方案"]
    completed = pe.run(plan, replan_at=3, replan_tasks=["3.5.补充数据", "3.6.交叉验证"])
    print(f"  初始计划: {plan}")
    print(f"  执行顺序（FIFO）:")
    for t in completed:
        print(f"    #{t.id} {t.name} → {t.status}")

    # ---- 6.4 TreeOfThought 演示 ----
    print("\n[4] TreeOfThought：Tree-of-Thought（树存思维分支）")
    print("-" * 72)
    tot = TreeOfThought(branching=3, max_depth=3, prune_threshold=0.3)
    root = tot.build("问题：如何分配 100 万预算到 3 个项目？")
    best_bfs, bfs_vis = tot.bfs_best()
    tot2 = TreeOfThought(branching=3, max_depth=3, prune_threshold=0.3)
    tot2.build("问题：如何分配 100 万预算到 3 个项目？")
    best_dfs, dfs_vis = tot2.dfs_best()
    print(f"  思维树展开: {tot.nodes_expanded} 节点, 剪枝 {tot.nodes_pruned} 节点")
    print(f"  BFS 最佳: score={best_bfs.score:.3f}, thought={best_bfs.thought}")
    print(f"  DFS 最佳: score={best_dfs.score:.3f}, thought={best_dfs.thought}")
    print(f"  BFS 访问 {bfs_vis} 节点, DFS 访问 {dfs_vis} 节点")

    # ---- 6.5 性能对比 ----
    print("\n[5] 性能对比")
    print("-" * 72)

    print("\n  (a) 状态机 vs 硬编码 if-else（5 步 agent 工作流）")
    r_sm = bench_state_machine_vs_hardcode()
    print(format_table(r_sm, baseline="硬编码 if-else"))
    m_sm = measure_state_machine_metrics()
    print(f"      状态机状态数: {m_sm['状态机状态数']}, 转移数: {m_sm['状态机转移数']}")
    print(f"      新增状态成本: 状态机 {m_sm['状态机新增状态成本']} 行 vs 硬编码 {m_sm['硬编码新增状态成本']} 行")

    print("\n  (b) ReAct 栈 vs 无栈（推理 + 回溯）")
    r_react = bench_react_stack_vs_nostack()
    print(format_table(r_react, baseline="无栈线性"))
    m_react = measure_react_metrics()
    print(f"      回溯能力: 栈={m_react['栈回溯能力']}, 无栈={m_react['无栈回溯能力']}")
    print(f"      出错恢复: 栈 {m_react['栈出错恢复步数']} 步 vs 无栈 {m_react['无栈出错恢复步数']} 步")

    print("\n  (c) Plan-Execute 队列 vs 栈（5 任务计划）")
    r_plan = bench_plan_queue_vs_stack()
    print(format_table(r_plan, baseline="栈(LIFO)"))
    m_plan = measure_plan_metrics()
    print(f"      执行顺序正确: 队列={m_plan['队列执行顺序正确']}, 栈={m_plan['栈执行顺序正确']}")
    print(f"      动态重规划: 队列={m_plan['队列支持动态重规划']}, 栈={m_plan['栈支持动态重规划']}")

    print("\n  (d) ToT 树搜索 vs 线性枚举（branching=3, depth=4）")
    r_tot = bench_tot_tree_vs_linear()
    print(format_table(r_tot, baseline="线性枚举"))
    m_tot = measure_tot_metrics()
    print(f"      展开节点: {m_tot['ToT 展开节点数']}, 剪枝: {m_tot['ToT 剪枝节点数']} ({m_tot['ToT 剪枝节省比例_pct']}%)")
    print(f"      访问数: BFS={m_tot['ToT BFS 访问数']}, DFS={m_tot['ToT DFS 访问数']}, 线性={m_tot['线性枚举访问数']}")

    # ---- 6.6 保存图 ----
    print("\n[6] 保存性能对比图到 figures/")
    print("-" * 72)

    # 状态机 vs 硬编码 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_sm.items()},
        figures_dir / "fig_state_machine_vs_hardcode.png",
        title="状态机 vs 硬编码 if-else（5 步工作流）",
        ylabel="耗时 (ms)",
        baseline="硬编码 if-else",
    )

    # ReAct 栈 vs 无栈 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_react.items()},
        figures_dir / "fig_react_stack_vs_nostack.png",
        title="ReAct 栈 vs 无栈（推理 + 回溯）",
        ylabel="耗时 (ms)",
        baseline="无栈线性",
    )

    # Plan-Execute 队列 vs 栈 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_plan.items()},
        figures_dir / "fig_plan_queue_vs_stack.png",
        title="Plan-Execute 队列 vs 栈（20 任务）",
        ylabel="耗时 (ms)",
        baseline="栈(LIFO)",
    )

    # ToT 树搜索 vs 线性 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_tot.items()},
        figures_dir / "fig_tot_tree_vs_linear.png",
        title="ToT 树搜索 vs 线性枚举",
        ylabel="耗时 (ms)",
        baseline="线性枚举",
    )

    # ToT 搜索访问节点数对比
    save_bar(
        {"BFS": m_tot["ToT BFS 访问数"], "DFS": m_tot["ToT DFS 访问数"], "线性": m_tot["线性枚举访问数"]},
        figures_dir / "fig_tot_visited_nodes.png",
        title="ToT 搜索访问节点数对比",
        ylabel="访问节点数",
    )

    # 状态机 vs 硬编码 新增状态成本
    save_bar(
        {"状态机": m_sm["状态机新增状态成本"], "硬编码": m_sm["硬编码新增状态成本"]},
        figures_dir / "fig_state_machine_extensibility.png",
        title="新增一个状态的代码修改量",
        ylabel="修改行数",
    )

    # ReAct 出错恢复步数
    save_bar(
        {"栈(回溯)": m_react["栈出错恢复步数"], "无栈(重跑)": m_react["无栈出错恢复步数"]},
        figures_dir / "fig_react_error_recovery.png",
        title="ReAct 出错恢复步数",
        ylabel="步数",
    )

    print(f"  图已保存到：{figures_dir}")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"    - {p.name}")

    # ---- 6.7 协作总结 ----
    print("\n[7] 4 种数据结构协作总结")
    print("-" * 72)
    print("  用户提问到达")
    print("    │")
    print("    ▼")
    print("  [图/状态转移图]  StateMachine：状态机驱动状态转移")
    print("    │                idle→planning→acting→observing→done")
    print("    │                对比硬编码：新增状态成本 1 行 vs 3 行")
    print("    ▼")
    print("  [栈]             ReActLoop：think→act→observe 推理链")
    print("    │                出错可 backtrack() 回溯，无需重跑")
    print("    │                对比无栈：恢复步数 2 vs 12")
    print("    ▼")
    print("  [队列]           PlanAndExecute：planner 生成任务队列，executor FIFO 执行")
    print("    │                保持依赖顺序，支持动态重规划（replan 入队）")
    print("    │                对比栈：LIFO 反序执行，破坏依赖")
    print("    ▼")
    print("  [树]             TreeOfThought：多分支思维搜索 + 剪枝")
    print("    │                BFS/DFS 搜索最优思维，低分分支剪枝")
    print("    │                对比线性：剪枝节省 ~30% 探索")
    print("    ▼")
    print("  [哈希表]         状态存储：state dict 存所有状态变量")
    print("    │                LangGraph 的 state dict，所有节点共享读写")
    print("    ▼")
    print("  输出答案 → 返回用户")

    print("\n" + "=" * 72)
    print("done.")


if __name__ == "__main__":
    main()