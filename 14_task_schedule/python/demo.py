"""第 14 章 demo：任务调度与批处理 × 数据结构协作（堆 + DAG + 队列 + 一致性哈希）

扩展专题：一个 AI 应用场景（LLM 推理服务调度）→ 多种数据结构协作
- PriorityHeap       ：堆，请求优先级调度（VIP/SLA 优先）
- DependencyDAG      ：DAG + 拓扑排序，任务依赖（推理依赖前处理）
- DynamicBatcher     ：队列，动态批处理（凑批或超时发批，类 vLLM continuous batching）
- ConsistentHashing  ：一致性哈希，多节点负载均衡（虚拟节点 + 环形空间）

模拟完整推理服务：
  请求进来 → 优先级排序（堆）→ 依赖检查（DAG）→ 批处理（队列）→ 负载均衡分发（一致性哈希）

性能对比：
1. 优先级调度 vs FIFO（VIP 请求延迟）
2. 动态批处理 vs 单请求（吞吐量）
3. 一致性哈希 vs 取模哈希（节点增减时数据迁移量）

跑法：
    python 14_task_schedule/python/demo.py
"""

from __future__ import annotations

import bisect
import hashlib
import heapq
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. PriorityHeap：堆，请求优先级调度
# ============================================================================


@dataclass(order=True)
class Request:
    """LLM 推理请求。priority 小的先调度（最小堆）。

    priority: 1=VIP/SLA 高优, 2=普通, 3=低优/后台
    seq: 同优先级内 FIFO 的插入序号（避免同优先级乱序）
    """

    priority: int
    seq: int
    user: str = field(compare=False)
    prompt: str = field(compare=False)
    tokens: int = field(compare=False, default=16)


class PriorityHeap:
    """请求优先级调度，用最小堆。

    - push(req): O(log n)
    - pop(): 取出优先级最高（priority 值最小）的请求 O(log n)
    - peek(): 看堆顶 O(1)

    对比 FIFO：FIFO 不区分优先级，VIP 请求可能排在大量普通请求后面，
    违反 SLA（服务等级协议）。
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []
        self._seq = 0

    def push(self, user: str, prompt: str, priority: int, tokens: int = 16) -> Request:
        req = Request(priority=priority, seq=self._seq, user=user,
                      prompt=prompt, tokens=tokens)
        self._seq += 1
        heapq.heappush(self._heap, req)
        return req

    def pop(self) -> Request | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Request | None:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)


class FIFOQueue:
    """FIFO 调度，对比基准。不区分优先级，先入先出。"""

    def __init__(self) -> None:
        self._queue: deque[Request] = deque()
        self._seq = 0

    def push(self, user: str, prompt: str, priority: int, tokens: int = 16) -> Request:
        req = Request(priority=priority, seq=self._seq, user=user,
                      prompt=prompt, tokens=tokens)
        self._seq += 1
        self._queue.append(req)
        return req

    def pop(self) -> Request | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)


# ============================================================================
# 2. DependencyDAG：任务依赖图 + 拓扑排序
# ============================================================================


class DependencyDAG:
    """任务依赖图，邻接表 + Kahn 拓扑排序。

    LLM 推理流水线常有依赖：
      tokenize → embed → [prefill] → [decode] → detokenize → postprocess
    用 DAG 表示依赖，拓扑排序给出合法执行顺序，拓扑层级给出可并行分组。

    add_edge(u, v) 表示 u 必须先于 v 执行。
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
        """Kahn 算法拓扑排序 O(V+E)。"""
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
            raise ValueError(f"依赖图有环，涉及节点: {remaining}")
        return order

    def topological_levels(self) -> list[list[str]]:
        """按拓扑层级分组，同层任务无依赖可并行。"""
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
# 3. DynamicBatcher：队列，动态批处理
# ============================================================================


@dataclass
class Batch:
    """一个推理批次。"""

    batch_id: int
    requests: list[Request]
    formed_at: int  # 形成时刻（全局时钟）

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.requests)


class DynamicBatcher:
    """动态批处理，用队列收集请求凑批。

    策略（类 vLLM continuous batching）：
    - 请求到达先入队列
    - 满足以下任一条件就发批：
      (a) 队列里请求数 >= batch_size（凑够一批）
      (b) 距上一个批的等待时间 >= max_wait（超时发批，避免低流量时请求饿死）
    - 批内请求一起做 prefill，共享 KV cache，吞吐量远高于逐请求处理

    对比单请求处理：每个请求独占一次 prefill，GPU 利用率低、吞吐量差。
    """

    def __init__(self, batch_size: int = 8, max_wait: int = 5) -> None:
        self._queue: deque[Request] = deque()
        self._batch_size = batch_size
        self._max_wait = max_wait
        self._batches: list[Batch] = []
        self._batch_id = 0
        self._last_flush_clock: int | None = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def pending(self) -> int:
        return len(self._queue)

    @property
    def batches(self) -> list[Batch]:
        return self._batches

    def add(self, req: Request) -> None:
        self._queue.append(req)

    def try_flush(self, clock: int) -> Batch | None:
        """尝试发批。返回形成的 Batch 或 None。

        触发条件：
        - 队列长度 >= batch_size → 凑够发批
        - 队列非空 且 距上次发批（或首个请求入队）已超 max_wait → 超时发批
        """
        if not self._queue:
            return None
        size_ok = len(self._queue) >= self._batch_size
        wait_ok = self._last_flush_clock is not None and \
            (clock - self._last_flush_clock) >= self._max_wait
        if not (size_ok or wait_ok):
            return None
        batch_reqs: list[Request] = []
        for _ in range(min(self._batch_size, len(self._queue))):
            batch_reqs.append(self._queue.popleft())
        batch = Batch(batch_id=self._batch_id, requests=batch_reqs, formed_at=clock)
        self._batch_id += 1
        self._batches.append(batch)
        self._last_flush_clock = clock
        return batch

    def flush_all(self, clock: int) -> list[Batch]:
        """强制把队列里所有请求发批（收尾用）。"""
        result: list[Batch] = []
        while self._queue:
            batch = self.try_flush(clock)
            if batch is None:
                # 强制发批：忽略触发条件
                batch_reqs: list[Request] = []
                for _ in range(min(self._batch_size, len(self._queue))):
                    batch_reqs.append(self._queue.popleft())
                batch = Batch(batch_id=self._batch_id, requests=batch_reqs,
                              formed_at=clock)
                self._batch_id += 1
                self._batches.append(batch)
                self._last_flush_clock = clock
            result.append(batch)
        return result


# ============================================================================
# 4. ConsistentHashing：一致性哈希，多节点负载均衡
# ============================================================================


class ConsistentHashing:
    """一致性哈希，环形空间 + 虚拟节点。

    解决的问题：多节点负载均衡时，节点增减要尽量少迁移数据。

    - 环形空间：把 hash 值域 [0, 2^64) 看成环，节点和 key 都映射到环上
    - key 归属：沿环顺时针找到的第一个节点
    - 虚拟节点：每个物理节点对应环上多个虚拟节点（默认 150 个），
      让节点在环上分布均匀，避免数据倾斜

    对比取模哈希 hash(key) % N：
    - 取模哈希：N 个节点变 N±1，几乎所有 key 都要重映射 → 迁移量 O(k)（k=key 数）
    - 一致性哈希：只有环上增减点附近的一小段 key 需要迁移 → 迁移量 O(k/n)
    """

    def __init__(self, virtual_nodes: int = 150) -> None:
        self._virtual_nodes = virtual_nodes
        self._ring: list[int] = []  # 排序后的虚拟节点 hash
        self._ring_to_node: dict[int, str] = {}
        self._nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        """把字符串映射到 [0, 2^64) 环上。用 md5 取前 8 字节。"""
        h = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    def add_node(self, node: str) -> None:
        """新增节点：在环上放 virtual_nodes 个虚拟节点。"""
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self._virtual_nodes):
            h = self._hash(f"{node}#{i}")
            self._ring_to_node[h] = node
            bisect.insort(self._ring, h)

    def remove_node(self, node: str) -> None:
        """移除节点：删环上该节点的所有虚拟节点。"""
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        for i in range(self._virtual_nodes):
            h = self._hash(f"{node}#{i}")
            if h in self._ring_to_node:
                del self._ring_to_node[h]
                pos = bisect.bisect_left(self._ring, h)
                if pos < len(self._ring) and self._ring[pos] == h:
                    self._ring.pop(pos)

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    def get_node(self, key: str) -> str | None:
        """key 归属：沿环顺时针找第一个虚拟节点，返回其物理节点。"""
        if not self._ring:
            return None
        h = self._hash(key)
        pos = bisect.bisect(self._ring, h)
        if pos == len(self._ring):
            pos = 0  # 环回绕
        return self._ring_to_node[self._ring[pos]]

    def key_distribution(self, keys: list[str]) -> dict[str, int]:
        """统计 key 在各节点上的分布（用于评估均衡度）。"""
        dist: dict[str, int] = {n: 0 for n in self._nodes}
        for k in keys:
            n = self.get_node(k)
            if n is not None:
                dist[n] += 1
        return dist


class ModHashing:
    """取模哈希，对比基准。hash(key) % N。

    节点增减时，几乎所有 key 都要重映射 → 迁移量 O(k)。
    """

    def __init__(self) -> None:
        self._nodes: list[str] = []

    def add_node(self, node: str) -> None:
        if node not in self._nodes:
            self._nodes.append(node)

    def remove_node(self, node: str) -> None:
        if node in self._nodes:
            self._nodes.remove(node)

    @property
    def nodes(self) -> set[str]:
        return set(self._nodes)

    def get_node(self, key: str) -> str | None:
        if not self._nodes:
            return None
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return self._nodes[h % len(self._nodes)]


def count_migration(
    old_map: dict[str, str | None], new_map: dict[str, str | None],
) -> int:
    """统计节点变化前后，key 归属改变的个数（迁移量）。"""
    return sum(1 for k in old_map if old_map[k] != new_map.get(k))


# ============================================================================
# 5. 完整推理服务模拟：4 种数据结构协作
# ============================================================================


def build_inference_pipeline() -> DependencyDAG:
    """构建 LLM 推理流水线 DAG。

    典型流水线：
      tokenize → embed → prefill → decode → detokenize → postprocess
    其中 prefill 和 decode 是核心，其余是前后处理。
    """
    dag = DependencyDAG()
    dag.add_edge("tokenize", "embed")
    dag.add_edge("embed", "prefill")
    dag.add_edge("prefill", "decode")
    dag.add_edge("decode", "detokenize")
    dag.add_edge("detokenize", "postprocess")
    return dag


def simulate_inference_service(
    n_requests: int = 100,
    batch_size: int = 8,
    max_wait: int = 5,
    n_nodes: int = 4,
    seed: int = 42,
) -> dict:
    """模拟完整 LLM 推理服务，4 种数据结构协作。

    流程：
    1. 请求按时间到达，入优先级堆（VIP 优先）
    2. 每个请求要过依赖 DAG（前处理 → 推理 → 后处理）
    3. 请求进入动态批处理器（队列凑批）
    4. 批形成后用一致性哈希分发到节点

    Returns:
        包含统计信息的字典
    """
    rng = random.Random(seed)
    heap = PriorityHeap()
    batcher = DynamicBatcher(batch_size=batch_size, max_wait=max_wait)
    ch = ConsistentHashing(virtual_nodes=150)
    for i in range(n_nodes):
        ch.add_node(f"node_{i}")

    # 生成请求：10% VIP, 70% 普通, 20% 低优
    users: list[tuple[str, int]] = []
    for i in range(n_requests):
        r = rng.random()
        if r < 0.10:
            users.append((f"vip_{i}", 1))
        elif r < 0.80:
            users.append((f"user_{i}", 2))
        else:
            users.append((f"bg_{i}", 3))

    # 模拟时间推进：请求按到达时间入堆，每轮尝试发批
    vip_done_at: dict[str, int] = {}
    all_done_at: dict[str, int] = {}
    clock = 0
    arrival_idx = 0
    # 每个时钟步可能有 0-3 个新请求到达
    arrivals_per_step: list[int] = [rng.randint(0, 3) for _ in range(n_requests * 2)]

    for step in range(len(arrivals_per_step)):
        clock = step
        # 新请求到达 → 入堆
        for _ in range(arrivals_per_step[step]):
            if arrival_idx >= n_requests:
                break
            user, pri = users[arrival_idx]
            heap.push(user, f"prompt_{arrival_idx}", priority=pri,
                      tokens=rng.randint(8, 64))
            arrival_idx += 1
        # 从堆里取已就绪请求进批处理器（堆决定优先级）
        while len(heap) > 0 and batcher.pending < batch_size * 2:
            req = heap.pop()
            assert req is not None
            batcher.add(req)
        # 尝试发批（队列决定凑批时机）
        batch = batcher.try_flush(clock)
        if batch is not None:
            for r in batch.requests:
                all_done_at[r.user] = clock
                if r.priority == 1:
                    vip_done_at[r.user] = clock

    # 收尾：把剩余请求都发批
    while len(heap) > 0:
        req = heap.pop()
        assert req is not None
        batcher.add(req)
    final_batches = batcher.flush_all(clock + 1)
    for batch in final_batches:
        for r in batch.requests:
            all_done_at[r.user] = batch.formed_at
            if r.priority == 1:
                vip_done_at[r.user] = batch.formed_at

    # 统计
    vip_latencies = [vip_done_at[u] for u, _ in users if u in vip_done_at]
    all_latencies = [all_done_at[u] for u, _ in users if u in all_done_at]
    batch_sizes = [b.size for b in batcher.batches]
    # 节点分布（用批 id 当 key 分发）
    batch_keys = [f"batch_{b.batch_id}" for b in batcher.batches]
    node_dist = ch.key_distribution(batch_keys)

    return {
        "n_requests": n_requests,
        "n_nodes": n_nodes,
        "n_batches": len(batcher.batches),
        "avg_batch_size": (sum(batch_sizes) / len(batch_sizes)) if batch_sizes else 0.0,
        "vip_avg_latency": (sum(vip_latencies) / len(vip_latencies)) if vip_latencies else 0.0,
        "all_avg_latency": (sum(all_latencies) / len(all_latencies)) if all_latencies else 0.0,
        "node_dist": node_dist,
        "batch_sizes": batch_sizes,
    }


# ============================================================================
# 6. 性能对比
# ============================================================================


def benchmark_priority_vs_fifo(n_requests: int = 200, seed: int = 42) -> dict:
    """对比：优先级调度 vs FIFO（VIP 请求延迟）。

    场景：200 个请求，10% VIP，普通请求先到、VIP 后到。
    - 优先级调度：VIP 插队，延迟低
    - FIFO：VIP 排在普通请求后面，延迟高
    """
    rng = random.Random(seed)
    # 生成请求：普通先到，VIP 后到
    push_order: list[tuple[str, int]] = []
    n_vip = n_requests // 10
    n_normal = n_requests - n_vip
    for i in range(n_normal):
        push_order.append((f"user_{i}", 2))
    for i in range(n_vip):
        push_order.append((f"vip_{i}", 1))

    # 优先级调度
    ph = PriorityHeap()
    for user, pri in push_order:
        ph.push(user, "prompt", priority=pri)
    order_p: list[str] = []
    while len(ph) > 0:
        r = ph.pop()
        assert r is not None
        order_p.append(r.user)

    # FIFO 调度
    fq = FIFOQueue()
    for user, pri in push_order:
        fq.push(user, "prompt", priority=pri)
    order_f: list[str] = []
    while len(fq) > 0:
        r = fq.pop()
        assert r is not None
        order_f.append(r.user)

    # VIP 请求的执行位置（位置越大延迟越多）
    vip_pos_p = [i for i, u in enumerate(order_p) if u.startswith("vip_")]
    vip_pos_f = [i for i, u in enumerate(order_f) if u.startswith("vip_")]

    return {
        "n_requests": n_requests,
        "n_vip": n_vip,
        "vip_pos_priority": vip_pos_p,
        "vip_pos_fifo": vip_pos_f,
        "vip_avg_pos_priority": (sum(vip_pos_p) / len(vip_pos_p)) if vip_pos_p else 0.0,
        "vip_avg_pos_fifo": (sum(vip_pos_f) / len(vip_pos_f)) if vip_pos_f else 0.0,
    }


def benchmark_batch_vs_single(
    n_requests: int = 200, batch_size: int = 8, seed: int = 42,
) -> dict:
    """对比：动态批处理 vs 单请求处理（吞吐量）。

    假设：
    - 单请求：每次 prefill 耗时 10ms（GPU 利用率低）
    - 批处理：批内 prefill 共享，批大小 b 时耗时 10 + 2*(b-1) ms（amortized）
    - 吞吐量 = 请求数 / 总耗时
    """
    rng = random.Random(seed)
    batcher = DynamicBatcher(batch_size=batch_size, max_wait=5)
    # 请求按时间到达
    clock = 0
    arrived = 0
    while arrived < n_requests:
        n_new = rng.randint(1, 4)
        for _ in range(n_new):
            if arrived >= n_requests:
                break
            req = Request(priority=2, seq=arrived, user=f"u_{arrived}",
                          prompt="p", tokens=rng.randint(8, 64))
            batcher.add(req)
            arrived += 1
        batcher.try_flush(clock)
        clock += 1
    batcher.flush_all(clock)

    # 批处理总耗时
    batch_time = 0.0
    for b in batcher.batches:
        batch_time += 10 + 2 * (b.size - 1)
    # 单请求总耗时
    single_time = 10.0 * n_requests

    return {
        "n_requests": n_requests,
        "n_batches": len(batcher.batches),
        "batch_size": batch_size,
        "avg_batch_size": (sum(b.size for b in batcher.batches) / len(batcher.batches))
            if batcher.batches else 0.0,
        "batch_total_ms": batch_time,
        "single_total_ms": single_time,
        "batch_throughput": n_requests / batch_time if batch_time > 0 else 0.0,
        "single_throughput": n_requests / single_time if single_time > 0 else 0.0,
    }


def benchmark_consistent_vs_mod(
    n_keys: int = 10000, n_initial: int = 4, virtual_nodes: int = 150,
) -> dict:
    """对比：一致性哈希 vs 取模哈希（节点增减时数据迁移量）。

    场景：10000 个 key 初始分布到 4 个节点，然后加 1 个节点、再删 1 个节点，
    统计每次节点变化后有多少 key 要迁移。

    理论：
    - 取模哈希：N→N+1 或 N→N-1，几乎所有 key 都要重映射 → 迁移量 ≈ k
    - 一致性哈希：只有环上增减点附近的一小段 key 迁移 → 迁移量 ≈ k/n
    """
    rng = random.Random(0)
    keys = [f"key_{i}" for i in range(n_keys)]

    # ---- 一致性哈希 ----
    ch = ConsistentHashing(virtual_nodes=virtual_nodes)
    for i in range(n_initial):
        ch.add_node(f"node_{i}")
    ch_old = {k: ch.get_node(k) for k in keys}
    ch_dist_old = ch.key_distribution(keys)

    # 加一个节点
    ch.add_node(f"node_{n_initial}")
    ch_new = {k: ch.get_node(k) for k in keys}
    ch_add_migrate = count_migration(ch_old, ch_new)
    ch_dist_after_add = ch.key_distribution(keys)

    # 删一个节点
    ch.remove_node("node_0")
    ch_new2 = {k: ch.get_node(k) for k in keys}
    ch_remove_migrate = count_migration(ch_new, ch_new2)
    ch_dist_after_remove = ch.key_distribution(keys)

    # ---- 取模哈希 ----
    mh = ModHashing()
    for i in range(n_initial):
        mh.add_node(f"node_{i}")
    mh_old = {k: mh.get_node(k) for k in keys}

    # 加一个节点
    mh.add_node(f"node_{n_initial}")
    mh_new = {k: mh.get_node(k) for k in keys}
    mh_add_migrate = count_migration(mh_old, mh_new)

    # 删一个节点
    mh.remove_node("node_0")
    mh_new2 = {k: mh.get_node(k) for k in keys}
    mh_remove_migrate = count_migration(mh_new, mh_new2)

    return {
        "n_keys": n_keys,
        "n_initial": n_initial,
        "virtual_nodes": virtual_nodes,
        "ch_add_migrate": ch_add_migrate,
        "ch_remove_migrate": ch_remove_migrate,
        "mh_add_migrate": mh_add_migrate,
        "mh_remove_migrate": mh_remove_migrate,
        "ch_dist_old": ch_dist_old,
        "ch_dist_after_add": ch_dist_after_add,
        "mh_add_ratio": mh_add_migrate / n_keys,
        "ch_add_ratio": ch_add_migrate / n_keys,
    }


# ============================================================================
# 7. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 14 章 demo：任务调度与批处理 × 数据结构协作")
    print("（堆 + DAG + 队列 + 一致性哈希）")
    print("=" * 72)

    print("\n[1] 优先级调度：堆（VIP/SLA 优先）")
    print("-" * 72)
    ph = PriorityHeap()
    # 模拟请求到达：普通先到，VIP 后到
    for i in range(10):
        ph.push(f"user_{i}", f"prompt_{i}", priority=2)
    ph.push("vip_0", "important_prompt", priority=1)
    ph.push("vip_1", "urgent_prompt", priority=1)
    print(f"  12 个请求入堆：10 个普通（priority=2）+ 2 个 VIP（priority=1，后到）")
    print(f"  堆大小 = {len(ph)}")
    print(f"  堆顶 = {ph.peek().user if ph.peek() else None}（VIP 先出）")
    print(f"  出队顺序（前 5 个）：")
    for _ in range(5):
        r = ph.pop()
        assert r is not None
        label = "VIP" if r.priority == 1 else "普通"
        print(f"    {label} {r.user}（priority={r.priority}）")
    print(f"  → VIP 后到但先出：堆让高优先级请求插队")

    print("\n[2] 任务依赖：DAG + 拓扑排序（推理流水线）")
    print("-" * 72)
    dag = build_inference_pipeline()
    print(f"  LLM 推理流水线 DAG：")
    print(f"    tokenize → embed → prefill → decode → detokenize → postprocess")
    print(f"  节点数 = {len(dag.nodes)}，边数 = {len(dag.edges)}")
    print(f"  有环？{dag.has_cycle()}")
    order = dag.topological_sort()
    print(f"  拓扑排序 = {order}")
    levels = dag.topological_levels()
    print(f"  拓扑层级（共 {len(levels)} 层，同层可并行）：")
    for i, lvl in enumerate(levels):
        print(f"    层 {i}: {lvl}")
    print(f"  → 流水线深度 = {len(levels)} 步（最少串行步数）")

    print("\n[3] 动态批处理：队列凑批（类 vLLM continuous batching）")
    print("-" * 72)
    batcher = DynamicBatcher(batch_size=4, max_wait=5)
    print(f"  配置：batch_size=4, max_wait=5")
    # 模拟请求到达
    for i in range(10):
        req = Request(priority=2, seq=i, user=f"u_{i}", prompt="p", tokens=16)
        batcher.add(req)
    print(f"  10 个请求入队，pending = {batcher.pending}")
    # 凑批
    clock = 0
    while batcher.pending > 0:
        batch = batcher.try_flush(clock)
        if batch is not None:
            print(f"    时刻 {clock}: 发批 #{batch.batch_id}，"
                  f"大小 {batch.size}，总 token {batch.total_tokens}，"
                  f"用户 {[r.user for r in batch.requests]}")
        clock += 1
        if clock > 20:
            break
    print(f"  → 队列凑够 batch_size 就发批，凑不够等 max_wait 超时发批")

    print("\n[4] 一致性哈希：多节点负载均衡（虚拟节点 + 环形空间）")
    print("-" * 72)
    ch = ConsistentHashing(virtual_nodes=150)
    for i in range(4):
        ch.add_node(f"node_{i}")
    print(f"  4 个节点，每节点 150 个虚拟节点 → 环上 {4 * 150} 个虚拟节点")
    # 分发 1000 个 key
    keys = [f"req_{i}" for i in range(1000)]
    dist = ch.key_distribution(keys)
    print(f"  1000 个 key 的分布：")
    for node in sorted(dist):
        bar = "#" * (dist[node] // 10)
        print(f"    {node}: {dist[node]:>4} {bar}")
    total = sum(dist.values())
    avg = total / len(dist)
    max_dev = max(abs(v - avg) for v in dist.values())
    print(f"  均值 = {avg:.0f}，最大偏差 = {max_dev}（虚拟节点保证均匀）")

    print("\n[5] 完整推理服务模拟：4 种数据结构协作")
    print("-" * 72)
    result = simulate_inference_service(
        n_requests=100, batch_size=8, max_wait=5, n_nodes=4, seed=42,
    )
    print(f"  配置：100 请求，batch_size=8，max_wait=5，4 节点")
    print(f"  结果：")
    print(f"    总批数 = {result['n_batches']}")
    print(f"    平均批大小 = {result['avg_batch_size']:.2f}")
    print(f"    VIP 请求平均延迟 = {result['vip_avg_latency']:.2f} 步")
    print(f"    所有请求平均延迟 = {result['all_avg_latency']:.2f} 步")
    print(f"    批分发到节点的分布：")
    for node in sorted(result["node_dist"]):
        print(f"      {node}: {result['node_dist'][node]} 批")
    print(f"  → 堆排序 → DAG 检查依赖 → 队列凑批 → 一致性哈希分发")

    print("\n[6] 性能对比：优先级调度 vs FIFO（VIP 请求延迟）")
    print("-" * 72)
    res_sched = benchmark_priority_vs_fifo(n_requests=200, seed=42)
    print(f"  场景：{res_sched['n_requests']} 请求，{res_sched['n_vip']} VIP 后到")
    print(f"  优先级调度：VIP 平均执行位置 = {res_sched['vip_avg_pos_priority']:.1f}")
    print(f"  FIFO 调度  ：VIP 平均执行位置 = {res_sched['vip_avg_pos_fifo']:.1f}")
    if res_sched["vip_avg_pos_fifo"] > res_sched["vip_avg_pos_priority"]:
        ratio = res_sched["vip_avg_pos_fifo"] / max(res_sched["vip_avg_pos_priority"], 1)
        print(f"  → FIFO 让 VIP 多等 {ratio:.1f}x，优先级调度保障 SLA")

    print("\n[7] 性能对比：动态批处理 vs 单请求（吞吐量）")
    print("-" * 72)
    res_batch = benchmark_batch_vs_single(n_requests=200, batch_size=8, seed=42)
    print(f"  场景：{res_batch['n_requests']} 请求，batch_size={res_batch['batch_size']}")
    print(f"  动态批处理：{res_batch['n_batches']} 批，平均批大小 {res_batch['avg_batch_size']:.2f}")
    print(f"  动态批处理总耗时 = {res_batch['batch_total_ms']:.0f} ms，"
          f"吞吐量 = {res_batch['batch_throughput']:.2f} req/ms")
    print(f"  单请求处理总耗时 = {res_batch['single_total_ms']:.0f} ms，"
          f"吞吐量 = {res_batch['single_throughput']:.4f} req/ms")
    speedup = res_batch["batch_throughput"] / res_batch["single_throughput"]
    print(f"  → 批处理吞吐量是单请求的 {speedup:.2f}x（共享 prefill/KV cache）")

    print("\n[8] 性能对比：一致性哈希 vs 取模哈希（节点增减迁移量）")
    print("-" * 72)
    res_hash = benchmark_consistent_vs_mod(n_keys=10000, n_initial=4, virtual_nodes=150)
    print(f"  场景：{res_hash['n_keys']} key，初始 {res_hash['n_initial']} 节点，"
          f"虚拟节点 {res_hash['virtual_nodes']}")
    print(f"  加 1 节点：")
    print(f"    一致性哈希迁移 = {res_hash['ch_add_migrate']} key "
          f"（{res_hash['ch_add_ratio']*100:.2f}%）")
    print(f"    取模哈希迁移   = {res_hash['mh_add_migrate']} key "
          f"（{res_hash['mh_add_ratio']*100:.2f}%）")
    print(f"  删 1 节点：")
    print(f"    一致性哈希迁移 = {res_hash['ch_remove_migrate']} key")
    print(f"    取模哈希迁移   = {res_hash['mh_remove_migrate']} key")
    if res_hash["mh_add_migrate"] > 0:
        ratio_add = res_hash["mh_add_migrate"] / res_hash["ch_add_migrate"]
        print(f"  → 加节点时取模哈希迁移量是一致性哈希的 {ratio_add:.1f}x")
    print(f"  → 一致性哈希：O(k/n) 迁移；取模哈希：O(k) 迁移")

    print("\n[9] 保存对比图到 figures/")
    print("-" * 72)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    # 图 1：VIP 延迟对比
    save_bar(
        {
            "优先级调度": res_sched["vip_avg_pos_priority"],
            "FIFO 调度": res_sched["vip_avg_pos_fifo"],
        },
        fig_dir / "vip_delay_priority_vs_fifo.png",
        title=f"VIP 请求平均执行位置（{res_sched['n_requests']} 请求）",
        ylabel="平均执行位置",
        baseline="优先级调度",
    )
    print(f"  ✓ 保存 figures/vip_delay_priority_vs_fifo.png")

    # 图 2：吞吐量对比
    save_bar(
        {
            "动态批处理": res_batch["batch_throughput"],
            "单请求处理": res_batch["single_throughput"],
        },
        fig_dir / "throughput_batch_vs_single.png",
        title=f"吞吐量对比（{res_batch['n_requests']} 请求，batch_size={res_batch['batch_size']}）",
        ylabel="吞吐量 (req/ms)",
        baseline="单请求处理",
    )
    print(f"  ✓ 保存 figures/throughput_batch_vs_single.png")

    # 图 3：迁移量对比
    save_bar(
        {
            "一致性哈希-加节点": float(res_hash["ch_add_migrate"]),
            "取模哈希-加节点": float(res_hash["mh_add_migrate"]),
            "一致性哈希-删节点": float(res_hash["ch_remove_migrate"]),
            "取模哈希-删节点": float(res_hash["mh_remove_migrate"]),
        },
        fig_dir / "migration_consistent_vs_mod.png",
        title=f"节点增减时 key 迁移量（{res_hash['n_keys']} key）",
        ylabel="迁移 key 数",
    )
    print(f"  ✓ 保存 figures/migration_consistent_vs_mod.png")

    # 图 4：节点分布均衡度
    save_bar(
        {k: float(v) for k, v in res_hash["ch_dist_old"].items()},
        fig_dir / "consistent_hash_distribution.png",
        title=f"一致性哈希 key 分布（{res_hash['n_keys']} key，4 节点，150 虚拟节点）",
        ylabel="key 数",
        xlabel="节点",
    )
    print(f"  ✓ 保存 figures/consistent_hash_distribution.png")

    # 图 5：批大小分布
    batch_size_dist: dict[str, float] = {}
    for b in result["batch_sizes"]:
        key = f"size_{b}"
        batch_size_dist[key] = batch_size_dist.get(key, 0.0) + 1.0
    if batch_size_dist:
        save_bar(
            batch_size_dist,
            fig_dir / "batch_size_distribution.png",
            title=f"批大小分布（{result['n_batches']} 批）",
            ylabel="批数",
            xlabel="批大小",
        )
        print(f"  ✓ 保存 figures/batch_size_distribution.png")

    print("\n" + "=" * 72)
    print("结论：LLM 推理服务调度是 4 种数据结构的协奏——")
    print("  堆         → 优先级调度（VIP/SLA 保障）")
    print("  DAG        → 任务依赖（推理流水线拓扑排序）")
    print("  队列       → 动态批处理（凑批提高吞吐量）")
    print("  一致性哈希 → 负载均衡（节点增减最少迁移）")
    print("生产级系统 vLLM 用 continuous batching + PagedAttention，")
    print("Triton Inference Server 用动态批处理 + 多模型调度，")
    print("本质都是这 4 种数据结构的不同组合。")
    print("=" * 72)


if __name__ == "__main__":
    main()