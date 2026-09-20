"""第 18 章 demo：模型部署与推理服务 × 数据结构协作（图 + 堆 + 一致性哈希 + 页表）

扩展专题：一个 AI 应用场景（大模型部署推理服务）
→ 多种数据结构协作
- OperatorFusion  : 算子融合（计算图上合并相邻算子，减少 kernel launch）
- RequestScheduler: 请求调度（堆/优先队列，按 SLA 调度推理请求）
- LoadBalancer    : 一致性哈希负载均衡（多 GPU/多节点分发请求）
- MemoryPaged     : 页表显存管理（PagedAttention 的分块分配，统计利用率）

模拟完整推理服务：
  请求进来 → [堆] 优先级调度 → [一致性哈希] 负载均衡分发到 GPU
           → [图] 算子融合优化计算图 → [页表] 显存分块管理 KV cache
           → 输出 token

性能对比：
1. 算子融合前后 kernel 数量（减少 kernel launch 开销）
2. 优先级调度 vs FIFO（SLA 违约率）
3. 一致性哈希 vs 取模哈希（节点变动时 key 迁移量）
4. 分页显存 vs 连续分配（显存利用率 & 碎片率）

跑法：
    python 18_model_deploy/python/demo.py
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. OperatorFusion：算子融合（计算图上合并相邻算子）
# ============================================================================


@dataclass
class Operator:
    """计算图中的一个算子节点。

    op_type : 算子类型（MatMul / Add / ReLU / Softmax / LayerNorm / Scale ...）
    name    : 算子名（唯一标识）
    inputs  : 输入算子名列表（前驱，构成 DAG）
    fused   : 是否已被融合（融合后不再单独 launch kernel）
    """

    op_type: str
    name: str
    inputs: list[str] = field(default_factory=list)
    fused: bool = False

    @property
    def is_elementwise(self) -> bool:
        """是否为 element-wise 算子（可融合的候选）。

        element-wise 算子对每个元素独立计算，天然可融合：
        Add / ReLU / Scale / Tanh / GELU 等。
        """
        return self.op_type in {"Add", "ReLU", "Scale", "Tanh", "GELU", "Sigmoid"}


class ComputeGraph:
    """计算图（DAG）：算子为节点，张量流动为边。

    AI 场景：大模型推理时，计算图由几十到几百个算子组成。
    每个算子单独 launch 一个 GPU kernel，开销主要来自：
    - kernel launch 本身（~5-10 μs / 次）
    - 中间张量的显存读写（HBM 带宽瓶颈）

    算子融合（operator fusion）把相邻的 element-wise 算子合并成一个 kernel：
    - 减少 kernel launch 次数
    - 中间结果留在寄存器/共享内存，不写回 HBM

    本 demo 用邻接表存 DAG，支持：
    - add_op(op)            : 添加算子
    - fuse_elementwise()    : 贪心融合相邻 element-wise 算子
    - kernel_count()        : 融合后需要 launch 的 kernel 数
    """

    def __init__(self) -> None:
        self._ops: dict[str, Operator] = {}

    def add_op(self, op: Operator) -> None:
        self._ops[op.name] = op

    def op(self, name: str) -> Operator:
        return self._ops[name]

    def topo_order(self) -> list[str]:
        """拓扑序（Kahn 算法）。inputs 中引用的不存在算子视为外部输入，忽略。"""
        in_deg = {n: 0 for n in self._ops}
        adj: dict[str, list[str]] = {n: [] for n in self._ops}
        for name, op in self._ops.items():
            for inp in op.inputs:
                if inp in self._ops:  # 只统计图内前驱
                    adj[inp].append(name)
                    in_deg[name] += 1
        q = deque([n for n, d in in_deg.items() if d == 0])
        order: list[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for m in adj[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    q.append(m)
        return order

    def fuse_elementwise(self) -> int:
        """贪心融合相邻 element-wise 算子，返回融合次数。

        策略：按拓扑序扫描，若当前算子是 element-wise 且：
        - 只有一个图内前驱
        - 前驱只有一个消费者（即当前算子）
        则把当前算子融合进前驱，标记 fused=True，不再单独 launch kernel。

        这模拟「producer-consumer 融合」：element-wise 算子直接接在前一个 kernel
        后面，中间结果不写回 HBM。生产级融合器（XLA/MLIR/TensorRT）会做更复杂的
        融合（reduction 融合、tiling、多 producer 融合等）。
        """
        # 先统计每个算子的消费者数
        consumer_count: dict[str, int] = {n: 0 for n in self._ops}
        for op in self._ops.values():
            for inp in op.inputs:
                if inp in consumer_count:
                    consumer_count[inp] += 1

        order = self.topo_order()
        fused_count = 0
        for name in order:
            op = self._ops[name]
            if not op.is_elementwise or op.fused:
                continue
            # 只有一个图内前驱
            in_preds = [p for p in op.inputs if p in self._ops]
            if len(in_preds) != 1:
                continue
            pred = self._ops[in_preds[0]]
            # 前驱未被融合且只有一个消费者
            if pred.fused or consumer_count[in_preds[0]] != 1:
                continue
            op.fused = True
            fused_count += 1
            # 融合后当前算子的消费者归并到前驱
            consumer_count[in_preds[0]] = consumer_count[name]
        return fused_count

    def kernel_count(self) -> int:
        """融合后需要 launch 的 kernel 数 = 未被融合的算子数。"""
        return sum(1 for op in self._ops.values() if not op.fused)

    def total_ops(self) -> int:
        return len(self._ops)

    @staticmethod
    def build_transformer_block(prefix: str = "blk") -> "ComputeGraph":
        """构造一个简化版 Transformer block 的计算图。

        典型结构（每个 block）：
            x → LayerNorm → MatMul(Q) ─┐
            x → LayerNorm → MatMul(K) ─┤→ Attention → Softmax → MatMul(out) → Add(residual)
            x → LayerNorm → MatMul(V) ─┘
            → LayerNorm → MatMul(FFN1) → GELU → MatMul(FFN2) → Add(residual)

        每个 block 约 14 个算子，其中 element-wise（Add/ReLU/GELU/Scale）可融合。
        """
        g = ComputeGraph()
        p = prefix
        g.add_op(Operator("LayerNorm", f"{p}_ln1", [f"{p}_in"]))
        g.add_op(Operator("MatMul", f"{p}_q", [f"{p}_ln1"]))
        g.add_op(Operator("MatMul", f"{p}_k", [f"{p}_ln1"]))
        g.add_op(Operator("MatMul", f"{p}_v", [f"{p}_ln1"]))
        g.add_op(Operator("Attention", f"{p}_attn", [f"{p}_q", f"{p}_k", f"{p}_v"]))
        g.add_op(Operator("Scale", f"{p}_scale", [f"{p}_attn"]))  # 1/sqrt(d)
        g.add_op(Operator("Softmax", f"{p}_softmax", [f"{p}_scale"]))
        g.add_op(Operator("MatMul", f"{p}_attn_out", [f"{p}_softmax", f"{p}_v"]))
        g.add_op(Operator("Add", f"{p}_res1", [f"{p}_attn_out", f"{p}_in"]))  # residual
        g.add_op(Operator("LayerNorm", f"{p}_ln2", [f"{p}_res1"]))
        g.add_op(Operator("MatMul", f"{p}_ffn1", [f"{p}_ln2"]))
        g.add_op(Operator("GELU", f"{p}_gelu", [f"{p}_ffn1"]))
        g.add_op(Operator("MatMul", f"{p}_ffn2", [f"{p}_gelu"]))
        g.add_op(Operator("Add", f"{p}_res2", [f"{p}_ffn2", f"{p}_res1"]))  # residual
        return g


# ============================================================================
# 2. RequestScheduler：请求调度（堆/优先队列，按 SLA 调度）
# ============================================================================


@dataclass
class InferenceRequest:
    """一个推理请求。

    req_id    : 请求 ID
    priority  : 优先级（数值越大越优先；由 SLA 等级决定）
    arrive_ts : 到达时间（ms）
    sla_ms    : SLA 时限（必须在 arrive_ts + sla_ms 前完成）
    tokens    : 输入 token 数
    """

    req_id: int
    priority: int
    arrive_ts: float
    sla_ms: float
    tokens: int = 16

    @property
    def deadline(self) -> float:
        return self.arrive_ts + self.sla_ms


class RequestScheduler:
    """请求调度器：堆/优先队列，按优先级 pop。

    AI 推理服务场景：
    - 高优先级请求（VIP 用户、在线交互）需要优先处理
    - 低优先级请求（后台批处理、离线推理）可延迟
    - SLA（服务等级协议）要求：请求必须在 deadline 前完成，否则违约

    实现用 heapq（小顶堆），存 (-priority, arrive_ts, req_id, request)：
    - 负优先级 → 大优先级先出
    - arrive_ts 作 tie-breaker：同优先级 FIFO
    - req_id 作最终 tie-breaker，避免比较 InferenceRequest

    对比 FIFO（deque）：先到先服务，不看优先级。
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, int, InferenceRequest]] = []

    def push(self, req: InferenceRequest) -> None:
        bisect.insort  # 占位避免未使用警告
        # heapq.push 用 list.append + heapq.heappush，这里手动维护
        import heapq

        heapq.heappush(
            self._heap, (-req.priority, req.arrive_ts, req.req_id, req)
        )

    def pop(self) -> InferenceRequest | None:
        if not self._heap:
            return None
        import heapq

        return heapq.heappop(self._heap)[3]

    def __len__(self) -> int:
        return len(self._heap)


class FIFOScheduler:
    """FIFO 调度器（对照组）：先到先服务，不看优先级。"""

    def __init__(self) -> None:
        self._q: deque[InferenceRequest] = deque()

    def push(self, req: InferenceRequest) -> None:
        self._q.append(req)

    def pop(self) -> InferenceRequest | None:
        return self._q.popleft() if self._q else None

    def __len__(self) -> int:
        return len(self._q)


def simulate_scheduling(
    scheduler, requests: list[InferenceRequest], serve_time_per_token: float = 0.5
) -> tuple[int, int, float]:
    """模拟调度过程，返回 (完成数, SLA 违约数, 平均等待时间)。

    每个请求的服务时间 = tokens * serve_time_per_token（模拟推理耗时）。
    """
    # 按到达时间排序后逐个 push（模拟请求到达）
    pending = sorted(requests, key=lambda r: r.arrive_ts)
    clock = 0.0
    done = 0
    violated = 0
    total_wait = 0.0
    idx = 0
    while idx < len(pending) or len(scheduler) > 0:
        # 把所有已到达的请求 push 进调度器
        while idx < len(pending) and pending[idx].arrive_ts <= clock:
            scheduler.push(pending[idx])
            idx += 1
        if len(scheduler) == 0:
            # 队列空，快进到下一个请求到达
            if idx < len(pending):
                clock = pending[idx].arrive_ts
            continue
        req = scheduler.pop()
        if req is None:
            break
        # 等待时间 = 当前时间 - 到达时间
        wait = max(0.0, clock - req.arrive_ts)
        total_wait += wait
        # 服务
        serve = req.tokens * serve_time_per_token
        clock += serve
        done += 1
        if clock > req.deadline:
            violated += 1
    return done, violated, total_wait / max(done, 1)


# ============================================================================
# 3. LoadBalancer：一致性哈希负载均衡（多 GPU 分发）
# ============================================================================


class ConsistentHashRing:
    """一致性哈希环：多 GPU/多节点分发请求。

    AI 推理服务场景：多个 GPU worker 分担推理请求。
    - 取模哈希：hash(key) % N → 节点。节点增减时几乎所有 key 要迁移。
    - 一致性哈希：把节点和 key 都映射到环上，key 顺时针找最近节点。
      节点增减时只有相邻区间的 key 需迁移（约 K/N 个，K=key 总数）。

    虚拟节点（vnode）：每个真实节点在环上放 v 个虚拟节点，解决数据倾斜。
    - 无 vnode：节点少时分布不均
    - 有 vnode：v=150 时负载方差显著降低

    实现：
    - 环用有序数组 + bisect 二分查找
    - hash 用 md5 取前 8 字节 → int（稳定且分布均匀）
    """

    def __init__(self, vnodes: int = 150) -> None:
        self._vnodes = vnodes
        self._ring: list[int] = []  # 有序的 hash 槽
        self._slot_to_node: dict[int, str] = {}

    @staticmethod
    def _hash(key: str) -> int:
        h = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big")

    def add_node(self, node: str) -> None:
        for i in range(self._vnodes):
            slot = self._hash(f"{node}#{i}")
            self._slot_to_node[slot] = node
            bisect.insort(self._ring, slot)

    def remove_node(self, node: str) -> None:
        for i in range(self._vnodes):
            slot = self._hash(f"{node}#{i}")
            if slot in self._slot_to_node:
                del self._slot_to_node[slot]
                pos = bisect.bisect_left(self._ring, slot)
                if pos < len(self._ring) and self._ring[pos] == slot:
                    self._ring.pop(pos)

    def get_node(self, key: str) -> str | None:
        if not self._ring:
            return None
        slot = self._hash(key)
        pos = bisect.bisect_right(self._ring, slot)
        if pos == len(self._ring):
            pos = 0  # 环绕
        return self._slot_to_node[self._ring[pos]]

    def nodes(self) -> set[str]:
        return set(self._slot_to_node.values())


class ModHashBalancer:
    """取模哈希负载均衡（对照组）：hash(key) % N。

    节点增减时 N 变化，几乎所有 key 重新映射 → 大规模迁移。
    """

    def __init__(self) -> None:
        self._nodes: list[str] = []

    def add_node(self, node: str) -> None:
        self._nodes.append(node)

    def remove_node(self, node: str) -> None:
        if node in self._nodes:
            self._nodes.remove(node)

    def get_node(self, key: str) -> str | None:
        if not self._nodes:
            return None
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
        return self._nodes[h % len(self._nodes)]

    def nodes(self) -> set[str]:
        return set(self._nodes)


def measure_rebalance(
    balancer, initial_nodes: list[str], keys: list[str]
) -> tuple[dict[str, str], dict[str, str], int]:
    """测量节点增减时的 key 迁移量。

    返回 (before_map, after_map, migrated_count)
    """
    for n in initial_nodes:
        balancer.add_node(n)
    before = {k: balancer.get_node(k) for k in keys}
    # 增加一个节点
    balancer.add_node(f"node_{len(initial_nodes)}")
    after = {k: balancer.get_node(k) for k in keys}
    migrated = sum(1 for k in keys if before[k] != after[k])
    return before, after, migrated


# ============================================================================
# 4. MemoryPaged：页表显存管理（PagedAttention 的分块分配）
# ============================================================================


@dataclass
class Block:
    """一个显存块（物理页）。

    block_id : 物理块 ID
    size     : 块大小（token 数）
    used     : 已用 token 数
    owner    : 所属序列 ID（None = 空闲）
    """

    block_id: int
    size: int
    used: int = 0
    owner: int | None = None

    @property
    def free(self) -> int:
        return self.size - self.used

    @property
    def is_free(self) -> bool:
        return self.owner is None


class MemoryPaged:
    """页表显存管理：PagedAttention 的分块分配。

    AI 场景：大模型推理时，每个请求的 KV cache 随生成长度增长。
    - 连续分配：预分配 max_seq_len 的连续显存 → 大量浪费（实际长度远小于 max）
    - 分页分配：把显存分成固定大小的 block，按需分配，逻辑页→物理页映射

    PagedAttention（vLLM）的核心思想：
    - 显存被分成固定大小的小 block（如 16 token / block）
    - 每个序列的逻辑页通过页表映射到物理页
    - 物理页可分散在显存任意位置（不需要连续）
    - 序列增长时按需分配新 block，不需要重新分配整个序列

    优势：
    - 显存利用率高（按需分配，不预留 max_seq_len）
    - 碎片率低（block 固定大小，可任意拼接）
    - 可共享（相同 prefix 的序列共享 block，如 system prompt）

    实现：
    - _blocks   : 物理块池
    - _page_table: seq_id → [逻辑页 → 物理块 ID 列表]
    - _free_list: 空闲块 ID 集合
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self._block_size = block_size
        self._blocks: dict[int, Block] = {
            i: Block(i, block_size) for i in range(num_blocks)
        }
        self._page_table: dict[int, list[int]] = {}
        self._free_list: set[int] = set(range(num_blocks))
        self._total_blocks = num_blocks

    def allocate(self, seq_id: int, n_tokens: int) -> bool:
        """为序列 seq_id 分配 n_tokens 的显存。返回是否成功。"""
        if seq_id in self._page_table:
            return self.append(seq_id, n_tokens)
        n_blocks_needed = math.ceil(n_tokens / self._block_size)
        if n_blocks_needed > len(self._free_list):
            return False  # 显存不足
        blocks = []
        for _ in range(n_blocks_needed):
            bid = self._free_list.pop()
            blk = self._blocks[bid]
            blk.owner = seq_id
            blocks.append(bid)
        self._page_table[seq_id] = blocks
        # 标记最后一个块的 used
        last = self._blocks[blocks[-1]]
        last.used = n_tokens - (n_blocks_needed - 1) * self._block_size
        for bid in blocks[:-1]:
            self._blocks[bid].used = self._block_size
        return True

    def append(self, seq_id: int, n_new_tokens: int) -> bool:
        """序列 seq_id 追加 n_new_tokens 个 token。"""
        if seq_id not in self._page_table:
            return self.allocate(seq_id, n_new_tokens)
        blocks = self._page_table[seq_id]
        last = self._blocks[blocks[-1]]
        remaining = n_new_tokens
        # 先填满最后一个块
        if last.free > 0:
            take = min(last.free, remaining)
            last.used += take
            remaining -= take
        # 还需要新块
        while remaining > 0:
            if not self._free_list:
                return False
            bid = self._free_list.pop()
            blk = self._blocks[bid]
            blk.owner = seq_id
            take = min(self._block_size, remaining)
            blk.used = take
            blocks.append(bid)
            remaining -= take
        return True

    def free(self, seq_id: int) -> None:
        """释放序列 seq_id 的所有块。"""
        if seq_id not in self._page_table:
            return
        for bid in self._page_table[seq_id]:
            blk = self._blocks[bid]
            blk.owner = None
            blk.used = 0
            self._free_list.add(bid)
        del self._page_table[seq_id]

    def utilization(self) -> float:
        """显存利用率 = 已用 token 数 / 总 token 容量。"""
        total_cap = self._total_blocks * self._block_size
        used = sum(b.used for b in self._blocks.values())
        return used / total_cap

    def fragmentation(self) -> float:
        """碎片率 = 已分配但未填满的块数 / 总块数。"""
        n_partial = sum(
            1
            for b in self._blocks.values()
            if b.owner is not None and 0 < b.used < b.size
        )
        return n_partial / self._total_blocks

    @property
    def free_blocks(self) -> int:
        return len(self._free_list)


class MemoryContiguous:
    """连续显存分配（对照组）：每个序列预分配 max_seq_len 的连续空间。

    问题：实际生成长度远小于 max_seq_len 时大量浪费。
    """

    def __init__(self, num_slots: int, max_seq_len: int) -> None:
        self._max = max_seq_len
        self._free_slots = num_slots
        self._used: dict[int, int] = {}  # seq_id → 实际 token 数

    def allocate(self, seq_id: int, n_tokens: int) -> bool:
        if self._free_slots <= 0 or n_tokens > self._max:
            return False
        self._free_slots -= 1
        self._used[seq_id] = n_tokens
        return True

    def append(self, seq_id: int, n_new: int) -> bool:
        if seq_id not in self._used:
            return self.allocate(seq_id, n_new)
        new_total = self._used[seq_id] + n_new
        if new_total > self._max:
            return False
        self._used[seq_id] = new_total
        return True

    def free(self, seq_id: int) -> None:
        if seq_id in self._used:
            del self._used[seq_id]
            self._free_slots += 1

    def utilization(self) -> float:
        """实际利用率 = 实际 token 数 / 预分配容量。"""
        total_cap = (self._free_slots + len(self._used)) * self._max
        if total_cap == 0:
            return 0.0
        used = sum(self._used.values())
        return used / total_cap

    @property
    def free_slots(self) -> int:
        return self._free_slots


# ============================================================================
# 5. 协作：模拟完整推理服务
# ============================================================================


@dataclass
class ServeStats:
    """推理服务运行统计。"""

    total_requests: int = 0
    completed: int = 0
    sla_violated: int = 0
    avg_wait_ms: float = 0.0
    gpu_load: dict[str, int] = field(default_factory=dict)
    kernels_per_block: int = 0
    fused_per_block: int = 0
    mem_utilization: float = 0.0
    mem_fragmentation: float = 0.0


def run_inference_service(
    n_requests: int = 500,
    n_gpus: int = 4,
    n_blocks: int = 256,
    block_size: int = 16,
    max_seq_len: int = 512,
    seed: int = 42,
) -> tuple[ServeStats, list[float]]:
    """模拟完整推理服务，4 种数据结构协作。

    流程：
      1. 生成 n_requests 个推理请求（不同优先级、SLA、token 数）
      2. RequestScheduler（堆）：按优先级调度
      3. LoadBalancer（一致性哈希）：分发到 n_gpus 个 GPU
      4. OperatorFusion（图）：每个 GPU 上的 Transformer block 算子融合
      5. MemoryPaged（页表）：KV cache 分块分配

    返回：(stats, util_curve)
    """
    rng = random.Random(seed)
    stats = ServeStats()
    stats.total_requests = n_requests

    # ---- 1. 生成请求 ----
    requests: list[InferenceRequest] = []
    for i in range(n_requests):
        # 优先级 1-5，5 最高（VIP）；80% 是普通用户（priority 2-3）
        r = rng.random()
        if r < 0.1:
            priority = 5  # VIP
        elif r < 0.3:
            priority = 4
        elif r < 0.7:
            priority = 3
        elif r < 0.9:
            priority = 2
        else:
            priority = 1
        arrive = i * rng.uniform(2.0, 8.0)  # 错峰到达，间隔足够大避免过载
        sla = rng.uniform(40, 160)  # SLA 40-160ms
        tokens = rng.randint(8, 64)
        requests.append(InferenceRequest(i, priority, arrive, sla, tokens))

    # ---- 2. 优先级调度 ----
    scheduler = RequestScheduler()
    done, violated, avg_wait = simulate_scheduling(
        scheduler, requests, serve_time_per_token=0.15
    )
    stats.completed = done
    stats.sla_violated = violated
    stats.avg_wait_ms = avg_wait

    # ---- 3. 一致性哈希分发到 GPU ----
    lb = ConsistentHashRing(vnodes=150)
    gpus = [f"gpu_{i}" for i in range(n_gpus)]
    for g in gpus:
        lb.add_node(g)
    gpu_count: dict[str, int] = {g: 0 for g in gpus}
    for req in requests:
        g = lb.get_node(f"req_{req.req_id}")
        if g:
            gpu_count[g] += 1
    stats.gpu_load = gpu_count

    # ---- 4. 算子融合（每个 GPU 跑一个 Transformer block）----
    g = ComputeGraph.build_transformer_block("blk")
    before_kernels = g.kernel_count()
    fused = g.fuse_elementwise()
    after_kernels = g.kernel_count()
    stats.kernels_per_block = before_kernels
    stats.fused_per_block = fused

    # ---- 5. 页表显存管理 ----
    mem = MemoryPaged(num_blocks=n_blocks, block_size=block_size)
    util_curve: list[float] = []
    for req in requests:
        # 模拟生成长度（实际远小于 max_seq_len）
        gen_len = rng.randint(16, 128)
        ok = mem.allocate(req.req_id, gen_len)
        if not ok:
            # 显存不足，释放一些老的
            old_seq = next(iter(mem._page_table), None)
            if old_seq is not None:
                mem.free(old_seq)
                mem.allocate(req.req_id, gen_len)
        util_curve.append(mem.utilization())
        # 模拟请求完成后释放（部分常驻）
        if rng.random() < 0.7:
            mem.free(req.req_id)
    stats.mem_utilization = sum(util_curve) / len(util_curve) if util_curve else 0.0
    stats.mem_fragmentation = mem.fragmentation()

    return stats, util_curve


# ============================================================================
# 6. 性能对比
# ============================================================================


def bench_operator_fusion() -> dict[str, int]:
    """对比算子融合前后的 kernel 数量。

    构造 12 个 Transformer block，统计融合前后 kernel 数。
    """
    n_blocks = 12
    total_ops = 0
    total_kernels_before = 0
    total_fused = 0
    for i in range(n_blocks):
        g = ComputeGraph.build_transformer_block(f"blk{i}")
        total_ops += g.total_ops()
        total_kernels_before += g.kernel_count()
        total_fused += g.fuse_elementwise()
    total_kernels_after = total_kernels_before - total_fused
    return {
        "n_blocks": n_blocks,
        "total_ops": total_ops,
        "kernels_before": total_kernels_before,
        "kernels_after": total_kernels_after,
        "fused": total_fused,
    }


def bench_priority_vs_fifo() -> dict[str, dict[str, float]]:
    """对比优先级调度 vs FIFO 的调度耗时。

    注：Python 中 heapq 是纯 Python 实现，deque 是 C 优化，所以 heapq 在
    单次 pop 耗时上不占优。优先级调度的核心优势是 **SLA 违约率**（见下），
    这里测耗时仅作参考。
    """
    rng = random.Random(0)
    requests = []
    for i in range(300):
        r = rng.random()
        priority = 5 if r < 0.1 else (4 if r < 0.3 else (3 if r < 0.7 else (2 if r < 0.9 else 1)))
        arrive = i * rng.uniform(2.0, 8.0)
        sla = rng.uniform(40, 160)
        tokens = rng.randint(8, 64)
        requests.append(InferenceRequest(i, priority, arrive, sla, tokens))

    def run_priority() -> None:
        s = RequestScheduler()
        simulate_scheduling(s, requests, serve_time_per_token=0.15)

    def run_fifo() -> None:
        s = FIFOScheduler()
        simulate_scheduling(s, requests, serve_time_per_token=0.15)

    return compare({"FIFO 调度": run_fifo, "优先级调度": run_priority}, repeat=3)


def measure_sla_violation() -> dict[str, int]:
    """测量两种调度的 SLA 违约次数。

    优先级调度的核心价值：高优先级请求（VIP/在线交互）优先服务，SLA 违约率
    显著低于 FIFO。这里构造一个**过载**场景（服务能力略低于请求速率），
    让两种调度的违约差异显现。
    """
    rng = random.Random(0)
    requests = []
    for i in range(300):
        r = rng.random()
        priority = 5 if r < 0.1 else (4 if r < 0.3 else (3 if r < 0.7 else (2 if r < 0.9 else 1)))
        arrive = i * rng.uniform(1.0, 3.0)  # 更密的请求 → 轻度过载
        sla = rng.uniform(30, 120)
        tokens = rng.randint(8, 64)
        requests.append(InferenceRequest(i, priority, arrive, sla, tokens))

    s1 = RequestScheduler()
    _, vio_p, _ = simulate_scheduling(s1, requests, serve_time_per_token=0.2)
    s2 = FIFOScheduler()
    _, vio_f, _ = simulate_scheduling(s2, requests, serve_time_per_token=0.2)
    return {"FIFO 违约": vio_f, "优先级违约": vio_p, "总请求": len(requests)}


def bench_consistent_vs_mod() -> dict[str, int]:
    """对比一致性哈希 vs 取模哈希在节点增减时的 key 迁移量。

    场景：3 个节点 → 加 1 个节点，看 1000 个 key 有多少要迁移。
    """
    rng = random.Random(0)
    keys = [f"req_{i}" for i in range(1000)]
    initial_nodes = ["node_0", "node_1", "node_2"]

    # 一致性哈希
    ch = ConsistentHashRing(vnodes=150)
    _, _, migrated_ch = measure_rebalance(ch, initial_nodes, keys)

    # 取模哈希
    mh = ModHashBalancer()
    _, _, migrated_mh = measure_rebalance(mh, initial_nodes, keys)

    return {
        "一致性哈希迁移": migrated_ch,
        "取模哈希迁移": migrated_mh,
        "总 key 数": len(keys),
    }


def bench_paged_vs_contiguous() -> dict[str, float]:
    """对比分页显存 vs 连续分配的利用率。

    场景：256 个 block * 16 token = 4096 token 容量。
    200 个请求，每个生成长度 16-128（远小于 max_seq_len=512）。
    """
    rng = random.Random(0)
    n_blocks = 256
    block_size = 16
    max_seq_len = 512
    n_req = 200

    # 分页
    paged = MemoryPaged(n_blocks, block_size)
    for i in range(n_req):
        gen = rng.randint(16, 128)
        if not paged.allocate(i, gen):
            old = next(iter(paged._page_table), None)
            if old is not None:
                paged.free(old)
                paged.allocate(i, gen)
    util_paged = paged.utilization()

    # 连续（同样总容量）
    n_slots = (n_blocks * block_size) // max_seq_len
    contig = MemoryContiguous(n_slots, max_seq_len)
    for i in range(n_req):
        gen = rng.randint(16, 128)
        contig.allocate(i, gen)
    util_contig = contig.utilization()

    return {
        "分页利用率": util_paged,
        "连续利用率": util_contig,
        "分页碎片率": paged.fragmentation(),
    }


# ============================================================================
# 7. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 18 章 demo：模型部署与推理服务 × 数据结构协作")
    print("图(算子融合) + 堆(请求调度) + 一致性哈希(负载均衡) + 页表(显存管理)")
    print("=" * 72)

    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- 7.1 数据结构单元自检 ----
    print("\n[1] 数据结构单元自检")
    print("-" * 72)

    # ComputeGraph
    g = ComputeGraph.build_transformer_block("blk")
    before = g.kernel_count()
    fused = g.fuse_elementwise()
    after = g.kernel_count()
    print(
        f"  ComputeGraph: Transformer block {g.total_ops()} 个算子, "
        f"融合前 {before} kernels, 融合 {fused} 个, 融合后 {after} kernels"
    )
    assert after < before, "融合后 kernel 数应减少"

    # RequestScheduler
    sched = RequestScheduler()
    for i in range(5):
        sched.push(InferenceRequest(i, priority=i % 3, arrive_ts=float(i), sla_ms=100))
    out = []
    while len(sched) > 0:
        out.append(sched.pop().priority)
    print(f"  RequestScheduler: push 5 请求(优先级 0,1,2,0,1) → pop 顺序 {out}")
    assert out[0] == 2, "最高优先级应先出"

    # ConsistentHashRing
    ring = ConsistentHashRing(vnodes=100)
    for n in ["gpu0", "gpu1", "gpu2"]:
        ring.add_node(n)
    dist: dict[str, int] = {"gpu0": 0, "gpu1": 0, "gpu2": 0}
    for i in range(300):
        dist[ring.get_node(f"key_{i}")] += 1
    print(f"  ConsistentHashRing(3 节点, 300 keys): 分布 {dist}")
    assert all(v > 50 for v in dist.values()), "分布应大致均匀"

    # MemoryPaged
    mem = MemoryPaged(num_blocks=8, block_size=4)
    ok1 = mem.allocate(1, 6)  # 需要 2 个 block
    ok2 = mem.allocate(2, 3)  # 需要 1 个 block
    util = mem.utilization()
    mem.free(1)
    util2 = mem.utilization()
    print(
        f"  MemoryPaged(8 blocks × 4 tokens): "
        f"seq1(6t)={ok1} seq2(3t)={ok2} 利用率={util:.2f}, free后={util2:.2f}"
    )
    assert ok1 and ok2

    # ---- 7.2 模拟完整推理服务 ----
    print("\n[2] 模拟完整推理服务（4 种数据结构协作）")
    print("-" * 72)
    stats, util_curve = run_inference_service(
        n_requests=500, n_gpus=4, n_blocks=256, block_size=16, max_seq_len=512
    )
    print(f"  总请求数          : {stats.total_requests}")
    print(f"  完成数            : {stats.completed}")
    print(f"  SLA 违约数        : {stats.sla_violated}")
    print(f"  平均等待时间(ms)  : {stats.avg_wait_ms:.2f}")
    print(f"  GPU 负载分布      : {stats.gpu_load}")
    print(
        f"  每 block 算子     : {stats.kernels_per_block} 个, "
        f"融合 {stats.fused_per_block} 个, "
        f"融合后 {stats.kernels_per_block - stats.fused_per_block} kernels"
    )
    print(f"  显存利用率        : {stats.mem_utilization:.2%}")
    print(f"  显存碎片率        : {stats.mem_fragmentation:.2%}")

    # ---- 7.3 性能对比 ----
    print("\n[3] 性能对比")
    print("-" * 72)

    print("\n  (a) 算子融合前后 kernel 数量（12 个 Transformer block）")
    r_fuse = bench_operator_fusion()
    print(f"      总算子数        : {r_fuse['total_ops']}")
    print(f"      融合前 kernel 数: {r_fuse['kernels_before']}")
    print(f"      融合后 kernel 数: {r_fuse['kernels_after']}")
    print(f"      融合算子数      : {r_fuse['fused']}")
    print(
        f"      kernel 减少比例 : "
        f"{1 - r_fuse['kernels_after'] / r_fuse['kernels_before']:.1%}"
    )

    print("\n  (b) 优先级调度 vs FIFO（300 请求，混合优先级）")
    r_sched = bench_priority_vs_fifo()
    print(format_table(r_sched, baseline="FIFO 调度"))
    vio = measure_sla_violation()
    print(
        f"      SLA 违约对比：FIFO={vio['FIFO 违约']}/{vio['总请求']}, "
        f"优先级={vio['优先级违约']}/{vio['总请求']}"
    )

    print("\n  (c) 一致性哈希 vs 取模哈希（3→4 节点，1000 keys 迁移量）")
    r_hash = bench_consistent_vs_mod()
    print(f"      一致性哈希迁移 : {r_hash['一致性哈希迁移']}/{r_hash['总 key 数']}")
    print(f"      取模哈希迁移   : {r_hash['取模哈希迁移']}/{r_hash['总 key 数']}")
    ratio = r_hash["取模哈希迁移"] / max(r_hash["一致性哈希迁移"], 1)
    print(f"      取模迁移是一致性的 {ratio:.1f}x")

    print("\n  (d) 分页显存 vs 连续分配（256 blocks × 16 tokens，200 请求）")
    r_mem = bench_paged_vs_contiguous()
    print(f"      分页利用率 : {r_mem['分页利用率']:.2%}")
    print(f"      连续利用率 : {r_mem['连续利用率']:.2%}")
    print(f"      分页碎片率 : {r_mem['分页碎片率']:.2%}")
    print(f"      分页/连续利用率比: {r_mem['分页利用率'] / r_mem['连续利用率']:.1f}x")

    # ---- 7.4 保存图 ----
    print("\n[4] 保存性能对比图到 figures/")
    print("-" * 72)

    # 算子融合前后 kernel 数
    save_bar(
        {"融合前": r_fuse["kernels_before"], "融合后": r_fuse["kernels_after"]},
        figures_dir / "fig_fusion_kernels.png",
        title="算子融合前后 kernel 数量（12 个 Transformer block）",
        ylabel="kernel 数",
        baseline="融合前",
    )

    # 优先级 vs FIFO 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_sched.items()},
        figures_dir / "fig_priority_vs_fifo.png",
        title="优先级调度 vs FIFO（300 请求）",
        ylabel="耗时 (ms)",
        baseline="FIFO 调度",
    )

    # 一致性哈希 vs 取模 迁移量
    save_bar(
        {"一致性哈希": r_hash["一致性哈希迁移"], "取模哈希": r_hash["取模哈希迁移"]},
        figures_dir / "fig_hash_rebalance.png",
        title="节点增减时 key 迁移量（3→4 节点，1000 keys）",
        ylabel="迁移 key 数",
        baseline="取模哈希",
    )

    # 分页 vs 连续 利用率
    save_bar(
        {"分页显存": r_mem["分页利用率"] * 100, "连续分配": r_mem["连续利用率"] * 100},
        figures_dir / "fig_paged_vs_contiguous.png",
        title="分页显存 vs 连续分配利用率",
        ylabel="利用率 (%)",
        baseline="连续分配",
    )

    # 显存利用率曲线
    if util_curve:
        step = max(1, len(util_curve) // 200)
        save_line(
            {"显存利用率": [x * 100 for x in util_curve[::step]]},
            figures_dir / "fig_mem_util_curve.png",
            title="推理过程中显存利用率曲线",
            ylabel="利用率 (%)",
            xlabel="请求序号（降采样）",
        )

    print(f"  图已保存到：{figures_dir}")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"    - {p.name}")

    # ---- 7.5 协作总结 ----
    print("\n[5] 4 种数据结构协作总结")
    print("-" * 72)
    print("  推理请求到达")
    print("    │")
    print("    ▼")
    print("  [堆/优先队列]   请求调度：按 SLA/优先级 pop，高优先级先服务")
    print("    │               对比 FIFO：SLA 违约率显著降低")
    print("    ▼")
    print("  [一致性哈希环]  负载均衡：分发到多 GPU，节点增减时迁移量小")
    print("    │               对比取模哈希：迁移量从 ~750 降到 ~250")
    print("    ▼")
    print("  [计算图]        算子融合：合并相邻 element-wise 算子，减少 kernel launch")
    print("    │               每 block kernel 数从 14 降到 ~10")
    print("    ▼")
    print("  [页表]          显存管理：PagedAttention 分块分配 KV cache")
    print("    │               对比连续分配：利用率从 ~20% 提升到 ~60%+")
    print("    ▼")
    print("  输出 token → 返回客户端")

    print("\n" + "=" * 72)
    print("done.")


if __name__ == "__main__":
    main()