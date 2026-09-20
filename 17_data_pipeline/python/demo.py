"""第 17 章 demo：数据管道与流式处理 × 数据结构协作（队列 + 环形缓冲 + 树状数组 + 布隆过滤器）

扩展专题：一个 AI 应用场景（AI 训练数据的流式加载 / 滑动窗口统计 / 近似去重）
→ 多种数据结构协作
- PipelineQueue : 多阶段管道（队列连接 producer→processor→augmenter→batcher）
- RingBuffer    : 流式数据窗口（环形缓冲，固定大小，新数据覆盖最老的）
- BIT           : 树状数组（O(log n) 区间和 / 前缀和，滑动窗口统计）
- BloomFilter   : 布隆过滤器（流式去重，O(1) 判重，可控误判率）

模拟完整数据管道：
  原始样本流 → [队列] 多阶段 pipeline（加载→预处理→增强→batch）
             → [环形缓冲] 滑动窗口保留最近 N 条
             → [BIT]      窗口内区间和统计（如最近 loss 之和）
             → [布隆]     流式去重（重复样本不进 batch）
             → batch 输出

性能对比：
1. BIT 区间和 vs 朴素累加（O(log n) vs O(n)）
2. 布隆去重 vs 集合去重（内存 & 速度）
3. 管道并行 vs 串行（多阶段流水线吞吐提升）

跑法：
    python 17_data_pipeline/python/demo.py
"""

from __future__ import annotations


import math
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. PipelineQueue：多阶段管道（用队列连接各阶段）
# ============================================================================


@dataclass
class Sample:
    """一个训练样本。

    id       : 样本 ID（用于去重）
    feature  : 特征向量（模拟图像 / 文本的 embedding）
    label    : 标签
    loss     : 该样本的损失（模拟训练过程中的 loss）
    """

    id: int
    feature: list[float]
    label: int
    loss: float = 0.0


class PipelineQueue:
    """多阶段管道：用队列把各处理阶段串联起来。

    AI 训练数据加载的经典 pipeline：
        producer (读盘) → processor (解码/归一化) → augmenter (数据增强) → batcher (组 batch)

    每个阶段之间用一个有界队列连接：
    - 上游 push 满了就阻塞（背压，防止 OOM）
    - 下游 pop 空了就等待

    教学简化：用 deque + 容量上限模拟有界队列，单线程轮转各阶段，
    展示「流水线并行」的思想（真实场景用多进程 / 多线程 / asyncio）。

    - push(item) : 入队（满则丢弃 oldest，模拟背压丢帧）
    - pop()      : 出队（空则返回 None）
    - size()     : 当前队列长度
    """

    def __init__(self, capacity: int = 64) -> None:
        self._q: deque = deque(maxlen=capacity)
        self._capacity = capacity
        # 统计：入队数、出队数、丢帧数
        self.in_count = 0
        self.out_count = 0
        self.drop_count = 0

    def push(self, item) -> bool:
        """入队。队列满时丢弃最老的（背压），返回是否成功入队。"""
        self.in_count += 1
        if len(self._q) >= self._capacity:
            self._q.popleft()  # 丢最老的
            self.drop_count += 1
        self._q.append(item)
        return True

    def pop(self):
        """出队。空则返回 None。"""
        if not self._q:
            return None
        self.out_count += 1
        return self._q.popleft()

    def size(self) -> int:
        return len(self._q)

    def is_empty(self) -> bool:
        return len(self._q) == 0


# ============================================================================
# 2. RingBuffer：流式数据窗口（环形缓冲）
# ============================================================================


class RingBuffer:
    """环形缓冲：固定大小的滑动窗口，新数据覆盖最老的。

    AI 场景：流式训练中保留「最近 N 个样本 / 最近 N 步的 loss」。
    - 不需要动态扩容，内存固定
    - 写入 O(1)，覆盖最老的也是 O(1)（不用 shift）
    - 支持按时间顺序遍历窗口内所有元素

    实现：预分配定长数组 + 一个写指针（自增取模）。
    - push(x)   : 写入 x，指针前移，超过容量就覆盖最老的
    - window()  : 返回当前窗口内所有元素（按写入顺序）
    - latest(k) : 返回最近 k 个元素

    对比 deque(maxlen=N)：deque 也是 O(1) 滑动窗口，但底层是双向链表块，
    RingBuffer 用纯数组 + 取模，更贴近「环形」的几何直觉，也更容易嵌入到
    BIT / 布隆等需要下标访问的协作场景。
    """

    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self._cap = capacity
        self._buf: list = [None] * capacity
        self._write = 0  # 下一个写入位置
        self._size = 0   # 已写入数量（未满时 < capacity）

    def push(self, x) -> None:
        """写入 x，覆盖最老的（如果已满）。"""
        self._buf[self._write] = x
        self._write = (self._write + 1) % self._cap
        if self._size < self._cap:
            self._size += 1

    def window(self) -> list:
        """返回当前窗口内所有元素（按写入顺序，最老的在前）。"""
        if self._size < self._cap:
            return [self._buf[i] for i in range(self._size)]
        # 已满：从 write 指针开始（最老的），绕一圈
        return [self._buf[(self._write + i) % self._cap] for i in range(self._cap)]

    def latest(self, k: int) -> list:
        """返回最近 k 个元素（最新的在后）。"""
        k = min(k, self._size)
        all_win = self.window()
        return all_win[-k:] if k > 0 else []

    @property
    def capacity(self) -> int:
        return self._cap

    def __len__(self) -> int:
        return self._size


# ============================================================================
# 3. BIT：树状数组（Binary Indexed Tree / Fenwick Tree）
# ============================================================================


class BIT:
    """树状数组：O(log n) 前缀和 / 区间和 / 单点更新。

    AI 场景：滑动窗口内统计指标（最近 N 步的 loss 之和、梯度范数之和、
    样本数等）。朴素做法每次区间和都 O(n) 累加，窗口大且查询频繁时很慢；
    BIT 用「分块前缀和」把更新和查询都压到 O(log n)。

    核心思想（lowbit = x & -x，最低位的 1）：
    - tree[i] 维护原数组区间 [i - lowbit(i) + 1, i] 的和
    - update(i, v)：从 i 往上跳，每跳一步把 tree[i] += v
    - query(i)    ：从 i 往下跳，累加 tree[i]，得到 a[1..i] 的前缀和
    - range_sum(l, r) = query(r) - query(l-1)

    本 demo 用 1-indexed（标准 BIT 写法）。窗口滑动时：
    - 新样本进窗口 → update(pos, value)
    - 老样本出窗口 → update(pos, -value)（或重置槽位）
    """

    def __init__(self, n: int) -> None:
        assert n > 0
        self._n = n
        self._tree = [0.0] * (n + 1)  # 1-indexed
        # 同步维护原数组，方便「单点赋值」语义（覆盖旧值）
        self._arr = [0.0] * (n + 1)

    def update(self, i: int, delta: float) -> None:
        """第 i 位加上 delta（1-indexed）。"""
        assert 1 <= i <= self._n
        self._arr[i] += delta
        while i <= self._n:
            self._tree[i] += delta
            i += i & -i  # lowbit 跳

    def set(self, i: int, value: float) -> None:
        """第 i 位赋值为 value（覆盖旧值）。"""
        assert 1 <= i <= self._n
        delta = value - self._arr[i]
        self.update(i, delta)

    def query(self, i: int) -> float:
        """前缀和 a[1..i]（1-indexed）。i=0 返回 0。"""
        if i <= 0:
            return 0.0
        assert i <= self._n
        s = 0.0
        while i > 0:
            s += self._tree[i]
            i -= i & -i
        return s

    def range_sum(self, l: int, r: int) -> float:
        """区间和 a[l..r]（1-indexed，闭区间）。"""
        assert 1 <= l <= r <= self._n
        return self.query(r) - self.query(l - 1)

    @property
    def n(self) -> int:
        return self._n


class NaivePrefixSum:
    """朴素前缀和：每次区间和 O(n) 累加，单点更新 O(1)。

    用作 BIT 的对照组：查询慢但实现直白。
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._arr = [0.0] * (n + 1)  # 1-indexed

    def update(self, i: int, delta: float) -> None:
        self._arr[i] += delta

    def set(self, i: int, value: float) -> None:
        self._arr[i] = value

    def range_sum(self, l: int, r: int) -> float:
        return sum(self._arr[l : r + 1])

    @property
    def n(self) -> int:
        return self._n


# ============================================================================
# 4. BloomFilter：流式去重（O(1) 判重）
# ============================================================================


class BloomFilter:
    """布隆过滤器：概率型集合，O(1) 判重，可控误判率。

    AI 场景：流式训练数据去重（百万级样本，重复样本会污染统计）。
    - 精确去重用 set（哈希表）：内存 = O(n × key_size)
    - 布隆去重用位数组 + k 个哈希：内存 = O(m bits)，m 远小于 n × key_size
    - 代价：可能误判「重复」（假阳性），但不会误判「不重复」（假阴性）

    参数选择（给定 n 和目标误判率 p）：
        m = -n * ln(p) / (ln2)^2      # 位数
        k = (m / n) * ln2             # 哈希函数个数

    实现：
    - 位数组用 bytearray（每字节 8 位）
    - k 个哈希用「双哈希线性组合」：h_i(x) = (h1 + i * h2) % m，避免 k 个独立哈希
    - add(x)    : 把 k 个位置都置 1
    - contains(x): 检查 k 个位置是否全为 1（全 1 → 可能存在；有 0 → 一定不存在）
    """

    def __init__(self, n: int, p: float = 0.01) -> None:
        assert n > 0 and 0 < p < 1
        self._n = n
        self._p = p
        # 最优位数 m = -n*ln(p)/(ln2)^2
        self._m = max(8, int(-n * math.log(p) / (math.log(2) ** 2)))
        # 最优哈希数 k = (m/n)*ln2
        self._k = max(1, int((self._m / n) * math.log(2)))
        self._bits = bytearray((self._m + 7) // 8)  # 位数组
        self._count = 0  # 已 add 的不同元素估计

    def _hashes(self, key) -> list[int]:
        """双哈希线性组合：h_i = (h1 + i*h2) % m，i = 0..k-1。

        用快速整数混合哈希（不依赖 hashlib，避免 Python 层 hashlib 开销）。
        教学版：重点展示布隆的「位数组 + 多哈希判重」机制，
        生产级会用更严格的哈希（如 xxHash / MurmurHash3）。
        """
        if isinstance(key, int):
            x = key & 0xFFFFFFFFFFFFFFFF
        elif isinstance(key, (bytes, bytearray)):
            # bytes：用 Python 内置 hash 混合（同进程内稳定），再截断到 64 位
            x = hash(bytes(key)) & 0xFFFFFFFFFFFFFFFF
        else:
            x = hash(key) & 0xFFFFFFFFFFFFFFFF
        # 双混合：h1 = mix(x), h2 = mix(x ^ 黄金分割常数)
        h1 = self._mix(x)
        h2 = self._mix(x ^ 0x9E3779B97F4A7C15)
        if h2 == 0:
            h2 = 1  # 保证 h2 非零，避免所有哈希塌缩到同一位
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    @staticmethod
    def _mix(x: int) -> int:
        """64→64 位混合（基于 splitmix64，质量好且 O(1)）。"""
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x = x ^ (x >> 31)
        return x

    def _set_bit(self, pos: int) -> None:
        self._bits[pos >> 3] |= 1 << (pos & 7)

    def _get_bit(self, pos: int) -> int:
        return (self._bits[pos >> 3] >> (pos & 7)) & 1

    def add(self, key) -> None:
        """添加元素（int / str / bytes 均可）。"""
        for pos in self._hashes(key):
            self._set_bit(pos)
        self._count += 1

    def contains(self, key) -> bool:
        """判断是否可能存在（False = 一定不存在，True = 可能存在）。"""
        return all(self._get_bit(pos) for pos in self._hashes(key))

    @property
    def bit_count(self) -> int:
        return self._m

    @property
    def hash_count(self) -> int:
        return self._k

    @property
    def byte_size(self) -> int:
        return len(self._bits)


# ============================================================================
# 5. 协作：模拟完整 AI 数据管道
# ============================================================================


@dataclass
class PipelineStats:
    """管道运行统计。"""

    total_in: int = 0          # 输入样本总数
    deduped: int = 0           # 去重丢弃数
    dropped_by_backpressure: int = 0  # 背压丢帧数
    batches_out: int = 0      # 输出 batch 数
    samples_out: int = 0      # 输出样本数
    window_loss_sum: float = 0.0  # 窗口内 loss 之和（BIT 统计）
    false_positive: int = 0   # 布隆误判次数（用于评估）


def run_pipeline(
    n_samples: int = 2000,
    dup_rate: float = 0.15,
    window_size: int = 128,
    batch_size: int = 32,
    queue_cap: int = 64,
    seed: int = 42,
) -> tuple[PipelineStats, list[float], list[float]]:
    """模拟完整 AI 数据管道，4 种数据结构协作。

    流程：
      1. producer  生成 n_samples 个样本（含 dup_rate 比例的重复 ID）
      2. 队列串联  producer → processor → augmenter → batcher
      3. 环形缓冲  保留最近 window_size 个样本的 loss
      4. BIT       维护窗口内 loss 的前缀和，O(log n) 查询区间和
      5. 布隆      流式去重，重复样本不进 batch
      6. batcher   攒够 batch_size 就输出一个 batch

    返回：(stats, window_loss_curve, batch_loss_curve)
    """
    rng = random.Random(seed)
    stats = PipelineStats()

    # 4 种数据结构
    q_load = PipelineQueue(queue_cap)       # producer → processor
    q_proc = PipelineQueue(queue_cap)       # processor → augmenter
    q_aug = PipelineQueue(queue_cap)        # augmenter → batcher
    ring = RingBuffer(window_size)          # 滑动窗口
    bit = BIT(window_size)                  # 窗口内 loss 区间和
    bloom = BloomFilter(n=n_samples, p=0.01)  # 流式去重

    # 真实 ID 集合（用于评估布隆误判）
    seen_ids: set[int] = set()

    window_loss_curve: list[float] = []
    batch_loss_curve: list[float] = []

    # ---- producer：生成样本（含重复）----
    raw_ids = [rng.randint(0, n_samples * 2) for _ in range(n_samples)]
    # 注入重复：随机选 dup_rate 比例的样本复制已有 ID
    n_dup = int(n_samples * dup_rate)
    dup_indices = rng.sample(range(n_samples), n_dup)
    for idx in dup_indices:
        raw_ids[idx] = raw_ids[rng.randint(0, idx)] if idx > 0 else raw_ids[0]

    # ---- 流式处理：单线程轮转各阶段（展示流水线思想）----
    batch: list[Sample] = []
    bit_slot = 0  # BIT 的写入槽位（环形复用）

    for raw_id in raw_ids:
        stats.total_in += 1

        # producer → q_load
        s = Sample(
            id=raw_id,
            feature=[rng.gauss(0, 1) for _ in range(8)],
            label=rng.randint(0, 9),
            loss=rng.uniform(0.1, 3.0),
        )
        q_load.push(s)

        # processor：q_load → q_proc（模拟解码 / 归一化）
        item = q_load.pop()
        if item is None:
            continue
        # 简单处理：特征归一化（除以范数）
        norm = math.sqrt(sum(x * x for x in item.feature)) or 1.0
        item.feature = [x / norm for x in item.feature]
        q_proc.push(item)

        # augmenter：q_proc → q_aug（模拟数据增强：特征 + 小噪声）
        item = q_proc.pop()
        if item is None:
            continue
        item.feature = [x + rng.gauss(0, 0.01) for x in item.feature]
        q_aug.push(item)

        # batcher：q_aug → 去重 → 窗口 → BIT → batch
        item = q_aug.pop()
        if item is None:
            continue

        # 布隆去重
        maybe_dup = bloom.contains(item.id)
        actually_dup = item.id in seen_ids
        if maybe_dup:
            # 布隆说可能重复
            if not actually_dup:
                stats.false_positive += 1  # 误判
            else:
                stats.deduped += 1
                continue  # 真重复，丢弃
        # 不重复：加入
        bloom.add(item.id)
        seen_ids.add(item.id)

        # 环形缓冲：写入最新 loss
        ring.push(item.loss)

        # BIT：环形复用槽位，更新 loss
        bit_slot = (bit_slot % window_size) + 1  # 1-indexed
        bit.set(bit_slot, item.loss)

        # 窗口内 loss 之和（BIT O(log n) 查询）
        win_len = min(len(ring), window_size)
        if win_len > 0:
            # 查询最近 win_len 个槽位的和（简化：全窗口和）
            stats.window_loss_sum = bit.range_sum(1, window_size)
            window_loss_curve.append(stats.window_loss_sum / win_len)  # 平均 loss

        # 攒 batch
        batch.append(item)
        if len(batch) >= batch_size:
            stats.batches_out += 1
            stats.samples_out += len(batch)
            batch_loss_curve.append(sum(s.loss for s in batch) / len(batch))
            batch.clear()

    stats.dropped_by_backpressure = (
        q_load.drop_count + q_proc.drop_count + q_aug.drop_count
    )
    # 处理剩余 batch
    if batch:
        stats.batches_out += 1
        stats.samples_out += len(batch)
        batch_loss_curve.append(sum(s.loss for s in batch) / len(batch))

    return stats, window_loss_curve, batch_loss_curve


# ============================================================================
# 6. 性能对比
# ============================================================================


def bench_bit_vs_naive() -> dict[str, dict[str, float]]:
    """对比 BIT 区间和 vs 朴素累加。

    场景：n 个元素的数组，先做 n 次单点更新，再做 n 次区间和查询。
    - BIT  ：更新 O(log n)，查询 O(log n)，总 O(n log n)
    - 朴素：更新 O(1)，查询 O(n)，总 O(n^2)
    """
    n = 4096
    rng = random.Random(0)
    values = [rng.uniform(0, 1) for _ in range(n)]
    queries = [(rng.randint(1, n), rng.randint(1, n)) for _ in range(n)]
    queries = [(min(a, b), max(a, b)) for a, b in queries]

    def run_bit() -> None:
        bit = BIT(n)
        for i, v in enumerate(values, 1):
            bit.set(i, v)
        for l, r in queries:
            bit.range_sum(l, r)

    def run_naive() -> None:
        ps = NaivePrefixSum(n)
        for i, v in enumerate(values, 1):
            ps.set(i, v)
        for l, r in queries:
            ps.range_sum(l, r)

    return compare({"朴素累加 O(n)": run_naive, "BIT O(log n)": run_bit}, repeat=3)


def bench_bloom_vs_set() -> dict[str, dict[str, float]]:
    """对比布隆去重 vs 集合去重（速度）。

    注意：Python 中 set 是 C 优化实现，纯 Python 的布隆在速度上不占优。
    布隆的核心优势是**内存**：位数组 vs 哈希表 + 元素对象。
    速度对比展示「Python 层布隆的哈希开销」，内存对比见 measure_bloom_memory()。
    """
    n = 50_000
    rng = random.Random(0)
    keys = [rng.randint(0, 1 << 30) for _ in range(n)]
    queries = [rng.randint(0, 1 << 30) for _ in range(n)]

    def run_set() -> None:
        s: set[int] = set()
        for k in keys:
            s.add(k)
        for q in queries:
            _ = q in s

    def run_bloom() -> None:
        bf = BloomFilter(n=n, p=0.01)
        for k in keys:
            bf.add(k)
        for q in queries:
            _ = bf.contains(q)

    return compare({"set 精确去重": run_set, "布隆近似去重": run_bloom}, repeat=3)


def measure_bloom_memory() -> dict[str, int]:
    """测量布隆 vs set 的内存占用（n=50000 个 int key）。"""
    import sys as _sys

    n = 50_000
    rng = random.Random(0)
    keys = [rng.randint(0, 1 << 30) for _ in range(n)]

    # set 内存：哈希表结构 + 所有 int 元素
    s: set[int] = set()
    for k in keys:
        s.add(k)
    set_mem = _sys.getsizeof(s) + sum(_sys.getsizeof(k) for k in s)

    # 布隆内存：位数组
    bf = BloomFilter(n=n, p=0.01)
    for k in keys:
        bf.add(k)
    bloom_mem = bf.byte_size

    return {"set 内存 (bytes)": set_mem, "布隆内存 (bytes)": bloom_mem}


def bench_pipeline_parallel_vs_serial() -> dict[str, dict[str, float]]:
    """对比管道背压 vs 无界缓存的运行时间。

    见 measure_pipeline_memory() 的内存峰值对比——那才是管道背压的核心价值。
    这里测运行时间作为参考。
    """
    n = 10_000
    capacity = 256

    def run_unbounded() -> None:
        buf: list = []
        for i in range(n):
            buf.append(i)
            if len(buf) >= 4:
                buf.pop(0)

    def run_bounded() -> None:
        q = PipelineQueue(capacity)
        for i in range(n):
            q.push(i)
            if q.size() >= 4:
                q.pop()

    return compare({"无界 list 缓存": run_unbounded, "有界队列(背压)": run_bounded}, repeat=3)


def measure_pipeline_memory() -> dict[str, int]:
    """测量管道两种模式的内存峰值（item 数）。

    模拟突发流量：上游连续 push n 个，下游消费慢（只消费 1/4）。
    - 无界 list：内存峰值 ≈ n（所有样本堆积）
    - 有界队列：内存峰值 = capacity（满了丢老的，背压）
    """
    n = 10_000
    capacity = 256

    # 无界 list：上游全 push，下游慢消费
    buf: list = []
    peak_unbounded = 0
    for i in range(n):
        buf.append(i)
        # 下游每 4 个才消费 1 个
        if i % 4 == 0 and buf:
            buf.pop(0)
        peak_unbounded = max(peak_unbounded, len(buf))

    # 有界队列：上游全 push（满了丢老的），下游慢消费
    q = PipelineQueue(capacity)
    peak_bounded = 0
    drop_count = 0
    for i in range(n):
        if q.size() >= capacity:
            drop_count += 1  # 即将丢帧
        q.push(i)
        if i % 4 == 0 and q.size() > 0:
            q.pop()
        peak_bounded = max(peak_bounded, q.size())

    return {
        "无界 list 峰值(item)": peak_unbounded,
        "有界队列峰值(item)": peak_bounded,
        "有界队列丢帧数": drop_count,
    }


def bench_ringbuffer_vs_list() -> dict[str, dict[str, float]]:
    """对比环形缓冲 vs list.pop(0) 滑动窗口。

    - list.pop(0)：O(n) 左移所有元素（CPython 用 memmove，小 n 时很快）
    - RingBuffer.push：O(1) 覆盖最老的

    cap 取大（8192）让 list.pop(0) 的 O(n) 左移开销明显。
    """
    n = 20_000
    cap = 8192
    rng = random.Random(0)
    data = [rng.uniform(0, 1) for _ in range(n)]

    def run_list() -> None:
        buf: list[float] = []
        for x in data:
            buf.append(x)
            if len(buf) > cap:
                buf.pop(0)  # O(n) 左移

    def run_ring() -> None:
        rb = RingBuffer(cap)
        for x in data:
            rb.push(x)

    return compare({"list.pop(0) O(n)": run_list, "RingBuffer O(1)": run_ring}, repeat=3)


# ============================================================================
# 7. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 17 章 demo：数据管道与流式处理 × 数据结构协作")
    print("队列 + 环形缓冲 + 树状数组(BIT) + 布隆过滤器")
    print("=" * 72)

    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- 7.1 数据结构单元自检 ----
    print("\n[1] 数据结构单元自检")
    print("-" * 72)

    # PipelineQueue
    q = PipelineQueue(capacity=4)
    for i in range(6):  # 超容量，丢最老的
        q.push(i)
    assert q.window() if hasattr(q, "window") else True
    out = []
    while not q.is_empty():
        out.append(q.pop())
    print(f"  PipelineQueue(cap=4) push 0..5 → 出队 {out}（丢了 0, 1）")
    assert out == [2, 3, 4, 5], out

    # RingBuffer
    rb = RingBuffer(4)
    for i in range(6):
        rb.push(i)
    print(f"  RingBuffer(cap=4) push 0..5 → window={rb.window()} latest(2)={rb.latest(2)}")
    assert rb.window() == [2, 3, 4, 5], rb.window()
    assert rb.latest(2) == [4, 5], rb.latest(2)

    # BIT
    bit = BIT(8)
    for i, v in enumerate([1, 2, 3, 4, 5, 6, 7, 8], 1):
        bit.set(i, v)
    print(f"  BIT(8) set 1..8 → range_sum(3,6)={bit.range_sum(3, 6)} (期望 3+4+5+6=18)")
    assert bit.range_sum(3, 6) == 18

    # BloomFilter
    bf = BloomFilter(n=100, p=0.01)
    for k in ["apple", "banana", "cherry"]:
        bf.add(k)
    print(
        f"  BloomFilter(n=100,p=0.01) add 3 keys → "
        f"contains('apple')={bf.contains('apple')} contains('missing')={bf.contains('missing')}"
    )
    assert bf.contains("apple")
    assert not bf.contains("missing")
    print(f"  BloomFilter 参数：m={bf.bit_count} bits, k={bf.hash_count} hashes, {bf.byte_size} bytes")

    # ---- 7.2 模拟完整数据管道 ----
    print("\n[2] 模拟完整 AI 数据管道（4 种数据结构协作）")
    print("-" * 72)
    stats, win_curve, batch_curve = run_pipeline(
        n_samples=2000, dup_rate=0.15, window_size=128, batch_size=32, queue_cap=64
    )
    print(f"  输入样本总数      : {stats.total_in}")
    print(f"  去重丢弃（真重复）: {stats.deduped}")
    print(f"  布隆误判（假阳性）: {stats.false_positive}")
    print(f"  背压丢帧          : {stats.dropped_by_backpressure}")
    print(f"  输出 batch 数     : {stats.batches_out}")
    print(f"  输出样本数        : {stats.samples_out}")
    print(f"  窗口平均 loss     : {stats.window_loss_sum / 128:.4f}")
    print(f"  batch 平均 loss 曲线（前 5）: {[f'{x:.3f}' for x in batch_curve[:5]]}")

    # ---- 7.3 性能对比 ----
    print("\n[3] 性能对比")
    print("-" * 72)

    print("\n  (a) BIT 区间和 vs 朴素累加（n=4096，n 次更新 + n 次查询）")
    r_bit = bench_bit_vs_naive()
    print(format_table(r_bit, baseline="朴素累加 O(n)"))

    print("\n  (b) 布隆去重 vs 集合去重（n=50000 插入 + 查询）")
    r_bloom = bench_bloom_vs_set()
    print(format_table(r_bloom, baseline="set 精确去重"))
    mem_bloom = measure_bloom_memory()
    ratio = mem_bloom["set 内存 (bytes)"] / mem_bloom["布隆内存 (bytes)"]
    print(
        f"      内存对比：set={mem_bloom['set 内存 (bytes)']:,} bytes, "
        f"布隆={mem_bloom['布隆内存 (bytes)']:,} bytes "
        f"(set 是布隆的 {ratio:.1f}x)"
    )
    print(f"      注：Python 中 set 是 C 优化，速度占优；布隆优势在内存（位数组 vs 哈希表+元素）")

    print("\n  (c) 管道背压 vs 无界缓存（n=10000，capacity=256）")
    r_pipe = bench_pipeline_parallel_vs_serial()
    print(format_table(r_pipe, baseline="无界 list 缓存"))
    mem = measure_pipeline_memory()
    print(f"      内存峰值对比：{mem}（有界队列内存固定，无界 list 随流量增长）")

    print("\n  (d) RingBuffer vs list.pop(0) 滑动窗口（n=20000, cap=8192）")
    r_ring = bench_ringbuffer_vs_list()
    print(format_table(r_ring, baseline="list.pop(0) O(n)"))

    # ---- 7.4 保存图 ----
    print("\n[4] 保存性能对比图到 figures/")
    print("-" * 72)

    # BIT vs 朴素
    save_bar(
        {k: v["mean_ms"] for k, v in r_bit.items()},
        figures_dir / "fig_bit_vs_naive.png",
        title="BIT 区间和 vs 朴素累加（n=4096）",
        ylabel="耗时 (ms)",
        baseline="朴素累加 O(n)",
    )

    # 布隆 vs set
    save_bar(
        {k: v["mean_ms"] for k, v in r_bloom.items()},
        figures_dir / "fig_bloom_vs_set.png",
        title="布隆去重 vs 集合去重（n=50000）",
        ylabel="耗时 (ms)",
        baseline="set 精确去重",
    )

    # 管道 vs 串行
    save_bar(
        {k: v["mean_ms"] for k, v in r_pipe.items()},
        figures_dir / "fig_pipeline_vs_serial.png",
        title="管道背压 vs 无界缓存（n=10000）",
        ylabel="耗时 (ms)",
        baseline="无界 list 缓存",
    )

    # RingBuffer vs list
    save_bar(
        {k: v["mean_ms"] for k, v in r_ring.items()},
        figures_dir / "fig_ringbuffer_vs_list.png",
        title="RingBuffer vs list.pop(0) 滑动窗口",
        ylabel="耗时 (ms)",
        baseline="list.pop(0) O(n)",
    )

    # 窗口 loss 曲线
    if win_curve:
        # 降采样以便画图
        step = max(1, len(win_curve) // 200)
        save_line(
            {"窗口平均 loss": win_curve[::step]},
            figures_dir / "fig_window_loss_curve.png",
            title="滑动窗口平均 loss 曲线（流式训练监控）",
            ylabel="平均 loss",
            xlabel="样本序号（降采样）",
        )

    # batch loss 曲线
    if batch_curve:
        save_line(
            {"batch 平均 loss": batch_curve},
            figures_dir / "fig_batch_loss_curve.png",
            title="batch 平均 loss 曲线",
            ylabel="平均 loss",
            xlabel="batch 序号",
        )

    print(f"  图已保存到：{figures_dir}")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"    - {p.name}")

    # ---- 7.5 协作总结 ----
    print("\n[5] 4 种数据结构协作总结")
    print("-" * 72)
    print("  原始样本流")
    print("    │")
    print("    ▼")
    print("  [PipelineQueue] 多阶段管道：producer → processor → augmenter → batcher")
    print("    │               队列连接各阶段，背压防 OOM")
    print("    ▼")
    print("  [RingBuffer]    滑动窗口：保留最近 N 个样本的 loss，O(1) 写入/覆盖")
    print("    │               新数据覆盖最老的，内存固定")
    print("    ▼")
    print("  [BIT]           窗口统计：O(log n) 区间和 / 前缀和")
    print("    │               朴素 O(n) 累加 → BIT O(log n)，窗口越大加速越明显")
    print("    ▼")
    print("  [BloomFilter]   流式去重：O(1) 判重，内存远小于 set")
    print("    │               代价：可控误判率（假阳性，不漏真新样本）")
    print("    ▼")
    print("  batch 输出 → 喂给模型训练")

    print("\n" + "=" * 72)
    print("done.")


if __name__ == "__main__":
    main()