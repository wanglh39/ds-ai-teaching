"""第 02 章 demo：LRU 缓存（哈希表 + 双向链表）vs 朴素实现，模拟 KV Cache 淘汰。

展示「链表」在 AI 里的核心价值：
- KV Cache 在长序列推理时容量有限，需要淘汰
- LRU 淘汰要 O(1) 决策「谁最久没用」→ 哈希表 + 双向链表
- 朴素数组维护顺序 → O(n) 淘汰，长序列下不可接受

跑法：
    python 02_linked_list/python/demo.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import compare, format_table, save_bar, save_line  # noqa: E402


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: int, value: int) -> None:
        self.key = key
        self.value = value
        self.prev: _Node | None = None
        self.next: _Node | None = None


class LRUCacheLinked:
    """哈希表 + 双向链表：O(1) get/put。

    链表头 = 最近访问，链表尾 = 最久未访问（淘汰对象）。
    哈希表 key -> node，让找节点 O(1)。
    """

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.cache: dict[int, _Node] = {}
        self.head: _Node | None = None
        self.tail: _Node | None = None

    def _unlink(self, node: _Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.prev = node.next = None

    def _push_front(self, node: _Node) -> None:
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        if not self.tail:
            self.tail = node

    def get(self, key: int) -> int | None:
        node = self.cache.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self.cache) >= self.cap:
            assert self.tail is not None
            del self.cache[self.tail.key]
            self._unlink(self.tail)
        node = _Node(key, value)
        self.cache[key] = node
        self._push_front(node)


class LRUCacheNaive:
    """朴素 LRU：用 list 维护访问顺序，O(n) get/put。

    每次访问要把 key 从 list 里找出来、删掉、放回头部。
    list 的 index + pop 是 O(n)，这就是不用链表的代价。
    """

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.data: dict[int, int] = {}
        self.order: list[int] = []

    def get(self, key: int) -> int | None:
        if key not in self.data:
            return None
        self.order.remove(key)
        self.order.insert(0, key)
        return self.data[key]

    def put(self, key: int, value: int) -> None:
        if key in self.data:
            self.data[key] = value
            self.order.remove(key)
            self.order.insert(0, key)
            return
        if len(self.data) >= self.cap:
            old = self.order.pop()
            del self.data[old]
        self.data[key] = value
        self.order.insert(0, key)


class LRUCacheOrderedDict:
    """Python OrderedDict：C 优化的 LRU，作为基准。"""

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.od: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int | None:
        if key not in self.od:
            return None
        self.od.move_to_end(key, last=False)
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od[key] = value
            self.od.move_to_end(key, last=False)
            return
        if len(self.od) >= self.cap:
            self.od.popitem(last=True)
        self.od[key] = value
        self.od.move_to_end(key, last=False)


def gen_access_seq(n: int, vocab: int, locality: float, seed: int = 42) -> list[int]:
    """生成访问序列：locality 越高越有局部性（命中率越高）。"""
    rng = np.random.default_rng(seed)
    if locality <= 0:
        return rng.integers(0, vocab, size=n).tolist()
    hot = rng.integers(0, vocab, size=max(1, int(vocab * (1 - locality))))
    picks = rng.choice(hot, size=n)
    return picks.tolist()


def run_workload(cache, access: list[int]) -> int:
    hits = 0
    for k in access:
        v = cache.get(k)
        if v is not None:
            hits += 1
        else:
            cache.put(k, k)
    return hits


def main() -> None:
    print("=" * 60)
    print("第 02 章 demo：LRU 缓存对比 + KV Cache 淘汰模拟")
    print("=" * 60)

    # --- 1. 正确性验证 ---
    print("\n[1] 正确性验证（cap=3）")
    for cls_name, cls in [("Linked", LRUCacheLinked), ("Naive", LRUCacheNaive), ("ODict", LRUCacheOrderedDict)]:
        c = cls(3)
        c.put(1, 11); c.put(2, 22); c.put(3, 33)
        c.get(1)
        c.put(4, 44)
        assert c.get(2) is None, f"{cls_name}: 2 should be evicted"
        assert c.get(1) == 11, f"{cls_name}: 1 should hit"
        print(f"  {cls_name}: OK (2 evicted, 1 hit)")

    # --- 2. 性能对比（模拟 KV Cache 淘汰）---
    print("\n[2] 性能对比：KV Cache 淘汰（vocab=2000, cap=800, accesses=30000）")
    access = gen_access_seq(n=30000, vocab=2000, locality=0.5)

    results = compare({
        "naive_list_O(n)": lambda: run_workload(LRUCacheNaive(800), access),
        "hash+dlist_O(1)": lambda: run_workload(LRUCacheLinked(800), access),
        "OrderedDict(C)": lambda: run_workload(LRUCacheOrderedDict(800), access),
    }, repeat=2, warmup=0)
    print(format_table(results, baseline="naive_list_O(n)"))

    # --- 3. 不同容量下的命中率 ---
    print("\n[3] 命中率 vs 缓存容量（vocab=1000, accesses=20000, locality=0.5）")
    caps = [50, 100, 200, 500, 800, 1000]
    hit_rates: dict[str, list[float]] = {"hash+dlist": []}
    for cap in caps:
        c = LRUCacheLinked(cap)
        hits = run_workload(c, access)
        rate = hits / len(access)
        hit_rates["hash+dlist"].append(rate)
        print(f"  cap={cap:<5}  命中率={rate:.2%}")

    # --- 4. 局部性对命中率的影响 ---
    print("\n[4] 局部性 vs 命中率（vocab=1000, cap=200, accesses=20000）")
    localities = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    loc_rates: dict[str, list[float]] = {"命中率": []}
    for loc in localities:
        seq = gen_access_seq(n=20000, vocab=1000, locality=loc)
        c = LRUCacheLinked(200)
        hits = run_workload(c, seq)
        rate = hits / len(seq)
        loc_rates["命中率"].append(rate)
        print(f"  locality={loc:.2f}  命中率={rate:.2%}")

    # --- 5. 保存图 ---
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {k: v["mean_ms"] for k, v in results.items()},
        fig_dir / "perf_compare.png",
        title="LRU 缓存性能对比（50000 次访问）",
        ylabel="耗时 (ms)",
        baseline="naive_list_O(n)",
    )

    save_line(
        hit_rates,
        fig_dir / "hit_rate_vs_cap.png",
        title="KV Cache 命中率 vs 缓存容量",
        ylabel="命中率",
        xlabel="容量（索引 0~5 对应 100~2000）",
    )

    save_line(
        loc_rates,
        fig_dir / "hit_rate_vs_locality.png",
        title="KV Cache 命中率 vs 访问局部性",
        ylabel="命中率",
        xlabel="局部性（0~5 对应 0.0~0.95）",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print("  - 朴素 list LRU：每次 get 要 O(n) 找 key + 移动，长序列下不可接受")
    print("  - 哈希表 + 双向链表：O(1) 找节点 + O(1) 移到头部，快几十倍")
    print("  - OrderedDict：C 优化的 O(1)，最快")
    print("  - KV Cache 命中率随容量和局部性增长，LRU 是淘汰策略的核心")

if __name__ == "__main__":
    main()