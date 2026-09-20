"""第 15 章 demo：上下文工程与记忆 × 数据结构协作（LRU + Trie + 哈希 + 环形缓冲 + 栈）

扩展专题：一个 AI 应用场景（LLM 上下文管理 / 记忆系统）→ 多种数据结构协作
- ContextLRU        ：LRU，context window 的 token 淘汰（超 token 上限时淘汰最久未用）
- PromptTrie        ：Trie，prompt 前缀共享（多请求共享 system prompt，省 token）
- MemoryStore       ：哈希表，记忆检索（按 key O(1) 取记忆）
- DialogRingBuffer  ：环形缓冲，对话滑动窗口（固定大小，新消息覆盖最老的）
- CallStack         ：栈，agent 调用嵌套（后进先出匹配返回）

模拟完整对话：
  用户消息进来 → 滑动窗口（环形缓冲）→ 记忆检索（哈希）→
  prompt 前缀共享（Trie）→ context 淘汰（LRU）→ agent 调用嵌套（栈）

性能对比：
1. LRU 淘汰 vs 随机淘汰（命中率）
2. Trie 前缀共享 vs 无共享（省 token 数）
3. 环形缓冲 vs 列表（push 性能）

跑法：
    python 15_context_engineering/python/demo.py
"""

from __future__ import annotations

import random
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. ContextLRU：LRU，context window 的 token 淘汰
# ============================================================================


@dataclass
class ContextEntry:
    """context window 里的一条内容（一段对话/一个工具结果/一段记忆）。

    tokens: 这条内容占的 token 数
    last_used: 最近被命中（get/Touch）的逻辑时钟，越大越最近
    """

    key: str
    text: str
    tokens: int
    last_used: int = 0


class ContextLRU:
    """context window 的 token 预算淘汰，用 LRU。

    LLM 的 context window 有 token 上限（如 GPT-4 128K、Claude 200K）。
    往 context 里塞内容时，如果总 token 超过上限，必须淘汰一些旧内容。
    策略：淘汰最久未用的（LRU），保留近期还在用的。

    实现：OrderedDict 按访问顺序维护，head = 最久未用。
    - put(key, text, tokens): 超预算时从 head 往外淘汰，直到够
    - get(key): 命中则移到末尾（最近用），返回内容；未命中返回 None
    - touch(key): 更新 last_used 不取内容

    对比随机淘汰：随机淘汰不区分"近期还在用"和"早就过时"，
    命中率显著低于 LRU。
    """

    def __init__(self, token_budget: int) -> None:
        self._token_budget = token_budget
        self._data: OrderedDict[str, ContextEntry] = OrderedDict()
        self._total_tokens: int = 0
        self._clock: int = 0
        self._evicted: int = 0  # 累计淘汰数

    @property
    def token_budget(self) -> int:
        return self._token_budget

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def evicted(self) -> int:
        return self._evicted

    def put(self, key: str, text: str, tokens: int) -> int:
        """加入一条内容，超预算时按 LRU 淘汰。返回淘汰条数。"""
        if key in self._data:
            # 已存在：先删旧再加新（更新内容）
            old = self._data.pop(key)
            self._total_tokens -= old.tokens
        evicted_here = 0
        # 淘汰到能放下为止（也要防止单条就超预算）
        while self._total_tokens + tokens > self._token_budget and self._data:
            _, ev = self._data.popitem(last=False)  # head = 最久未用
            self._total_tokens -= ev.tokens
            self._evicted += 1
            evicted_here += 1
        self._clock += 1
        entry = ContextEntry(key=key, text=text, tokens=tokens,
                             last_used=self._clock)
        self._data[key] = entry
        self._total_tokens += tokens
        return evicted_here

    def get(self, key: str) -> str | None:
        """命中则移到末尾（最近用），返回内容；未命中返回 None。"""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        self._clock += 1
        self._data[key].last_used = self._clock
        return self._data[key].text

    def touch(self, key: str) -> None:
        """只更新最近用时，不取内容（如背景记忆被引用）。"""
        if key in self._data:
            self._data.move_to_end(key)
            self._clock += 1
            self._data[key].last_used = self._clock

    def keys_in_order(self) -> list[str]:
        """从最久未用到最近用。"""
        return list(self._data.keys())

    def snapshot(self) -> list[tuple[str, int]]:
        """返回 [(key, tokens), ...] 从最久未用到最近用。"""
        return [(k, e.tokens) for k, e in self._data.items()]


class RandomEvictionContext:
    """随机淘汰，对比基准。

    超预算时随机挑一条淘汰，不区分"近期还在用"。
    命中率显著低于 LRU。
    """

    def __init__(self, token_budget: int) -> None:
        self._token_budget = token_budget
        self._data: dict[str, ContextEntry] = {}
        self._total_tokens: int = 0
        self._evicted: int = 0
        self._rng = random.Random(0)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def evicted(self) -> int:
        return self._evicted

    def put(self, key: str, text: str, tokens: int) -> int:
        if key in self._data:
            old = self._data.pop(key)
            self._total_tokens -= old.tokens
        evicted_here = 0
        while self._total_tokens + tokens > self._token_budget and self._data:
            # 随机挑一个淘汰
            victim_key = self._rng.choice(list(self._data.keys()))
            ev = self._data.pop(victim_key)
            self._total_tokens -= ev.tokens
            self._evicted += 1
            evicted_here += 1
        self._data[key] = ContextEntry(key=key, text=text, tokens=tokens)
        self._total_tokens += tokens
        return evicted_here

    def get(self, key: str) -> str | None:
        return self._data[key].text if key in self._data else None


# ============================================================================
# 2. PromptTrie：Trie，prompt 前缀共享
# ============================================================================


class PromptTrie:
    """prompt 前缀共享，用 Trie。

    场景：多个请求的 prompt 都以同一段 system prompt 开头，例如：
      req_1: [sys, user_1, q_1]
      req_2: [sys, user_2, q_2]
      req_3: [sys, user_3, q_3]
    system prompt 可能几千 token，如果每个请求都独立存一份，浪费。
    用 Trie 把公共前缀合并，公共节点只存一次。

    - insert(tokens: list[str])：插入一个 prompt 的 token 序列
    - shared_nodes()：被 ≥2 个 prompt 共享的节点数（= 省下的 token 数）
    - total_nodes()：Trie 总节点数
    - independent_storage()：不共享时所需的总 token 数 = sum(len(prompt))

    对比无共享：每个 prompt 独立存，token 数 = sum(len(prompt))。
    Trie 共享后，实际 token 数 = total_nodes()，省下的 = shared_nodes()。
    """

    def __init__(self) -> None:
        # 每个节点：children dict + 经过该节点的 prompt 数
        self._children: list[dict[str, int]] = [{}]
        self._count: list[int] = [0]  # 各节点的经过 prompt 数
        self._n_prompts: int = 0
        self._independent_tokens: int = 0  # 不共享时的总 token 数

    @property
    def n_prompts(self) -> int:
        return self._n_prompts

    @property
    def independent_tokens(self) -> int:
        """无前缀共享时所需的总 token 数 = sum(len(prompt))。"""
        return self._independent_tokens

    def insert(self, tokens: list[str]) -> None:
        """插入一个 prompt 的 token 序列。"""
        self._n_prompts += 1
        self._independent_tokens += len(tokens)
        node = 0
        self._count[node] += 1
        for tok in tokens:
            children = self._children[node]
            if tok not in children:
                new_node = len(self._children)
                self._children.append({})
                self._count.append(0)
                children[tok] = new_node
            node = children[tok]
            self._count[node] += 1

    def total_nodes(self) -> int:
        """Trie 总节点数（不含根）= 共享后实际存的 token 数。"""
        return len(self._children) - 1

    def shared_nodes(self) -> int:
        """被 ≥2 个 prompt 共享的节点数 = 省下的 token 数。

        每个被 k 个 prompt 共享的节点，独立存储要 k 份，Trie 只存 1 份，
        省下 k-1 份。
        """
        saved = 0
        for c in self._count:
            if c >= 2:
                saved += c - 1
        return saved

    def shared_prefix_length(self, tokens: list[str]) -> int:
        """查询一个 prompt 与已有 prompt 的最长公共前缀长度。"""
        node = 0
        length = 0
        for tok in tokens:
            children = self._children[node]
            if tok not in children:
                break
            node = children[tok]
            length += 1
        return length


# ============================================================================
# 3. MemoryStore：哈希表，记忆检索
# ============================================================================


@dataclass
class MemoryItem:
    """一条长期记忆。"""

    key: str
    value: str
    tags: tuple[str, ...] = ()
    importance: float = 1.0  # 0-1，越大越重要


class MemoryStore:
    """记忆检索，用哈希表。

    场景：agent 把对话中提到的事实存成长期记忆（如"用户喜欢 Python"），
    后续对话按 key 快速取回，不用重新读全部历史。

    - put(key, value, tags, importance): O(1)
    - get(key): O(1) 哈希查找
    - get_by_tag(tag): O(n) 标签倒排（这里简化为线性扫描）
    - remove(key): O(1)

    对比线性扫描：把记忆存成 list，每次 get 都遍历整个 list，O(n)。
    记忆条数多时（如 10K 条），哈希表 O(1) 远快于 list O(n)。
    """

    def __init__(self) -> None:
        self._data: dict[str, MemoryItem] = {}
        self._tag_index: dict[str, set[str]] = {}

    def put(self, key: str, value: str,
            tags: tuple[str, ...] = (), importance: float = 1.0) -> None:
        if key in self._data:
            old = self._data[key]
            for t in old.tags:
                self._tag_index.get(t, set()).discard(key)
        item = MemoryItem(key=key, value=value, tags=tags,
                          importance=importance)
        self._data[key] = item
        for t in tags:
            self._tag_index.setdefault(t, set()).add(key)

    def get(self, key: str) -> str | None:
        item = self._data.get(key)
        return item.value if item is not None else None

    def has(self, key: str) -> bool:
        return key in self._data

    def get_by_tag(self, tag: str) -> list[MemoryItem]:
        keys = self._tag_index.get(tag, set())
        return [self._data[k] for k in keys if k in self._data]

    def remove(self, key: str) -> bool:
        if key not in self._data:
            return False
        old = self._data.pop(key)
        for t in old.tags:
            self._tag_index.get(t, set()).discard(key)
        return True

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[str]:
        return list(self._data.keys())


class LinearMemoryStore:
    """线性扫描记忆，对比基准。把记忆存成 list，get 遍历整个 list。"""

    def __init__(self) -> None:
        self._data: list[MemoryItem] = []

    def put(self, key: str, value: str,
            tags: tuple[str, ...] = (), importance: float = 1.0) -> None:
        # 先删同 key
        self._data = [m for m in self._data if m.key != key]
        self._data.append(MemoryItem(key=key, value=value, tags=tags,
                                     importance=importance))

    def get(self, key: str) -> str | None:
        for m in self._data:
            if m.key == key:
                return m.value
        return None

    def __len__(self) -> int:
        return len(self._data)


# ============================================================================
# 4. DialogRingBuffer：环形缓冲，对话滑动窗口
# ============================================================================


@dataclass
class Message:
    """一条对话消息。"""

    role: str  # "user" / "assistant" / "system" / "tool"
    content: str
    tokens: int
    msg_id: int = 0


class DialogRingBuffer:
    """对话滑动窗口，用环形缓冲。

    场景：LLM 对话只看最近 N 条消息（滑动窗口），老消息自动滚出。
    用环形缓冲（固定大小数组 + 写指针）实现：
    - push(msg): 写到 write_ptr 位置，write_ptr 前进一格（覆盖最老的）
    - get_all(): 按时间顺序返回当前窗口内所有消息

    优点：
    - push O(1)，不移动任何元素
    - 内存固定，不随消息数增长
    - 缓存友好（连续数组）

    对比 list + pop(0)：每次 push 都 pop(0) 是 O(n)（要把后面 n-1 个元素前移），
    消息多时慢得多。
    """

    def __init__(self, capacity: int) -> None:
        assert capacity > 0
        self._capacity = capacity
        self._buf: list[Message | None] = [None] * capacity
        self._write_ptr: int = 0  # 下一个写入位置
        self._size: int = 0  # 当前消息数（未满时 < capacity）
        self._total_tokens: int = 0
        self._msg_counter: int = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def push(self, role: str, content: str, tokens: int) -> Message:
        """写入一条消息，覆盖最老的（如果已满）。"""
        self._msg_counter += 1
        msg = Message(role=role, content=content, tokens=tokens,
                      msg_id=self._msg_counter)
        old = self._buf[self._write_ptr]
        if old is not None:
            self._total_tokens -= old.tokens
        else:
            self._size += 1
        self._buf[self._write_ptr] = msg
        self._total_tokens += tokens
        self._write_ptr = (self._write_ptr + 1) % self._capacity
        return msg

    def get_all(self) -> list[Message]:
        """按时间顺序返回当前窗口内所有消息（最老到最新）。"""
        if self._size < self._capacity:
            # 未满：从 0 到 write_ptr
            return [m for m in self._buf[:self._write_ptr] if m is not None]
        # 已满：从 write_ptr（最老）绕一圈到 write_ptr-1（最新）
        result: list[Message] = []
        for i in range(self._capacity):
            idx = (self._write_ptr + i) % self._capacity
            m = self._buf[idx]
            if m is not None:
                result.append(m)
        return result

    def latest(self, n: int = 1) -> list[Message]:
        """取最近 n 条消息。"""
        all_msgs = self.get_all()
        return all_msgs[-n:] if n < len(all_msgs) else all_msgs


class ListDialogWindow:
    """list + pop(0) 滑动窗口，对比基准。

    每次 push 超过容量就 pop(0)，pop(0) 是 O(n)（前移所有元素）。
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data: list[Message] = []
        self._msg_counter: int = 0

    @property
    def size(self) -> int:
        return len(self._data)

    def push(self, role: str, content: str, tokens: int) -> Message:
        self._msg_counter += 1
        msg = Message(role=role, content=content, tokens=tokens,
                      msg_id=self._msg_counter)
        self._data.append(msg)
        if len(self._data) > self._capacity:
            self._data.pop(0)  # O(n)
        return msg

    def get_all(self) -> list[Message]:
        return list(self._data)


# ============================================================================
# 5. CallStack：栈，agent 调用嵌套
# ============================================================================


@dataclass
class AgentCall:
    """一次 agent 调用。

    caller: 调用者 agent 名（顶层是 "root"）
    callee: 被调 agent 名
    task: 调用任务描述
    args_snapshot: 调用参数摘要（用于返回时恢复上下文）
    """

    caller: str
    callee: str
    task: str
    args_snapshot: str
    call_id: int = 0


class CallStack:
    """agent 调用嵌套，用栈。

    场景：agent 可以调用别的 agent（如主 agent 调用代码 agent、代码 agent
    又调用测试 agent），形成嵌套调用。返回时必须按"后进先出"匹配：
    最先返回的是最内层的 agent，最后返回的是最外层。

    - push(call): 入栈，开始一个新嵌套层
    - pop(): 出栈，返回到调用者（最内层先返回）
    - peek(): 看栈顶（当前最内层 agent）
    - depth(): 当前嵌套深度

    用栈保证返回顺序正确：最后调用的 agent 最先返回。
    错误做法是用队列：先调用的 agent 先返回，会乱套。

    也用于异常处理：内层 agent 抛异常时，栈逐层 pop 触发各层的 fallback。
    """

    def __init__(self) -> None:
        self._stack: list[AgentCall] = []
        self._call_counter: int = 0
        self._history: list[AgentCall] = []  # 完整调用历史

    def push(self, caller: str, callee: str, task: str,
             args_snapshot: str = "") -> AgentCall:
        self._call_counter += 1
        call = AgentCall(caller=caller, callee=callee, task=task,
                         args_snapshot=args_snapshot,
                         call_id=self._call_counter)
        self._stack.append(call)
        self._history.append(call)
        return call

    def pop(self) -> AgentCall | None:
        if not self._stack:
            return None
        return self._stack.pop()

    def peek(self) -> AgentCall | None:
        return self._stack[-1] if self._stack else None

    def depth(self) -> int:
        return len(self._stack)

    @property
    def history(self) -> list[AgentCall]:
        return list(self._history)

    def call_chain(self) -> list[str]:
        """当前嵌套链：从最外层到最内层。"""
        return [c.callee for c in self._stack]


class CallQueue:
    """队列做调用嵌套，对比基准（错误做法）。

    先调用的 agent 先返回，违反"后进先出"的嵌套语义。
    """

    def __init__(self) -> None:
        from collections import deque
        self._queue: deque[AgentCall] = deque()
        self._call_counter: int = 0

    def push(self, caller: str, callee: str, task: str,
             args_snapshot: str = "") -> AgentCall:
        self._call_counter += 1
        call = AgentCall(caller=caller, callee=callee, task=task,
                         args_snapshot=args_snapshot,
                         call_id=self._call_counter)
        self._queue.append(call)
        return call

    def pop(self) -> AgentCall | None:
        if not self._queue:
            return None
        return self._queue.popleft()  # 先进先出（错误！）

    def depth(self) -> int:
        return len(self._queue)


# ============================================================================
# 6. 完整对话模拟：5 种数据结构协作
# ============================================================================


def simulate_conversation(
    n_turns: int = 20,
    token_budget: int = 4096,
    window_capacity: int = 8,
    seed: int = 42,
) -> dict:
    """模拟完整 LLM 对话，5 种数据结构协作。

    流程（每个对话轮）：
    1. 用户消息进来 → 滑动窗口（环形缓冲）保留最近 N 条
    2. 记忆检索（哈希表）：按 user_id 取长期记忆
    3. prompt 前缀共享（Trie）：多轮共享 system prompt
    4. context 淘汰（LRU）：把窗口+记忆+system 拼成 context，超 token 上限按 LRU 淘汰
    5. agent 调用嵌套（栈）：主 agent 可能调用子 agent，子 agent 返回时按栈 pop

    Returns:
        包含统计信息的字典
    """
    rng = random.Random(seed)
    ring = DialogRingBuffer(capacity=window_capacity)
    memory = MemoryStore()
    trie = PromptTrie()
    ctx = ContextLRU(token_budget=token_budget)
    call_stack = CallStack()

    # 预置一些长期记忆
    memory.put("user_pref_lang", "Python",
               tags=("preference",), importance=0.9)
    memory.put("user_pref_editor", "VSCode",
               tags=("preference",), importance=0.7)
    memory.put("user_project", "data_structure_tutorial",
               tags=("context",), importance=0.8)
    memory.put("user_level", "beginner",
               tags=("context",), importance=0.8)

    system_prompt_tokens = ["system", "你", "是", "一", "个", "数", "据", "结",
                            "构", "助", "手", "，", "用", "简", "体", "中",
                            "文", "回", "答"]  # 19 token

    user_id = "user_0"
    n_evictions = 0
    n_memory_hits = 0
    n_sub_agent_calls = 0
    max_depth = 0
    token_saved_by_trie = 0

    for turn in range(n_turns):
        # 1. 用户消息 → 滑动窗口
        user_msg = f"第 {turn + 1} 个问题：解释一下 LRU 缓存"
        user_tokens = rng.randint(20, 80)
        ring.push("user", user_msg, user_tokens)

        # 2. 记忆检索（哈希表）
        for key in ["user_pref_lang", "user_level", "user_project"]:
            if memory.get(key) is not None:
                n_memory_hits += 1

        # 3. prompt 前缀共享（Trie）
        # 构造完整 prompt：system + 窗口消息
        prompt_tokens = list(system_prompt_tokens)
        for m in ring.get_all():
            prompt_tokens.append(f"[{m.role}]")
            prompt_tokens.append(m.content)
        trie.insert(prompt_tokens)
        # 这一轮省下的 token = 独立存储 - Trie 节点数（增量）
        # 简化：累计 shared_nodes
        token_saved_by_trie = trie.shared_nodes()

        # 4. context 淘汰（LRU）
        # 把每条消息作为一个 context entry
        for m in ring.get_all():
            ev = ctx.put(f"msg_{m.msg_id}", m.content, m.tokens)
            n_evictions += ev
        # system prompt 也放进去
        ctx.put("system_prompt", "你是一个数据结构助手", len(system_prompt_tokens))
        # 记忆也放进去
        for key in memory.keys():
            val = memory.get(key)
            if val is not None:
                ctx.put(f"mem_{key}", val, len(val))

        # 5. agent 调用嵌套（栈）
        call_stack.push("root", "main_agent", f"处理第 {turn + 1} 轮")
        # 30% 概率主 agent 调子 agent
        if rng.random() < 0.3:
            sub = "code_agent" if rng.random() < 0.5 else "search_agent"
            call_stack.push("main_agent", sub, "子任务")
            n_sub_agent_calls += 1
            # 10% 概率子 agent 再调孙 agent
            if rng.random() < 0.1:
                call_stack.push(sub, "tool_agent", "工具调用")
        max_depth = max(max_depth, call_stack.depth())
        # 返回：栈 pop 直到回到 root
        while call_stack.depth() > 0:
            call_stack.pop()

    return {
        "n_turns": n_turns,
        "token_budget": token_budget,
        "window_capacity": window_capacity,
        "n_evictions": n_evictions,
        "n_memory_hits": n_memory_hits,
        "n_sub_agent_calls": n_sub_agent_calls,
        "max_call_depth": max_depth,
        "token_saved_by_trie": token_saved_by_trie,
        "trie_total_nodes": trie.total_nodes(),
        "trie_independent_tokens": trie.independent_tokens,
        "final_context_tokens": ctx.total_tokens,
        "final_context_size": ctx.size,
        "memory_size": len(memory),
    }


# ============================================================================
# 7. 性能对比
# ============================================================================


def benchmark_lru_vs_random(
    n_ops: int = 4000, token_budget: int = 1024, seed: int = 42,
) -> dict:
    """对比：LRU 淘汰 vs 随机淘汰（命中率）。

    场景：4000 次 get 操作，token 预算 1024（让淘汰压力更大）。
    访问模式有强局部性：90% 访问最近 10% 的 key（热点）。
    - LRU：保留热点，命中率高
    - 随机：可能淘汰热点，命中率低
    """
    rng = random.Random(seed)
    keys = [f"key_{i}" for i in range(300)]
    # 每个 key 的 token 数 30-80（让预算压力更大）
    key_tokens = {k: rng.randint(30, 80) for k in keys}

    # 生成操作序列：95% 访问热点 key（前 5%），5% 访问冷 key
    # 热点 15 个，token 预算 1024 能放下，LRU 能保住热点
    # 冷 key 的 put 会冲击缓存，随机淘汰容易把热点挤出
    hot_keys = keys[:15]
    cold_keys = keys[15:]
    ops: list[tuple[str, str]] = []
    for _ in range(n_ops):
        r = rng.random()
        if r < 0.85:
            ops.append(("get", rng.choice(hot_keys)))
        elif r < 0.90:
            ops.append(("get", rng.choice(cold_keys)))
        else:
            # 10% 概率 put 冷 key，冲击缓存
            ops.append(("put", rng.choice(cold_keys)))

    # LRU
    lru = ContextLRU(token_budget=token_budget)
    lru_hits = 0
    lru_misses = 0
    for op, k in ops:
        if op == "put":
            lru.put(k, f"val_{k}", key_tokens[k])
        else:
            if lru.get(k) is not None:
                lru_hits += 1
            else:
                lru_misses += 1
                # miss 后 put 进去
                lru.put(k, f"val_{k}", key_tokens[k])

    # 随机淘汰
    rnd = RandomEvictionContext(token_budget=token_budget)
    rnd_hits = 0
    rnd_misses = 0
    for op, k in ops:
        if op == "put":
            rnd.put(k, f"val_{k}", key_tokens[k])
        else:
            if rnd.get(k) is not None:
                rnd_hits += 1
            else:
                rnd_misses += 1
                rnd.put(k, f"val_{k}", key_tokens[k])

    lru_total = lru_hits + lru_misses
    rnd_total = rnd_hits + rnd_misses
    return {
        "n_ops": n_ops,
        "token_budget": token_budget,
        "lru_hits": lru_hits,
        "lru_misses": lru_misses,
        "lru_hit_rate": lru_hits / lru_total if lru_total > 0 else 0.0,
        "random_hits": rnd_hits,
        "random_misses": rnd_misses,
        "random_hit_rate": rnd_hits / rnd_total if rnd_total > 0 else 0.0,
        "lru_evicted": lru.evicted,
        "random_evicted": rnd.evicted,
    }


def benchmark_trie_vs_no_share(
    n_requests: int = 100, sys_prompt_len: int = 500,
    user_msg_len: int = 50, seed: int = 42,
) -> dict:
    """对比：Trie 前缀共享 vs 无共享（省 token 数）。

    场景：100 个请求，每个 prompt = 500 token system prompt + 50 token 用户消息。
    - 无共享：每个请求独立存，总 token = 100 * 550 = 55000
    - Trie 共享：system prompt 只存一次，总 token = 500 + 100*50 = 5500
    """
    rng = random.Random(seed)
    trie = PromptTrie()
    sys_tokens = [f"sys_{i}" for i in range(sys_prompt_len)]

    for i in range(n_requests):
        user_tokens = [f"user_{i}_{j}" for j in range(user_msg_len)]
        trie.insert(sys_tokens + user_tokens)

    independent = trie.independent_tokens  # 无共享时的总 token
    shared = trie.total_nodes()  # Trie 共享后的总 token
    saved = trie.shared_nodes()  # 省下的 token
    return {
        "n_requests": n_requests,
        "sys_prompt_len": sys_prompt_len,
        "user_msg_len": user_msg_len,
        "independent_tokens": independent,
        "shared_tokens": shared,
        "saved_tokens": saved,
        "save_ratio": saved / independent if independent > 0 else 0.0,
    }


def benchmark_ring_vs_list(
    n_pushes: int = 100000, capacity: int = 5000,
) -> dict:
    """对比：环形缓冲 vs list+pop(0)（push 性能）。

    场景：100000 次 push，容量 5000（满了就覆盖最老的）。
    - 环形缓冲：push O(1)，与容量无关
    - list+pop(0)：push 满后每次 pop(0) 是 O(n)，n=5000 时 memmove 代价显著
    """
    ring = DialogRingBuffer(capacity=capacity)
    t_ring = Timer()
    t_ring.start()
    for i in range(n_pushes):
        ring.push("user", f"msg_{i}", 10)
    t_ring.stop()

    lst = ListDialogWindow(capacity=capacity)
    t_list = Timer()
    t_list.start()
    for i in range(n_pushes):
        lst.push("user", f"msg_{i}", 10)
    t_list.stop()

    return {
        "n_pushes": n_pushes,
        "capacity": capacity,
        "ring_ms": t_ring.elapsed_ms,
        "list_ms": t_list.elapsed_ms,
        "speedup": t_list.elapsed_ms / t_ring.elapsed_ms
        if t_ring.elapsed_ms > 0 else float("inf"),
    }


def benchmark_memory_hash_vs_linear(n_memories: int = 5000,
                                     n_queries: int = 5000,
                                     seed: int = 42) -> dict:
    """对比：哈希表记忆检索 vs 线性扫描（get 性能）。

    场景：5000 条记忆，5000 次 get 查询。
    - 哈希表：每次 get O(1)
    - 线性扫描：每次 get O(n)
    """
    rng = random.Random(seed)
    keys = [f"mem_{i}" for i in range(n_memories)]

    hash_store = MemoryStore()
    linear_store = LinearMemoryStore()
    for k in keys:
        hash_store.put(k, f"val_{k}")
        linear_store.put(k, f"val_{k}")

    query_keys = [rng.choice(keys) for _ in range(n_queries)]

    t_hash = Timer()
    t_hash.start()
    for k in query_keys:
        hash_store.get(k)
    t_hash.stop()

    t_linear = Timer()
    t_linear.start()
    for k in query_keys:
        linear_store.get(k)
    t_linear.stop()

    return {
        "n_memories": n_memories,
        "n_queries": n_queries,
        "hash_ms": t_hash.elapsed_ms,
        "linear_ms": t_linear.elapsed_ms,
        "speedup": t_linear.elapsed_ms / t_hash.elapsed_ms
        if t_hash.elapsed_ms > 0 else float("inf"),
    }


# ============================================================================
# 8. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 15 章 demo：上下文工程与记忆 × 数据结构协作")
    print("（LRU + Trie + 哈希 + 环形缓冲 + 栈）")
    print("=" * 72)

    print("\n[1] context window 淘汰：LRU（token 预算淘汰）")
    print("-" * 72)
    ctx = ContextLRU(token_budget=500)
    print(f"  token 预算 = {ctx.token_budget}")
    items = [("sys", 100), ("msg_1", 80), ("msg_2", 90),
             ("msg_3", 110), ("tool_1", 130)]
    for k, t in items:
        ev = ctx.put(k, f"content_{k}", t)
        print(f"  put({k}, tokens={t}): 淘汰 {ev} 条, "
              f"总 token = {ctx.total_tokens}/{ctx.token_budget}")
    print(f"  当前 context（从最久未用到最近用）：{ctx.snapshot()}")
    # 访问 msg_2 让它变最近用
    ctx.get("msg_2")
    print(f"  get(msg_2) 后顺序：{ctx.snapshot()}")
    # 再加一条超预算的，触发淘汰
    ev = ctx.put("big", "x" * 200, 200)
    print(f"  put(big, tokens=200): 淘汰 {ev} 条（最久未用的先被淘汰）, "
          f"总 token = {ctx.total_tokens}")
    print(f"  → LRU 保留近期还在用的，淘汰最久未用的")

    print("\n[2] prompt 前缀共享：Trie（多请求共享 system prompt）")
    print("-" * 72)
    trie = PromptTrie()
    sys_prompt = ["你", "是", "一", "个", "助", "手"]
    for i in range(3):
        user_msg = [f"u{i}_1", f"u{i}_2"]
        trie.insert(sys_prompt + user_msg)
    print(f"  3 个请求，每个 prompt = 6 token system + 2 token 用户")
    print(f"  无共享总 token = {trie.independent_tokens}")
    print(f"  Trie 共享后总 token = {trie.total_nodes()}")
    print(f"  省下 token = {trie.shared_nodes()} "
          f"({trie.shared_nodes() / trie.independent_tokens * 100:.1f}%)")
    print(f"  → system prompt 只存一次，多请求共享前缀")

    print("\n[3] 记忆检索：哈希表（按 key O(1) 取记忆）")
    print("-" * 72)
    mem = MemoryStore()
    mem.put("user_lang", "Python", tags=("preference",), importance=0.9)
    mem.put("user_level", "beginner", tags=("context",), importance=0.8)
    mem.put("project", "tutorial", tags=("context",), importance=0.7)
    print(f"  存了 {len(mem)} 条记忆：{mem.keys()}")
    print(f"  get('user_lang') = {mem.get('user_lang')}")
    print(f"  get('user_level') = {mem.get('user_level')}")
    print(f"  按标签 'context' 查：{[m.key for m in mem.get_by_tag('context')]}")
    print(f"  → 哈希表 O(1) 查找，长期记忆按 key 秒取")

    print("\n[4] 对话滑动窗口：环形缓冲（固定大小，新覆盖老）")
    print("-" * 72)
    ring = DialogRingBuffer(capacity=4)
    print(f"  容量 = {ring.capacity}")
    for i in range(6):
        ring.push("user", f"msg_{i}", 10)
        msgs = [m.content for m in ring.get_all()]
        print(f"  push(msg_{i}): 窗口 = {msgs}（size={ring.size}）")
    print(f"  最近 2 条：{[m.content for m in ring.latest(2)]}")
    print(f"  → 环形缓冲 O(1) push，固定内存，老消息自动滚出")

    print("\n[5] agent 调用嵌套：栈（后进先出匹配返回）")
    print("-" * 72)
    cs = CallStack()
    cs.push("root", "main_agent", "处理用户问题")
    print(f"  调用链：{cs.call_chain()}（深度 {cs.depth()}）")
    cs.push("main_agent", "code_agent", "写代码")
    print(f"  调用链：{cs.call_chain()}（深度 {cs.depth()}）")
    cs.push("code_agent", "test_agent", "跑测试")
    print(f"  调用链：{cs.call_chain()}（深度 {cs.depth()}）")
    print(f"  返回顺序（栈 pop，后进先出）：")
    while cs.depth() > 0:
        c = cs.pop()
        assert c is not None
        print(f"    {c.callee} 返回到 {c.caller}")
    print(f"  → 栈保证最内层 agent 先返回，匹配嵌套语义")

    print("\n[6] 完整对话模拟：5 种数据结构协作")
    print("-" * 72)
    result = simulate_conversation(
        n_turns=20, token_budget=4096, window_capacity=8, seed=42,
    )
    print(f"  配置：20 轮对话，token 预算 4096，窗口 8 条")
    print(f"  结果：")
    print(f"    context 淘汰次数 = {result['n_evictions']}")
    print(f"    记忆命中次数 = {result['n_memory_hits']}")
    print(f"    子 agent 调用次数 = {result['n_sub_agent_calls']}")
    print(f"    最大调用深度 = {result['max_call_depth']}")
    print(f"    Trie 省下 token = {result['token_saved_by_trie']}")
    print(f"    Trie 节点数 = {result['trie_total_nodes']}, "
          f"无共享时 = {result['trie_independent_tokens']}")
    print(f"    最终 context: {result['final_context_size']} 条, "
          f"{result['final_context_tokens']} token")
    print(f"  → 环形缓冲 → 哈希记忆 → Trie 共享 → LRU 淘汰 → 栈嵌套")

    print("\n[7] 性能对比：LRU 淘汰 vs 随机淘汰（命中率）")
    print("-" * 72)
    res_lru = benchmark_lru_vs_random(n_ops=4000, token_budget=1024, seed=42)
    print(f"  场景：{res_lru['n_ops']} 次操作，token 预算 "
          f"{res_lru['token_budget']}")
    print(f"  LRU 淘汰  ：命中 {res_lru['lru_hits']}, "
          f"未命中 {res_lru['lru_misses']}, "
          f"命中率 {res_lru['lru_hit_rate'] * 100:.1f}%")
    print(f"  随机淘汰  ：命中 {res_lru['random_hits']}, "
          f"未命中 {res_lru['random_misses']}, "
          f"命中率 {res_lru['random_hit_rate'] * 100:.1f}%")
    if res_lru["random_hit_rate"] > 0:
        ratio = res_lru["lru_hit_rate"] / res_lru["random_hit_rate"]
        print(f"  → LRU 命中率是随机的 {ratio:.2f}x（保留热点）")

    print("\n[8] 性能对比：Trie 前缀共享 vs 无共享（省 token）")
    print("-" * 72)
    res_trie = benchmark_trie_vs_no_share(
        n_requests=100, sys_prompt_len=500, user_msg_len=50, seed=42,
    )
    print(f"  场景：{res_trie['n_requests']} 请求，"
          f"system prompt {res_trie['sys_prompt_len']} token, "
          f"用户消息 {res_trie['user_msg_len']} token")
    print(f"  无共享总 token = {res_trie['independent_tokens']}")
    print(f"  Trie 共享后   = {res_trie['shared_tokens']}")
    print(f"  省下 token    = {res_trie['saved_tokens']} "
          f"({res_trie['save_ratio'] * 100:.1f}%)")
    print(f"  → system prompt 只存一次，省 90%+ token")

    print("\n[9] 性能对比：环形缓冲 vs list+pop(0)（push 性能）")
    print("-" * 72)
    res_ring = benchmark_ring_vs_list(n_pushes=100000, capacity=5000)
    print(f"  场景：{res_ring['n_pushes']} 次 push，容量 {res_ring['capacity']}")
    print(f"  环形缓冲    ：{res_ring['ring_ms']:.2f} ms（push O(1)）")
    print(f"  list+pop(0) ：{res_ring['list_ms']:.2f} ms（push 满 O(n)）")
    print(f"  → 环形缓冲快 {res_ring['speedup']:.1f}x")

    print("\n[10] 性能对比：哈希记忆 vs 线性扫描（get 性能）")
    print("-" * 72)
    res_mem = benchmark_memory_hash_vs_linear(
        n_memories=5000, n_queries=5000, seed=42,
    )
    print(f"  场景：{res_mem['n_memories']} 条记忆，"
          f"{res_mem['n_queries']} 次 get 查询")
    print(f"  哈希表    ：{res_mem['hash_ms']:.2f} ms（每次 O(1)）")
    print(f"  线性扫描  ：{res_mem['linear_ms']:.2f} ms（每次 O(n)）")
    print(f"  → 哈希表快 {res_mem['speedup']:.1f}x")

    print("\n[11] 保存对比图到 figures/")
    print("-" * 72)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    # 图 1：LRU vs 随机 命中率
    save_bar(
        {
            "LRU 淘汰": res_lru["lru_hit_rate"] * 100,
            "随机淘汰": res_lru["random_hit_rate"] * 100,
        },
        fig_dir / "hit_rate_lru_vs_random.png",
        title=f"context window 命中率（{res_lru['n_ops']} 次操作）",
        ylabel="命中率 (%)",
        baseline="随机淘汰",
    )
    print(f"  ✓ 保存 figures/hit_rate_lru_vs_random.png")

    # 图 2：Trie 共享省 token
    save_bar(
        {
            "无共享": float(res_trie["independent_tokens"]),
            "Trie 共享": float(res_trie["shared_tokens"]),
        },
        fig_dir / "token_trie_vs_no_share.png",
        title=f"prompt 前缀共享省 token（{res_trie['n_requests']} 请求）",
        ylabel="总 token 数",
        baseline="无共享",
    )
    print(f"  ✓ 保存 figures/token_trie_vs_no_share.png")

    # 图 3：环形缓冲 vs list push 性能
    save_bar(
        {
            "环形缓冲": res_ring["ring_ms"],
            "list+pop(0)": res_ring["list_ms"],
        },
        fig_dir / "push_ring_vs_list.png",
        title=f"滑动窗口 push 性能（{res_ring['n_pushes']} 次 push）",
        ylabel="耗时 (ms)",
        baseline="list+pop(0)",
    )
    print(f"  ✓ 保存 figures/push_ring_vs_list.png")

    # 图 4：哈希 vs 线性 get 性能
    save_bar(
        {
            "哈希表": res_mem["hash_ms"],
            "线性扫描": res_mem["linear_ms"],
        },
        fig_dir / "get_hash_vs_linear.png",
        title=f"记忆检索 get 性能（{res_mem['n_queries']} 次查询）",
        ylabel="耗时 (ms)",
        baseline="线性扫描",
    )
    print(f"  ✓ 保存 figures/get_hash_vs_linear.png")

    # 图 5：随对话轮数 token 节省曲线
    turns = list(range(1, 21))
    saved_curve: list[float] = []
    for n in turns:
        r = simulate_conversation(n_turns=n, token_budget=4096,
                                  window_capacity=8, seed=42)
        saved_curve.append(float(r["token_saved_by_trie"]))
    save_line(
        {"Trie 省下 token": saved_curve},
        fig_dir / "token_saved_over_turns.png",
        title="Trie 前缀共享省 token 随对话轮数增长",
        ylabel="省下 token 数",
        xlabel="对话轮数",
    )
    print(f"  ✓ 保存 figures/token_saved_over_turns.png")

    print("\n" + "=" * 72)
    print("结论：LLM 上下文工程是 5 种数据结构的协奏——")
    print("  LRU        → context window 淘汰（保留热点，token 不超限）")
    print("  Trie       → prompt 前缀共享（多请求共享 system prompt）")
    print("  哈希表     → 记忆检索（按 key O(1) 取长期记忆）")
    print("  环形缓冲   → 对话滑动窗口（固定大小，O(1) push）")
    print("  栈         → agent 调用嵌套（后进先出匹配返回）")
    print("生产级系统 LangChain Memory、MemGPT、Letta 用类似机制：")
    print("  滑动窗口 + 摘要记忆 + 向量检索 + 优先级淘汰。")
    print("本质都是这 5 种数据结构的不同组合。")
    print("=" * 72)


if __name__ == "__main__":
    main()