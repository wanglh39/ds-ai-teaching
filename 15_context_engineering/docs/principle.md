# 上下文工程与记忆的数据结构原理：LRU + Trie + 哈希 + 环形缓冲 + 栈

> 本文档讲清「LLM 上下文管理为什么需要 5 种数据结构协作」。AI 应用（context window 管理、prompt 工程、记忆系统、agent 嵌套）见 `ai_application.md`。前 12 章是「一个数据结构 → 一个 AI 优化」，本章是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。

## 0. 为什么是 5 种数据结构

LLM 上下文工程（Context Engineering）的核心问题是：**在有限的 context window 里，塞什么、不塞什么、怎么塞、按什么顺序塞**。这一个问题分解出 5 个子问题，每个子问题对应一种数据结构：

| 子问题 | 数据结构 | 作用 |
|---|---|---|
| token 超限时淘汰谁？ | **LRU** | context window 淘汰（保留近期还在用的） |
| 多请求共享 system prompt 怎么省 token？ | **Trie** | prompt 前缀共享（公共前缀只存一次） |
| 按 key 取记忆怎么快？ | **哈希表** | 记忆检索 O(1) |
| 对话只看最近 N 条怎么实现？ | **环形缓冲** | 滑动窗口 O(1) push，固定内存 |
| agent 调 agent 怎么匹配返回？ | **栈** | 后进先出匹配嵌套返回 |

这 5 种数据结构不是独立工作，而是**协奏**：用户消息进来 → 环形缓冲保留最近 N 条 → 哈希表按 key 取长期记忆 → Trie 把 system prompt 前缀共享 → LRU 把窗口+记忆+system 拼成 context 并按 token 预算淘汰 → 栈管理 agent 嵌套调用。

## 1. LRU 与 context window 淘汰

### 1.1 context window 的 token 上限

LLM 的 context window 有 token 上限：

| 模型 | context window |
|---|---|
| GPT-3.5 | 16K |
| GPT-4 | 8K / 32K / 128K |
| Claude 3 | 200K |
| Llama 3 | 8K / 128K |

往 context 里塞内容时，总 token 不能超过上限。超了必须淘汰一些旧内容。问题是：**淘汰谁**？

### 1.2 为什么不能用 FIFO

FIFO（先入先出）淘汰最早进入 context 的内容。问题：最早进入的不一定是最没用的。

```
对话场景：
  时刻 1: system prompt（1000 token）—— 全程都要用
  时刻 2: 用户第一条消息（100 token）—— 早过时了
  时刻 3: 用户第二条消息（100 token）—— 早过时了
  ...
  时刻 20: 用户第十九条消息（100 token）

FIFO 淘汰：先把 system prompt 淘汰 → 后续每轮都要重新塞 system prompt
LRU 淘汰：system prompt 每轮都被引用（最近用），永远不会被淘汰
```

system prompt 是每轮都要用的，FIFO 会把它淘汰掉，导致每轮重新塞，浪费 token。LRU 因为 system prompt 每轮都被访问（`last_used` 不断更新），永远不会被淘汰。

### 1.3 LRU：最近最少使用

**LRU**（Least Recently Used）淘汰策略：当容量超限时，淘汰最久未访问的元素。

实现用 `OrderedDict`：按访问顺序维护，head 是最久未用，tail 是最近用。

```python
class ContextLRU:
    def __init__(self, token_budget: int):
        self._data: OrderedDict[str, ContextEntry] = OrderedDict()
        self._total_tokens: int = 0

    def put(self, key, text, tokens):
        # 超预算时从 head 往外淘汰
        while self._total_tokens + tokens > self._token_budget and self._data:
            _, ev = self._data.popitem(last=False)  # head = 最久未用
            self._total_tokens -= ev.tokens
        self._data[key] = ContextEntry(...)
        self._total_tokens += tokens

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)  # 命中后移到末尾（最近用）
        return self._data[key].text
```

关键：`get` 命中后调用 `move_to_end(key)`，把元素移到末尾（最近用），这样 head 永远是最久未用的。

### 1.4 token 预算淘汰 vs 条目数淘汰

普通 LRU 淘汰是按**条目数**（如最多 100 条），context window 的 LRU 是按 **token 总数**（如最多 128K token）。区别：

- 条目数 LRU：每条大小一样，`while len > capacity: pop`
- token 预算 LRU：每条大小不同（一条消息可能 10 token 也可能 1000 token），`while total_tokens + new_tokens > budget: pop`

token 预算 LRU 更复杂：单条可能就超预算（如一条 200K token 的消息超过 128K 窗口），要处理这种边界。

### 1.5 复杂度与命中率

| 操作 | LRU | 随机淘汰 |
|---|---|---|
| put | O(1) 均摊 | O(1) |
| get 命中 | O(1) | O(1) |
| 命中率（有局部性） | 高 | 低 |

LRU 的优势在**命中率**：当访问有局部性（80% 访问 20% 热点）时，LRU 保留热点，命中率高；随机淘汰可能淘汰热点，命中率低。实测：4000 次操作，85% 访问 5% 热点，LRU 命中率 82.5%，随机 71.4%，LRU 高 11 个百分点。

## 2. Trie 与 prompt 前缀共享

### 2.1 多请求共享 system prompt

LLM 应用的请求模式：多个请求的 prompt 都以同一段 system prompt 开头：

```
req_1: [system_prompt] + [user_1_msg] + [q_1]
req_2: [system_prompt] + [user_2_msg] + [q_2]
req_3: [system_prompt] + [user_3_msg] + [q_3]
```

system prompt 可能几千 token（角色设定、工具说明、few-shot 示例）。如果每个请求独立存一份，浪费。用 **Trie** 把公共前缀合并，公共节点只存一次。

### 2.2 Trie：前缀树

**Trie**（前缀树）是一种树，每条边标一个字符（或 token），从根到某节点的路径拼成一个字符串（或 token 序列）。公共前缀共享同一条路径。

```python
class PromptTrie:
    def __init__(self):
        self._children: list[dict[str, int]] = [{}]  # 节点 -> children
        self._count: list[int] = [0]  # 各节点的经过 prompt 数

    def insert(self, tokens: list[str]):
        node = 0
        for tok in tokens:
            children = self._children[node]
            if tok not in children:
                new_node = len(self._children)
                self._children.append({})
                self._count.append(0)
                children[tok] = new_node
            node = children[tok]
            self._count[node] += 1
```

`_count[node]` 记录有多少个 prompt 经过该节点。`_count >= 2` 的节点被多个 prompt 共享，省下的 token 数 = `sum(count - 1 for count in _count if count >= 2)`。

### 2.3 省多少 token

假设 N 个请求，每个 prompt = S token system + U token 用户消息：

- 无共享：总 token = N × (S + U)
- Trie 共享：总 token = S + N × U（system 只存一次）

当 S >> U（system prompt 500 token，用户消息 50 token），省的比例 = (N-1)×S / (N×(S+U)) ≈ S/(S+U) ≈ 90%。

实测：100 请求，system 500 token + 用户 50 token，无共享 55000 token，Trie 共享 5500 token，**省 90.2%**。

### 2.4 Trie 在 prefix caching 中的应用

vLLM、SGLang 等推理框架的 **prefix caching**：多个请求共享 system prompt 前缀时，前缀的 KV cache 只算一次。底层就是 Trie 思想——按 token 序列建前缀树，公共前缀的 KV cache 共享。这让多轮对话、多请求 batch 的 prefill 成本大幅下降。

## 3. 哈希表与记忆检索

### 3.1 短期记忆 vs 长期记忆

LLM 的记忆分两层：

- **短期记忆**：当前对话的最近几轮消息，在 context window 里（环形缓冲管理）
- **长期记忆**：跨对话的事实（如"用户喜欢 Python"），存在外部存储，按需检索

长期记忆的检索方式：

| 检索方式 | 数据结构 | 复杂度 | 适用 |
|---|---|---|---|
| 按 key 精确取 | **哈希表** | O(1) | "用户偏好语言" → "Python" |
| 按语义相似度 | 向量索引（HNSW） | O(log n) | "用户喜欢什么" → 相似记忆 |
| 按标签过滤 | 倒排索引 | O(1) 找标签 | 所有 "preference" 标签的记忆 |

本章聚焦**按 key 精确取**的哈希表。语义检索见第 12 章（HNSW）。

### 3.2 哈希表：O(1) 查找

**哈希表**（Hash Table）把 key 经哈希函数映射到桶，每个桶存 (key, value)。查找时算 hash(key) 找到桶，再在桶里比对 key。

```python
class MemoryStore:
    def __init__(self):
        self._data: dict[str, MemoryItem] = {}
        self._tag_index: dict[str, set[str]] = {}  # 标签倒排

    def put(self, key, value, tags, importance):
        self._data[key] = MemoryItem(key, value, tags, importance)
        for t in tags:
            self._tag_index.setdefault(t, set()).add(key)

    def get(self, key):
        item = self._data.get(key)  # O(1)
        return item.value if item else None
```

`_tag_index` 是标签倒排索引：tag → set of keys，让 `get_by_tag(tag)` 也是 O(1) 找到相关 key 集合。

### 3.3 哈希表 vs 线性扫描

| 操作 | 哈希表 | 线性扫描 |
|---|---|---|
| put | O(1) | O(1) append |
| get | O(1) | O(n) 遍历 |
| remove | O(1) | O(n) 找到再删 |

记忆条数多时（如 10K 条），哈希表 O(1) 远快于线性扫描 O(n)。实测：5000 条记忆，5000 次 get，哈希表 1.74 ms，线性扫描 470 ms，**快 270x**。

## 4. 环形缓冲与对话滑动窗口

### 4.1 对话只看最近 N 条

LLM 对话通常只看最近 N 条消息（滑动窗口），老消息自动滚出。原因：

- **token 限制**：context window 装不下整个对话历史
- **相关性**：老消息通常和当前问题无关
- **成本**：token 越多推理越慢越贵

### 4.2 环形缓冲：固定大小数组 + 写指针

**环形缓冲**（Ring Buffer）用一个固定大小的数组 + 一个写指针实现滑动窗口：

```python
class DialogRingBuffer:
    def __init__(self, capacity: int):
        self._buf: list[Message | None] = [None] * capacity
        self._write_ptr: int = 0  # 下一个写入位置

    def push(self, role, content, tokens):
        # 写到 write_ptr 位置，覆盖最老的
        self._buf[self._write_ptr] = Message(...)
        self._write_ptr = (self._write_ptr + 1) % self._capacity

    def get_all(self):
        # 从最老到最新绕一圈
        ...
```

`push` 只写一个位置然后指针前移一格，O(1)，不移动任何元素。满了之后新消息覆盖最老的（write_ptr 转一圈回到最老的位置）。

### 4.3 环形缓冲 vs list + pop(0)

| 操作 | 环形缓冲 | list + pop(0) |
|---|---|---|
| push | O(1) | O(n)（满了 pop(0) 前移所有元素） |
| 内存 | 固定 | 固定（但每次 pop 前移） |
| 缓存友好 | 是（连续数组） | 是 |

`list.pop(0)` 要把后面 n-1 个元素前移一格，O(n)。环形缓冲只改指针，O(1)。实测：100000 次 push，容量 5000，环形缓冲 132 ms，list+pop(0) 1881 ms，**快 14x**。

### 4.4 环形缓冲在流式场景的优势

环形缓冲特别适合**流式写入**场景：消息不断到达，只保留最近 N 条。如：

- 聊天对话：保留最近 20 轮
- 日志监控：保留最近 1000 条日志
- 传感器数据：保留最近 N 个采样

写指针绕圈转，无需分配新内存，无需移动元素，对实时系统友好。

## 5. 栈与 agent 调用嵌套

### 5.1 agent 调 agent 的嵌套

现代 AI agent 可以调用别的 agent，形成嵌套：

```
main_agent 接到用户问题
  → 调用 code_agent 写代码
    → code_agent 调用 test_agent 跑测试
    ← test_agent 返回测试结果给 code_agent
  ← code_agent 返回代码给 main_agent
main_agent 继续处理
```

返回时必须按**后进先出**：最后调用的 agent 最先返回（test_agent 先返回给 code_agent，code_agent 再返回给 main_agent）。

### 5.2 栈：后进先出

**栈**（Stack）是后进先出（LIFO）结构：

```python
class CallStack:
    def __init__(self):
        self._stack: list[AgentCall] = []

    def push(self, caller, callee, task):
        self._stack.append(AgentCall(caller, callee, task))

    def pop(self):
        return self._stack.pop()  # 最后入栈的先出

    def call_chain(self):
        return [c.callee for c in self._stack]  # 从外到内
```

`push` 入栈开始新嵌套层，`pop` 出栈返回到调用者。栈顶永远是当前最内层的 agent。

### 5.3 为什么不能用队列

如果用队列（FIFO）管理调用嵌套，先调用的 agent 先返回：

```
错误（队列）：
  push main_agent → push code_agent → push test_agent
  pop → main_agent 先返回？？但 main_agent 还在等 code_agent！
```

这违反嵌套语义。嵌套调用的返回顺序必然是后进先出，只有栈正确。

### 5.4 栈在异常处理中的作用

agent 嵌套调用中，内层 agent 抛异常时，栈逐层 `pop` 触发各层的 fallback：

```
test_agent 抛异常
  → pop test_agent，触发 code_agent 的 fallback（如返回"测试失败"）
  → code_agent 决定是否继续
  → 如果 code_agent 也抛异常，pop，触发 main_agent 的 fallback
```

这就是编程语言异常处理的栈展开（stack unwinding）机制，agent 系统复用了这个思路。

## 6. 5 种数据结构如何协作

### 6.1 完整对话流程

一个对话轮的完整流程，5 种数据结构协作：

```
用户消息到达
  │
  ├─→ [环形缓冲] push 消息，保留最近 N 条（滑动窗口）
  │
  ├─→ [哈希表] 按 user_id 取长期记忆（如"用户偏好 Python"）
  │
  ├─→ [Trie] 把 system prompt + 窗口消息 + 记忆拼成 prompt 序列
  │         多轮共享 system prompt 前缀，省 token
  │
  ├─→ [LRU] 把 prompt 序列塞进 context window
  │         超 token 预算时淘汰最久未用的（保留热点）
  │
  └─→ [栈] 主 agent 处理，可能调子 agent（push）
            子 agent 返回时 pop，按后进先出匹配
```

### 6.2 各结构各司其职

| 阶段 | 数据结构 | 解决的问题 |
|---|---|---|
| 消息到达 | 环形缓冲 | 保留最近 N 条，老的自动滚出 |
| 取记忆 | 哈希表 | 按 key O(1) 取长期记忆 |
| 拼 prompt | Trie | system prompt 前缀共享，省 token |
| 塞 context | LRU | 超 token 预算时淘汰最久未用的 |
| agent 嵌套 | 栈 | 后进先出匹配嵌套返回 |

### 6.3 为什么不能只用一种

有人可能想：只用一个 list 存所有东西不行吗？不行，因为每个子问题的需求不同：

- **list 不能 O(1) 淘汰最久未用**（LRU 的核心）
- **list 不能 O(1) 按前缀共享**（Trie 的核心）
- **list 不能 O(1) 按 key 查找**（哈希表的核心）
- **list 的 pop(0) 是 O(n)**（环形缓冲的核心优势）
- **list 的 pop() 是 O(1) 但语义是末尾**（栈正好用 list 末尾实现，但概念上是栈）

5 种数据结构各有不可替代的优势，组合起来才能高效解决上下文工程的全链路问题。

## 7. 生产级系统的近似实现

### 7.1 LangChain Memory

LangChain 的 Memory 模块用类似机制：

- `ConversationBufferWindowMemory`：环形缓冲（保留最近 K 轮）
- `ConversationSummaryMemory`：摘要记忆（老消息摘要成一段）
- `ConversationKGMemory`：知识图谱记忆（三元组存储，按实体检索）
- `VectorStoreRetrieverMemory`：向量记忆（embedding + 相似度检索）

### 7.2 MemGPT / Letta

MemGPT（现 Letta）把操作系统的内存管理思想搬到 LLM：

- **主 context**（像寄存器）：当前对话窗口，LRU 淘汰
- **记忆上下文**（像内存）：存不下的内容，按需 page in
- **归档记忆**（像硬盘）：长期存储，向量检索

LRU 决定什么留在主 context、什么 page out 到记忆、什么归档。

### 7.3 vLLM prefix caching

vLLM 的 prefix caching 用 Trie 思想：按 token 序列建前缀树，公共前缀的 KV cache 共享。多轮对话、多请求 batch 的 prefill 成本大幅下降。

## 8. 小结

上下文工程是 5 种数据结构的协奏：

- **LRU**：context window 淘汰，保留热点，token 不超限
- **Trie**：prompt 前缀共享，多请求共享 system prompt，省 90%+ token
- **哈希表**：记忆检索，按 key O(1) 取长期记忆
- **环形缓冲**：对话滑动窗口，O(1) push，固定内存
- **栈**：agent 调用嵌套，后进先出匹配返回

每种结构解决一个子问题，组合起来高效解决上下文工程的全链路。生产级系统（LangChain Memory、MemGPT、vLLM prefix caching）用不同组合实现类似机制。