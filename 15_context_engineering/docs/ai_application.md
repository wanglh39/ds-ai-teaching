# 上下文工程与记忆的 AI 应用：LLM context window 管理

> 本文档讲清「上下文工程与记忆在 AI 里干什么」。原理见 `principle.md`。核心：LLM 上下文工程要在有限 context window 里塞什么、不塞什么、怎么塞、按什么顺序塞，需要 5 种数据结构协作——LRU 做淘汰、Trie 做前缀共享、哈希做记忆检索、环形缓冲做滑动窗口、栈做 agent 嵌套。

## 1. 上下文工程是什么

### 1.1 从 prompt 工程到上下文工程

**prompt 工程**（Prompt Engineering）关注：怎么写一个 prompt 让 LLM 输出更好。这是单轮的事。

**上下文工程**（Context Engineering）关注：多轮对话中，**往 context window 里塞什么、不塞什么、按什么顺序塞**。这是跨轮的事。

```
prompt 工程（单轮）：
  "请用简体中文解释 LRU 缓存" → LLM → 解释

上下文工程（多轮）：
  轮 1: 用户问 LRU → 塞进 context → LLM 回答
  轮 2: 用户问 Trie → 塞轮 1 的问答吗？塞多少？塞记忆吗？
  轮 3: 用户问哈希 → 塞轮 1、2 的问答吗？token 超了怎么办？
  ...
```

随着对话变长，context window 装不下所有历史，必须做选择：保留哪些、淘汰哪些、怎么压缩、怎么检索相关记忆。这就是上下文工程。

### 1.2 为什么上下文工程是核心

LLM 应用的质量很大程度取决于 context 里塞了什么：

- **塞太多**：token 超限、推理慢、成本高、噪声多反而答得差
- **塞太少**：缺关键上下文、答非所问、丢失用户意图
- **塞错顺序**：重要信息在中间（lost in the middle 问题），LLM 注意不到

好的上下文工程让 LLM 用有限 token 看到最相关的信息，是 LLM 应用从 demo 到生产的关键一跃。

## 2. context window 限制与 LRU 淘汰

### 2.1 context window 的硬限制

LLM 的 context window 是硬限制：输入 token 数不能超过上限。不同模型上限不同：

| 模型 | context window | 大致能装 |
|---|---|---|
| GPT-3.5 | 16K | 一篇短文 |
| GPT-4 | 128K | 一本书的一章 |
| Claude 3 | 200K | 一本小书 |
| Gemini 1.5 | 1M | 一本大书 |

但 context window 越大，推理越慢越贵。即使有 128K 窗口，也不应该塞满——研究表明 LLM 在超长 context 上有 **lost in the middle** 问题：中间的信息容易被忽略。

### 2.2 淘汰策略：LRU vs FIFO vs 随机

context 超限时必须淘汰。三种策略：

```
FIFO（先入先出）：
  淘汰最早进入 context 的
  问题：system prompt 最早进入，会被淘汰，但每轮都要用

随机淘汰：
  随机挑一个淘汰
  问题：可能淘汰正在用的热点

LRU（最近最少使用）：
  淘汰最久未访问的
  优势：system prompt 每轮都访问（最近用），永远不会被淘汰
```

实测：4000 次操作，85% 访问 5% 热点，LRU 命中率 82.5%，随机 71.4%，**LRU 高 11 个百分点**。

### 2.3 LRU 在 context 管理中的具体应用

LRU 在 context 管理中决定：

- **system prompt**：每轮都引用，`last_used` 不断更新，永远不被淘汰
- **最近几轮对话**：刚访问过，`last_used` 大，不会被淘汰
- **工具调用结果**：用完就不再引用，`last_used` 小，优先被淘汰
- **检索到的文档**：只在当前轮用，下一轮不引用，会被淘汰

这正好符合直觉：system prompt 和最近对话最重要，老的工具结果和过时的检索文档可以淘汰。

### 2.4 token 预算 vs 条目数预算

普通 LRU 按条目数淘汰（最多 100 条），context LRU 按 token 总数淘汰（最多 128K token）。区别：

- 条目数 LRU：每条等权，淘汰一条就够
- token 预算 LRU：每条占的 token 不同，可能要淘汰多条才够放一条大的

如：预算 4096 token，要放一条 2000 token 的消息，可能要淘汰好几条小消息才够。token 预算 LRU 更贴合 context window 的实际限制。

## 3. prompt 工程与 Trie 前缀共享

### 3.1 system prompt 的重复

LLM 应用的请求几乎都带 system prompt：

```
req_1: [你是一个数据结构助手，用简体中文回答...] + [用户问题 1]
req_2: [你是一个数据结构助手，用简体中文回答...] + [用户问题 2]
req_3: [你是一个数据结构助手，用简体中文回答...] + [用户问题 3]
```

system prompt 可能几千 token（角色设定 + 工具说明 + few-shot 示例）。如果每个请求独立处理，每次都要重新算前缀的 KV cache，浪费。

### 3.2 Trie 共享前缀

用 Trie 把多个请求的 prompt 按前缀合并：

```
Trie 结构（system prompt = [你, 是, 一个, 助手]）：
  根
   └─ 你
       └─ 是
           └─ 一个
               └─ 助手  ← system prompt 到这里，3 个请求共享
                   ├─ 用户问题 1
                   ├─ 用户问题 2
                   └─ 用户问题 3
```

system prompt 部分只存一次，3 个请求共享。省下的 token = (N-1) × len(system_prompt)。

实测：100 请求，system 500 token + 用户 50 token，无共享 55000 token，Trie 共享 5500 token，**省 90.2%**。

### 3.3 prefix caching 在推理框架中的应用

vLLM、SGLang 等推理框架的 **prefix caching** 就是 Trie 思想的工程实现：

- 按 token 序列算 hash，建前缀树
- 公共前缀的 KV cache 只算一次，存起来
- 新请求来时，先查前缀树，命中前缀的 KV cache 直接复用

这让多轮对话（每轮共享前 N-1 轮的 KV cache）和 batch 推理（多个请求共享 system prompt）的 prefill 成本大幅下降。vLLM 的 prefix caching 让多轮对话的 TTFT（首 token 延迟）降 5-10x。

### 3.4 prompt 工程的其他 token 优化

除了前缀共享，prompt 工程还有：

- **prompt 压缩**：用更少的 token 表达同样意思（如 "用中文答" 代替 "请使用简体中文进行回答"）
- **few-shot 选择**：从大量示例中选最相关的几个，而不是全塞
- **工具说明懒加载**：只在用到工具时才塞工具说明，不一开始就全塞

这些都是为了在有限 context window 里塞更多有效信息。

## 4. 记忆系统：短期 + 长期

### 4.1 为什么需要记忆

LLM 本身是无状态的——每次调用都是独立的，不记得之前说过什么。对话的"记忆"靠把历史消息塞进 context window 实现。但 context window 有限，长对话装不下所有历史。

记忆系统解决：**怎么把对话中的关键信息存下来，后续按需取回**。

### 4.2 短期记忆：滑动窗口

短期记忆 = 当前对话的最近几轮消息，用**环形缓冲**管理：

```
对话历史：[msg_1, msg_2, ..., msg_100]
滑动窗口（保留最近 8 条）：[msg_93, ..., msg_100]
```

环形缓冲 O(1) push，固定内存，老消息自动滚出。这是最简单的记忆策略，LangChain 的 `ConversationBufferWindowMemory` 就是这个。

缺点：老消息直接丢，可能丢失关键信息（如用户在第 1 轮说了偏好，第 100 轮还在用）。

### 4.3 长期记忆：哈希表 + 向量检索

长期记忆 = 跨对话的事实，存在外部存储。两种检索方式：

**按 key 精确取（哈希表）**：
```
记忆库：
  "user_pref_lang" → "Python"
  "user_level" → "beginner"
  "user_project" → "data_structure_tutorial"

每轮开始时：
  lang = memory.get("user_pref_lang")  # O(1)
  level = memory.get("user_level")     # O(1)
  → 塞进 context：[用户偏好 Python，是新手]
```

适合存结构化事实（用户偏好、项目信息、设定参数）。哈希表 O(1) 查找，5000 条记忆 5000 次查询只要 1.74 ms。

**按语义相似度（向量检索）**：
```
记忆库（embedding）：
  "用户喜欢 Python" → [0.1, 0.8, ...]
  "用户在学数据结构" → [0.3, 0.7, ...]
  "用户用 VSCode" → [0.5, 0.2, ...]

查询："用户喜欢什么编程语言"
  → embedding("用户喜欢什么编程语言") = [0.1, 0.7, ...]
  → 最近邻搜索 → "用户喜欢 Python"
```

适合存非结构化记忆，按语义相关性检索。用 HNSW 等向量索引（见第 12 章）。

### 4.4 摘要记忆：压缩老消息

另一种记忆策略：把老消息**摘要**成一段，而不是直接丢或全留：

```
原始历史（100 轮，20000 token）：
  msg_1: 用户问 LRU
  msg_2: 助手答 LRU 是...
  ...
  msg_100: 用户问栈

摘要记忆（1 段，200 token）：
  "之前讨论了 LRU、Trie、哈希表、环形缓冲的原理和应用"

当前 context：
  [摘要 200 token] + [最近 8 轮原始消息 2000 token]
  → 用 2200 token 覆盖了 100 轮的关键信息
```

LangChain 的 `ConversationSummaryMemory` 实现这个。摘要本身要调 LLM（有成本），但摘要后每轮省大量 token。

### 4.5 MemGPT 的分级记忆

MemGPT（现 Letta）把操作系统的内存管理思想搬到 LLM，分级记忆：

| 层级 | 类比 | 内容 | 淘汰策略 |
|---|---|---|---|
| 主 context | 寄存器 | 当前对话窗口 | LRU |
| 记忆上下文 | 内存 | 存不下的内容 | 按需 page in/out |
| 归档记忆 | 硬盘 | 长期存储 | 向量检索 |

LRU 决定什么留主 context、什么 page out 到记忆、什么归档。这模拟了 OS 的 LRU 页面置换，让 LLM 在有限 context 里"记住"远超窗口的信息。

## 5. agent 调用嵌套与栈

### 5.1 多 agent 协作

现代 AI 应用常用多 agent 协作：主 agent 负责理解用户意图，子 agent 负责具体执行：

```
main_agent（理解意图）
  ├── code_agent（写代码）
  │     └── test_agent（跑测试）
  ├── search_agent（搜索资料）
  │     └── tool_agent（调用搜索 API）
  └── review_agent（审查结果）
```

main_agent 调用 code_agent，code_agent 又调用 test_agent，形成嵌套。

### 5.2 栈管理嵌套返回

嵌套调用必须按**后进先出**返回：最后调用的 agent 最先返回。

```
调用顺序：
  push(main_agent) → push(code_agent) → push(test_agent)

返回顺序（栈 pop）：
  pop → test_agent 返回给 code_agent
  pop → code_agent 返回给 main_agent
  pop → main_agent 返回给用户
```

用栈保证返回顺序正确。如果错误地用队列（先调用的先返回），main_agent 会先返回，但 code_agent 还在跑，逻辑混乱。

### 5.3 栈在异常传播中的作用

agent 嵌套中，内层 agent 抛异常时，栈逐层 pop 触发各层 fallback：

```
test_agent 抛异常 "测试失败"
  → pop test_agent
  → code_agent 收到异常，fallback：返回"代码需要修改"
  → code_agent 决定重试或返回
  → 如果 code_agent 也抛异常，pop
  → main_agent 收到异常，fallback：向用户道歉并请求澄清
```

这就是编程语言异常处理的栈展开（stack unwinding），agent 系统复用这个机制做错误传播和恢复。

### 5.4 递归 agent 与栈深度

有些 agent 会递归调用（如 agent A 调 agent B，agent B 又调 agent A）。栈的深度反映递归深度。要防止无限递归：

- 设最大栈深度（如 10），超过就拒绝
- 检测环（A 调 B 调 A），拒绝或用缓存打破环

这和编程语言防止栈溢出的思路一样。

## 6. demo 结果解读

### 6.1 LRU vs 随机淘汰

```
场景：4000 次操作，token 预算 1024，85% 访问 5% 热点
LRU 淘汰  ：命中率 82.5%
随机淘汰  ：命中率 71.4%
→ LRU 高 11 个百分点
```

解读：访问有局部性时（大部分访问集中在小部分热点），LRU 保留热点，命中率显著高于随机。11 个百分点在生产环境意味着：每 100 次访问，LRU 多 11 次命中，少 11 次重新加载/计算，对延迟和成本影响巨大。

### 6.2 Trie 前缀共享

```
场景：100 请求，system prompt 500 token + 用户消息 50 token
无共享总 token = 55000
Trie 共享后   = 5500
省 90.2%
```

解读：system prompt 占 prompt 的大部分（500/550 = 91%），共享后只存一次，省的比例约等于 system prompt 占比。生产环境 system prompt 常几千 token，省的比例更高。vLLM 的 prefix caching 就是这个思路的工程实现。

### 6.3 环形缓冲 vs list

```
场景：100000 次 push，容量 5000
环形缓冲    ：132 ms（push O(1)）
list+pop(0) ：1881 ms（push 满 O(n)）
快 14x
```

解读：list.pop(0) 要前移所有元素，O(n)，n=5000 时每次前移 4999 个元素，100000 次就是 5 亿次元素移动。环形缓冲只改指针，O(1)，与容量无关。流式写入场景（消息不断到达）环形缓冲优势显著。

### 6.4 哈希 vs 线性扫描

```
场景：5000 条记忆，5000 次 get 查询
哈希表    ：1.74 ms（每次 O(1)）
线性扫描  ：470 ms（每次 O(n)）
快 270x
```

解读：线性扫描每次 get 要遍历整个 list 找 key，5000 条 × 5000 次 = 2500 万次比较。哈希表每次 get 直接定位桶，5000 次只要 5000 次哈希。记忆条数越多，哈希表优势越大。

## 7. 生产级实现

### 7.1 LangChain Memory

LangChain 的 Memory 模块覆盖本章讨论的多种记忆策略：

| Memory 类 | 机制 | 对应数据结构 |
|---|---|---|
| `ConversationBufferMemory` | 全量保留 | list |
| `ConversationBufferWindowMemory` | 滑动窗口 | 环形缓冲 |
| `ConversationSummaryMemory` | 摘要记忆 | list + LLM 摘要 |
| `ConversationSummaryBufferMemory` | 摘要 + 滑动窗口 | 环形缓冲 + LLM 摘要 |
| `ConversationKGMemory` | 知识图谱记忆 | 图 + 三元组 |
| `VectorStoreRetrieverMemory` | 向量记忆 | HNSW + embedding |

### 7.2 MemGPT / Letta

MemGPT（现 Letta）的分级记忆：

- **主 context**：LRU 淘汰，像 OS 的页面置换
- **记忆上下文**：按需 page in/out，像 OS 的内存
- **归档记忆**：向量检索，像 OS 的硬盘

MemGPT 让 LLM 在 8K context 里"记住"远超 8K 的信息，靠的就是 LRU + 分级存储 + 按需检索。

### 7.3 vLLM prefix caching

vLLM 的 prefix caching 用 Trie 思想：

- 按 token 序列算 hash，建前缀树
- 公共前缀的 KV cache 只算一次
- 新请求命中前缀直接复用 KV cache

让多轮对话的 TTFT 降 5-10x，是 vLLM 性能领先的关键技术之一。

### 7.4 OpenAI Assistants API

OpenAI 的 Assistants API 内置记忆管理：

- **Threads**：对话历史存储（环形缓冲 + 滑动窗口）
- **Messages**：按 thread 组织消息
- **Retrieval**：文件上传后自动 embedding + 向量检索
- **Code Interpreter**：工具调用（栈管理嵌套）

用户不用自己管 context window，API 内部用类似机制处理。

## 8. 设计权衡

### 8.1 保留多少历史

| 策略 | token 成本 | 信息保留 | 适用 |
|---|---|---|---|
| 全量保留 | 高（超窗口就崩） | 完整 | 短对话 |
| 滑动窗口 | 低 | 最近 N 轮 | 多轮对话 |
| 摘要记忆 | 中 | 摘要 + 最近几轮 | 长对话 |
| 向量检索 | 低（按需检索） | 相关片段 | 超长对话 |

没有银弹，按对话长度和 token 预算选。

### 8.2 淘汰策略选择

| 策略 | 命中率 | 实现复杂度 | 适用 |
|---|---|---|---|
| FIFO | 低 | 简单 | 均匀访问 |
| LRU | 高 | 中 | 有局部性 |
| LFU | 高（频率热点） | 中 | 频率稳定的访问 |
| ARC（自适应） | 更高 | 复杂 | 访问模式变化 |

context 管理常用 LRU（实现简单，命中率高）。数据库缓存常用 LFU 或 ARC。

### 8.3 记忆检索方式

| 检索方式 | 精确度 | 召回率 | 适用 |
|---|---|---|---|
| 哈希表（按 key） | 精确 | 低（要 key 完全匹配） | 结构化事实 |
| 向量检索（按语义） | 模糊 | 高 | 非结构化记忆 |
| 全文搜索 | 中 | 中 | 关键词匹配 |
| 知识图谱 | 精确 | 中 | 实体关系 |

结构化记忆用哈希表，非结构化用向量检索，复杂关系用知识图谱。生产系统常组合多种。

## 9. 小结

上下文工程是 LLM 应用从 demo 到生产的关键：

- **context window 限制** → LRU 淘汰（保留热点，token 不超限）
- **prompt 前缀共享** → Trie（多请求共享 system prompt，省 90%+ token）
- **长期记忆检索** → 哈希表（按 key O(1) 取，270x 快于线性扫描）
- **对话滑动窗口** → 环形缓冲（O(1) push，14x 快于 list）
- **agent 调用嵌套** → 栈（后进先出匹配返回）

生产级系统（LangChain Memory、MemGPT、vLLM prefix caching、OpenAI Assistants API）用不同组合实现类似机制。理解这 5 种数据结构的协作，才能设计出在有限 context 里塞最多有效信息的上下文工程系统。