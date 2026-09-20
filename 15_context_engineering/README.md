# 第 15 章 · 上下文工程与记忆（扩展专题）

> AI 场景：LLM 的 context window 管理、prompt 工程、短期/长期记忆、agent 调用嵌套
> 数据结构：LRU + Trie + 哈希 + 环形缓冲 + 栈（5 种协作）

## 本章要回答的问题

- context window 的 token 超限时淘汰谁？（LRU）
- 多请求共享 system prompt 怎么省 token？（Trie 前缀共享）
- 按 key 取长期记忆怎么快？（哈希表 O(1)）
- 对话只看最近 N 条怎么高效实现？（环形缓冲滑动窗口）
- agent 调 agent 的嵌套返回怎么匹配？（栈后进先出）
- 5 种数据结构如何协奏？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示 LRU、Trie、哈希表、环形缓冲、栈如何协作解决 LLM 上下文工程的全链路问题。

## 目录

```
15_context_engineering/
├── python/
│   └── demo.py              # LLM 上下文管理 demo（5 种数据结构协作）
├── docs/
│   ├── principle.md         # 5 种数据结构原理 + 协作机制
│   └── ai_application.md    # 上下文工程应用（LangChain/MemGPT/vLLM）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟 LLM 上下文管理系统，5 种数据结构协作：

```
用户消息到达
  ├─→ [环形缓冲] 滑动窗口（保留最近 N 条）
  ├─→ [哈希表]   记忆检索（按 key 取长期记忆）
  ├─→ [Trie]     prompt 前缀共享（多轮共享 system prompt）
  ├─→ [LRU]      context 淘汰（超 token 预算淘汰最久未用）
  └─→ [栈]       agent 嵌套（后进先出匹配返回）
```

- `ContextLRU`：LRU，context window 的 token 预算淘汰
- `PromptTrie`：Trie，prompt 前缀共享（多请求共享 system prompt）
- `MemoryStore`：哈希表，记忆检索（按 key O(1) 取）
- `DialogRingBuffer`：环形缓冲，对话滑动窗口（固定大小，O(1) push）
- `CallStack`：栈，agent 调用嵌套（后进先出匹配返回）

性能对比：
1. LRU 淘汰 vs 随机淘汰（命中率 82.5% vs 71.4%，高 11 个百分点）
2. Trie 前缀共享 vs 无共享（省 90.2% token）
3. 环形缓冲 vs list+pop(0)（push 快 14x）
4. 哈希记忆 vs 线性扫描（get 快 270x）

## 跑法

```bash
python 15_context_engineering/python/demo.py
```

## 状态

✅ 已完成