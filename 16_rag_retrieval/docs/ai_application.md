# RAG 与知识检索：检索增强生成的数据结构协作

> 本文档讲清 RAG（检索增强生成）的背景、流程、多路检索融合、知识图谱增强、去重的重要性，解读 demo 结果，并介绍生产级实现（LangChain RAG、LlamaIndex、GraphRAG）。数据结构原理见 `principle.md`。

## 1. RAG：检索增强生成的背景

### 1.1 LLM 的知识缺陷

大语言模型（LLM）很强，但有几个致命缺陷：

1. **知识截止**：训练数据有截止日期，不知道之后的事。GPT-4 训练到 2023 年 4 月，问它 2024 年的事它不知道。
2. **幻觉**（hallucination）：不知道的问题也硬编一个答案，听起来像真的。问「爱因斯坦的第三篇论文标题是什么」它编一个。
3. **无私有知识**：企业内部文档、个人笔记，LLM 没见过，答不了。
4. **不可溯源**：LLM 直接生成的答案，不知道依据是什么，无法验证。

### 1.2 RAG 的思路

**RAG**（Retrieval-Augmented Generation，检索增强生成）：先从知识库检索相关文档，再把文档塞进 LLM 的 context，让 LLM 基于文档生成回答。

```
用户提问："HNSW 用了哪些数据结构？"
  │
  ├─→ 检索：从知识库找相关文档
  │     → [doc_9: "HNSW 是近似最近邻搜索算法..."]
  │     → [doc_7: "跳表是多层索引链表..."]
  │     → [doc_8: "图用邻接表或邻接矩阵存储..."]
  │
  └─→ 生成：LLM(context = 文档 + 提问)
        → "HNSW 用了跳表（提供分层入口点）和近邻图（贪心搜索）"
```

RAG 解决了 LLM 的四个缺陷：

| 缺陷 | RAG 怎么解决 |
|---|---|
| 知识截止 | 知识库可以随时更新 |
| 幻觉 | 让 LLM 基于检索到的文档生成，减少编造 |
| 无私有知识 | 把私有文档放进知识库 |
| 不可溯源 | 返回检索到的文档作为引用 |

### 1.3 RAG 的标准流程

```
[离线建库]
  文档 → 切片（chunk）→ Embedding → 向量入库（HNSW）
                                → 关键词入库（倒排索引）
                                → 实体关系入库（知识图谱）

[在线查询]
  query → Embedding → 语义检索（HNSW）
       → 分词     → 关键词检索（倒排索引）
       → 抽实体   → 多跳推理（知识图谱）
       → 融合去重（布隆）
       → rerank（cross-encoder 重排）
       → LLM 生成（context = 检索结果 + query）
```

---

## 2. 多路检索融合

### 2.1 为什么需要多路

单路检索各有盲区：

- **纯语义检索**：「快」vs「迅速」能匹配，但「HNSW」这种专有名词可能向量不够近，漏掉。
- **纯关键词检索**：「HNSW」能精确匹配，但「快速排序」的 query 搜不到「分治算法」的文档（用词不同）。
- **纯知识图谱**：「A 的导师的学生」能多跳推理，但模糊语义匹配不行。

**多路融合**取各路并集，互补盲区。本 demo 的结果：

```
单路（仅语义）  ：召回率 62.0%
多路（语义+关键词+图谱）：召回率 100.0%
→ 多路融合召回率提升 38.0 个百分点
```

### 2.2 融合策略

**去重合并**（本 demo）：取所有路的并集，布隆去重。最简单。

**加权融合**：给每个文档打综合分数：

```
score(doc) = w_sem · sem_score(doc) + w_kw · bm25(doc) + w_graph · graph_score(doc)
```

**RRF**（Reciprocal Rank Fusion）：按各路排名取倒数加权，不需要分数对齐：

```
RRF_score(doc) = sum(1 / (k + rank_i(doc)))  for each retriever i
```

k 通常取 60。RRF 是生产级 RAG 最常用的融合策略，因为它不依赖各路分数的绝对值（语义检索的 cosine 和 BM25 的分数量纲不同）。

**重排**（rerank）：融合后取 top-50，用 cross-encoder（如 BGE-reranker）对 query 和每个文档做交叉注意力打分，重新排序取 top-5。cross-encoder 比双塔 Embedding 更准但更慢，所以只对候选集做。

### 2.3 生产级的多路检索

LangChain 的 `EnsembleRetriever` 支持多路融合：

```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever, graph_retriever],
    weights=[0.5, 0.3, 0.2],
)
docs = ensemble.invoke(query)
```

---

## 3. 知识图谱增强（GraphRAG）

### 3.1 传统 RAG 的局限

传统 RAG 只做「1 跳」检索：query 直接匹配文档片段。但有些问题需要跨文档推理：

- 「这本书的作者还写过哪些书？」→ 需要查「作者」实体，再查作者的其他书
- 「这个 API 的参数类型有哪些方法？」→ 需要查参数类型，再查类型的方法
- 「A 公司的 CEO 之前在哪家公司？」→ 需要查 CEO，再查 CEO 的前公司

这些问题单次检索答不了，需要多跳。

### 3.2 GraphRAG 的思路

**GraphRAG**（微软 2024 年提出）：在向量索引之外，再建一个知识图谱：

1. **实体抽取**：用 LLM 从文档里抽实体和关系（如「HNSW --uses--> 跳表」）
2. **图构建**：实体做节点，关系做边，存到图数据库
3. **多跳检索**：query 抽实体 → 在图上 BFS 多跳 → 路径上的文档

```
query: "RAG 用了哪些数据结构？"
  → 抽实体: ["RAG"]
  → 图上多跳:
    RAG --uses--> 向量检索 --is_a--> 语义搜索
    RAG --uses--> 倒排索引 --for--> 搜索引擎
    RAG --uses--> 知识图谱 --uses--> 图
    RAG --uses--> 布隆过滤器
  → 路径上的文档: [doc_12, doc_14, doc_10, doc_13]
```

### 3.3 本 demo 的知识图谱

本 demo 手工构建了一个小知识图谱（25 实体、23 关系），演示多跳推理：

```python
kg.add_relation("HNSW", "uses", "跳表", doc_id=9)
kg.add_relation("HNSW", "uses", "图", doc_id=9)
kg.add_relation("RAG", "uses", "向量检索", doc_id=12)
kg.add_relation("RAG", "uses", "倒排索引", doc_id=12)
# ...

paths = kg.multi_hop("RAG", hops=2)
# → RAG → 向量检索 → 语义搜索
# → RAG → 倒排索引 → 搜索引擎
# → RAG → 知识图谱 → 图
```

### 3.4 生产级 GraphRAG

微软的 [graphrag](https://github.com/microsoft/graphrag)：

- 用 LLM 自动抽实体和关系（不需要手工标注）
- 社区检测（Leiden 算法）把图分成主题聚类
- 查询时：局部查询（多跳）+ 全局查询（社区摘要）

Neo4j + LangChain 的 GraphRAG：

```python
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(url, username, password)
chain = GraphCypherQAChain.from_llm(llm, graph)
answer = chain.invoke("A 的导师的学生是谁？")
```

---

## 4. 去重的重要性

### 4.1 多路召回必然有重复

三路检索经常召回同一文档：

```
语义检索：[doc_12, doc_1, doc_16, doc_18, doc_3]
关键词检索：[doc_12, doc_6, doc_7, doc_8]
知识图谱：[doc_12, doc_14, doc_10, doc_13]
```

doc_12 被三路都召回（因为它既语义相关、又含关键词、又在图谱路径上）。如果不去重，doc_12 会在 context 里出现三次，浪费 token。

### 4.2 去重的代价

LLM 的 context window 有上限（GPT-4 是 128K token）。每个重复文档浪费几百到几千 token。在 RAG 系统里，context 要塞：

- system prompt（几百 token）
- 检索到的文档（每个几百到几千 token）
- 对话历史（可能几千 token）
- 用户 query（几十 token）

如果 10 个文档里有 3 个重复，浪费 30% 的文档 token。这些 token 本可以塞更多不重复的文档，提升召回率。

### 4.3 布隆去重 vs 集合去重

本 demo 对比了两种去重方式：

| 指标 | 布隆过滤器 | set |
|---|---|---|
| 内存（10 万元素） | 117 KB | 2734 KB |
| 误判率 | 0.4% | 0% |
| lookup 延迟 | 220 ms | 7 ms |

**布隆省内存 23 倍**，代价是 0.4% 的误判率（把新文档误判为重复 → 漏掉）。在 RAG 里误判不致命：多路召回本来就有冗余，漏一个还有其他路补上。

**set 在 Python 里 lookup 更快**，因为 set 是 C 实现的。布隆的优势在：

1. **内存**：大规模下（千万级）省几十倍内存
2. **跨语言/跨进程**：布隆的位数组可以序列化共享，set 不行
3. **C++/Rust 实现**下布隆 lookup 也是 O(1) 且常数小

### 4.4 什么时候用布隆

- **文档数千万级以上**：set 内存吃不起，布隆省几十倍
- **跨进程共享去重状态**：布隆的位数组可以 mmap 共享
- **允许少量误判**：RAG 多路召回有冗余，漏一个不致命
- **流式去重**：新文档不断来，布隆可以增量 add

---

## 5. demo 结果解读

### 5.1 各路检索演示

```
查询：「RAG 用了哪些数据结构？」
  [semantic] 0.145 ms, 召回 5 个: [12, 1, 16, 18, 3]
  [ keyword] 0.004 ms, 召回 0 个: []
  [   graph] 0.021 ms, 召回 10 个: [12, 12, 12, 12, 12, 14, 10, 12, 12, 13]
  融合前 = 15 个（含重复）
  布隆去重后 = 8 个: [12, 1, 16, 18, 3, 14, 10, 13]
```

- **语义检索**召回 5 个（top-k=5），doc_12 是 RAG 文档，cosine 最高
- **关键词检索**召回 0 个——因为查询词「RAG」「数据结构」在 bigram 分词后没匹配到（RAG 是英文，数据结构是中文 bigram，文档里「数据结构」的 bigram 是「数据」「据结」「结构」）
- **知识图谱**召回 10 个（含重复），多跳路径上的文档
- **布隆去重**把 15 个（含重复）去重为 8 个

### 5.2 多路 vs 单路召回率

```
单路（仅语义）  ：召回率 62.0%
多路（语义+关键词+图谱）：召回率 100.0%
→ 提升 38 个百分点
```

50% 的查询向量与目标文档不相似（模拟用户用词与文档不同），单路语义检索 miss；多路靠关键词和图谱补上。

### 5.3 倒排索引 vs 全文扫描

```
倒排索引 ：323 ms
全文扫描 ：1740 ms
→ 倒排索引快 5.4x
```

2000 文档、1000 次 AND 查询。倒排索引查 posting list 交集，全文扫描每次扫所有文档做子串匹配。文档数越多差距越大（倒排 O(sum posting len) vs 全文 O(n·L)）。

### 5.4 布隆误判率

```
设计 p=1%，实测误判率 0.4%（10 万元素）
```

实测误判率低于设计值，因为实际元素数可能小于预期。布隆的误判率随元素数增加而上升，到预期容量时接近设计值。

---

## 6. 生产级 RAG 实现

### 6.1 LangChain RAG

LangChain 是最流行的 LLM 应用框架，RAG 是核心功能：

```python
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# 1. 切片 + Embedding + 入库
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 2. 多路检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
keyword_retriever = BM25Retriever.from_documents(chunks)
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.5, 0.5],
)

# 3. rerank
reranker = CrossEncoderReranker(model=model, top_n=5)

# 4. RAG chain
rag_chain = (
    {"context": ensemble | reranker | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = rag_chain.invoke("HNSW 用了哪些数据结构？")
```

### 6.2 LlamaIndex

LlamaIndex 专注 RAG，数据连接器更丰富：

```python
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex
from llama_index.core import StorageContext, load_index_from_storage

# 向量索引
vector_index = VectorStoreIndex.from_documents(documents)

# 知识图谱索引
kg_index = KnowledgeGraphIndex.from_documents(documents)

# 查询
query_engine = vector_index.as_query_engine(similarity_top_k=5)
response = query_engine.query("HNSW 用了哪些数据结构？")
```

### 6.3 GraphRAG（微软）

```python
from graphrag.index import run_index
from graphrag.query import run_query

# 1. 用 LLM 抽实体关系，建知识图谱
run_index(config)

# 2. 全局查询（社区摘要）+ 局部查询（多跳）
result = run_query("RAG 用了哪些数据结构？", method="global")
```

### 6.4 向量数据库选型

| 数据库 | 特点 | 适用场景 |
|---|---|---|
| FAISS | Meta 开源，纯向量检索，单机快 | 中小规模（< 1000 万） |
| Milvus | 分布式向量数据库，支持多种索引 | 大规模（亿级） |
| pgvector | PostgreSQL 扩展，向量 + 关系混合查询 | 已有 PG 的项目 |
| Qdrant | Rust 实现，支持过滤 + 向量 | 需要元数据过滤 |
| Chroma | 轻量级，开发友好 | 原型 / 小项目 |

### 6.5 检索框架选型

| 框架 | 特点 |
|---|---|
| LangChain | 通用 LLM 框架，RAG 是一部分，生态最全 |
| LlamaIndex | 专注 RAG，数据连接器丰富 |
| Haystack | deepset 出品，企业级 RAG |
| GraphRAG | 微软出品，知识图谱增强 |

---

## 7. RAG 的工程挑战

### 7.1 切片策略

文档怎么切片直接影响检索质量：

- **固定长度切片**：每 500 字符切一片，简单但可能切断语义
- **递归切片**：按段落 → 句子 → 字符递归切，保持语义完整
- **语义切片**：用 Embedding 计算相邻句子相似度，在相似度低的地方切
- **滑动窗口**：重叠切片（overlap=50），避免边界信息丢失

### 7.2 Embedding 模型选择

| 模型 | 维度 | 中文 | 特点 |
|---|---|---|---|
| BGE-large-zh | 1024 | 好 | 智源，中文最强 |
| text-embedding-3-small | 1536 | 一般 | OpenAI，便宜 |
| text-embedding-3-large | 3072 | 一般 | OpenAI，最准 |
| m3e-base | 768 | 好 | 开源中文 |
| Cohere embed-v3 | 1024 | 一般 | 多语言 |

### 7.3 rerank 的价值

Embedding 是**双塔模型**（query 和 doc 独立编码），快但不够准。rerank 用 **cross-encoder**（query 和 doc 一起编码），准但慢：

```
Embedding：query → vec_q, doc → vec_d, score = cosine(vec_q, vec_d)
Cross-encoder：score = model(query, doc)  # 一起编码
```

生产级 RAG 的标准做法：Embedding 检索 top-50 → cross-encoder rerank top-5。

### 7.4 评估指标

- **召回率**（recall@k）：正确文档在 top-k 里的比例
- **MRR**（Mean Reciprocal Rank）：正确文档排名的倒数平均
- **nDCG**：考虑排名位置的加权得分
- ** Faithfulness**：生成的答案是否忠于检索到的文档
- **Answer Relevance**：生成的答案是否回答了问题

---

## 8. 小结

RAG 是大模型时代最重要的 AI 应用模式之一。它的核心是**检索**，而检索的质量取决于**数据结构**：

1. **HNSW** 做语义检索——快（O(log n)）、准（语义匹配）
2. **倒排索引** 做关键词检索——精确（专有名词）、快（O(sum posting len)）
3. **知识图谱** 做多跳推理——跨文档推理、可溯源
4. **布隆过滤器** 做去重——O(1) 判重、省内存

多路融合召回率显著高于单路（本 demo 62% → 100%）；布隆省内存几十倍（代价是可控误判）；倒排索引比全文扫描快数倍。

生产级 RAG 用 LangChain / LlamaIndex 框架 + FAISS / Milvus 向量库 + Elasticsearch 倒排 + Neo4j 图谱 + cross-encoder rerank，处理百万到亿级文档。本 demo 用 200 行 Python 展示了同样的协作原理。