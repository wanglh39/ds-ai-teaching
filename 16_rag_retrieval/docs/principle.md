# RAG 与知识检索：4 种数据结构协作的原理

> 本文档讲清 RAG（检索增强生成）中 4 种数据结构——HNSW、倒排索引、知识图谱、布隆过滤器——各自的原理，以及它们如何协作解决「多路检索 + 去重」的全链路问题。AI 应用（RAG）见 `ai_application.md`。

## 1. 问题：为什么单一检索不够

RAG 的核心是「先检索再生成」：用户提问 → 从知识库检索相关文档 → 把文档塞进 LLM 的 context → LLM 生成回答。检索质量直接决定回答质量。

但**没有一种检索方式是万能的**：

| 检索方式 | 擅长 | 不擅长 |
|---|---|---|
| 语义检索（向量） | 「快」vs「迅速」这种意思相近 | 精确匹配专有名词、代码标识符 |
| 关键词检索（倒排） | 精确匹配、稀有词 | 同义词、 paraphrase |
| 知识图谱（图） | 多跳推理「A 的导师的学生」 | 模糊匹配、全文语义 |

**多路检索 + 融合**是生产级 RAG 的标准做法。但多路检索会召回重复文档（同一文档被语义和关键词都命中），需要**去重**。这就是 4 种数据结构协作的动机：

```
用户 query
  ├─→ [HNSW]      语义检索（向量近邻）
  ├─→ [倒排索引]  关键词检索（词 → 文档列表）
  ├─→ [知识图谱]  多跳推理（实体关系，BFS）
  └─→ [布隆]      融合去重（O(1) 判重）
```

本文档逐一讲清 4 种数据结构的原理，再讲它们如何协作。

---

## 2. HNSW：亚线性向量检索

### 2.1 向量检索是什么

把文本/图片用 Embedding 模型编码成稠密向量（如 768 维），找离查询向量最近的 k 个向量。这是**语义检索**的基础——「意思相近」的文本向量距离也近。

**暴力做法**：算查询向量到所有 n 个向量的距离，排序取前 k。复杂度 O(n·d)，100 万向量、768 维，一次查询要算 7.68 亿次浮点运算，太慢。

### 2.2 HNSW 的核心思想

**HNSW**（Hierarchical Navigable Small World）= **跳表的分层思想** + **近邻图的贪心搜索**：

1. **近邻图**：每个向量是一个节点，连到它最近的 M 个邻居。搜索时从入口点出发，贪心走向更近的节点，直到找不到更近的。
2. **分层结构**（跳表思想）：上层图是下层的抽样，越上层节点越少越稀疏。搜索从最高层开始快速定位到大致区域，逐层下降精细搜索。

```
第 2 层:  A ──────────────→ J          （稀疏，快速跳过远距离）
第 1 层:  A ────→ D ────→ G ────→ J    （中等密度）
第 0 层:  A → B → C → D → E → F → ... → J  （完整近邻图）
```

搜索 q：从第 2 层 A 出发，贪心走到 J（最近的大区域）；下降到第 1 层 J 附近，贪心走到 G；下降到第 0 层 G 附近，精搜到 q 的 top-k。

### 2.3 复杂度

- **构建**：O(n·log n)（每个节点插入时搜索 log n 个邻居）
- **搜索**：O(log n)（亚线性，对比暴力 O(n)）
- **空间**：O(n·M)（每个节点存 M 个邻居）

### 2.4 本 demo 的简化

本 demo 用**暴力 KNN** 代替完整 HNSW（重点是展示协作，不是 HNSW 本身）。完整 HNSW 见第 12 章。接口一致：`build(docs)` + `search(query_vec, k)` → `[(doc_id, score), ...]`。

```python
class SimpleHNSW:
    def search(self, query_vec, k=5):
        scored = [(d.id, cosine(query_vec, d.vector)) for d in self._docs]
        scored.sort(reverse=True)
        return scored[:k]
```

### 2.5 在 RAG 中的角色

用户 query → Embedding 模型 → query 向量 → HNSW 搜 top-k 相似文档。解决「意思相近但用词不同」的语义匹配。

---

## 3. 倒排索引：词到文档的映射

### 3.1 全文扫描的瓶颈

关键词检索最朴素的做法：遍历所有文档，检查查询词是否在文档里出现（子串匹配）。复杂度 O(n·L)，n 是文档数，L 是平均文档长度。100 万文档、平均 1KB，一次查询要扫 1GB 数据。

### 3.2 倒排索引的结构

**倒排索引**（Inverted Index）：预先建好「词 → 出现该词的文档 ID 列表」的映射。查询时直接查映射，不扫文档。

```
词项        文档 ID 列表（posting list，按 ID 升序）
─────────   ──────────────────────────
"排序"      [3, 4, 9, 12]
"算法"      [3, 4, 5]
"哈希"      [6, 11]
"向量"      [9, 12, 14]
```

### 3.3 构建流程

1. **分词**：把文档文本切成词项（token）。英文按空格 + 去标点 + 小写；中文需要分词器（本 demo 用 bigram 简化）。
2. **去重**：同一文档同一词只记一次。
3. **排序**：每个词的文档 ID 列表按升序排，便于后续归并。

```python
class InvertedIndex:
    def build(self, docs):
        for d in docs:
            toks = self._tokenize(d.text)
            for tok in set(toks):
                self._postings[tok].append(d.id)
        for word in self._postings:
            self._postings[word].sort()
```

### 3.4 查询：AND / OR / phrase

**AND 查询**（所有词都出现）：多个 posting list 求交集。用**双指针归并**，O(sum L)：

```python
def search_and(self, words):
    postings = [self._postings[w] for w in words]
    result = postings[0]
    for p in postings[1:]:
        result = self._intersect(result, p)  # 双指针求交集
    return result
```

**OR 查询**（任一词出现）：多个 posting list 求并集，去重。

**短语查询**（连续匹配）：先 AND 查候选，再验证 token 序列里是否连续出现。

### 3.5 中文分词的简化

生产级中文分词用 jieba、HanLP 等分词器。本 demo 用 **bigram（2-gram）** 简化：把连续中文字符切成所有相邻 2 字组合。

```
"排序算法" → ["排", "序", "算", "法", "排序", "序算", "算法"]
```

这样查询「排序」能匹配到文档里的「排序」bigram。虽然会有些噪声（如「序算」这种无意义 bigram），但教学够用。

### 3.6 复杂度

- **构建**：O(n·L)（扫所有文档分词）
- **AND 查询**：O(sum(每个词的 posting list 长度))，对稀有词极快（posting list 短）
- **空间**：O(总词频) 或 O(去重后词频)

### 3.7 在 RAG 中的角色

用户 query → 分词 → 倒排索引查 AND/OR → 命中文档。解决「精确匹配专有名词、代码标识符」——这些词语义检索可能漏掉（向量不够近），但关键词检索能精确命中。

---

## 4. 知识图谱：实体关系的图存储

### 4.1 为什么需要知识图谱

语义检索和关键词检索都是「1 跳」的：query 直接匹配文档。但有些问题需要**多跳推理**：

- 「A 的导师是谁的学生？」→ A →导师→ B →学生→ C（2 跳）
- 「Python 的创建者用的语言？」→ Python →创建者→ Guido →使用→ Python（2 跳）
- 「HNSW 用了哪些数据结构？」→ HNSW →uses→ 跳表/图（1 跳）

这种问题单次检索答不了，需要在**知识图谱**上走多跳。

### 4.2 知识图谱的结构

知识图谱 = **实体**（节点）+ **关系**（边）。用**邻接表**存储：

```
实体         邻接表：[(关系, 目标实体, 来源文档ID), ...]
─────────   ──────────────────────────────────────
"HNSW"      [("uses", "跳表", 9), ("uses", "图", 9), ("for", "向量检索", 9)]
"RAG"       [("uses", "向量检索", 12), ("uses", "倒排索引", 12), ...]
"BERT"      [("uses", "Transformer", 16)]
```

每条关系还带**来源文档 ID**，这样多跳推理找到的实体能回溯到原文档。

### 4.3 多跳推理：BFS

从起始实体出发，BFS 扩展 hops 层，记录所有可达路径：

```python
def multi_hop(self, start, hops=2):
    queue = [(start, [start], frozenset())]  # (实体, 路径, 文档ID集)
    for _ in range(hops):
        for entity, path, doc_ids in queue:
            for rel, dst, doc_id in self._adj[entity]:
                # 扩展到邻居
                new_path = path + [dst]
                new_doc_ids = doc_ids | {doc_id}
                results.append((new_path, new_doc_ids))
    return results
```

**避免环**：用 `visited_paths` 记录已访问路径，不重复扩展。

### 4.4 最短路径：BFS

找两个实体间的最短路径（如 BERT → Transformer），用标准 BFS。

### 4.5 复杂度

- **空间**：O(V + E)，V 实体数，E 关系数
- **1 跳查询**：O(1)（直接查邻接表）
- **多跳查询**：O(b^h)，b 是平均分支因子，h 是跳数
- **最短路径**：O(V + E)（BFS）

### 4.6 在 RAG 中的角色

用户 query → 抽实体 → 知识图谱多跳 → 路径上的文档 ID。这就是 **GraphRAG** 的核心：用知识图谱增强检索，解决多跳推理问题。微软 2024 年的 GraphRAG 论文就是这条路。

---

## 5. 布隆过滤器：概率去重

### 5.1 去重的问题

多路检索的结果合并时，同一文档可能被多路召回：

```
语义检索：[doc_1, doc_3, doc_5]
关键词检索：[doc_2, doc_3, doc_6]
知识图谱：[doc_3, doc_7]
合并：[doc_1, doc_3, doc_5, doc_2, doc_3, doc_6, doc_3, doc_7]
去重后：[doc_1, doc_3, doc_5, doc_2, doc_6, doc_7]
```

doc_3 被三路都召回，去重只保留一次。

### 5.2 用 set 去重的问题

最直接的去重是用 `set`：

```python
seen = set()
for doc_id in all_ids:
    if doc_id not in seen:
        seen.add(doc_id)
        result.append(doc_id)
```

set 的问题：**每个元素存完整值**。Python 里一个 int 约 28 字节，100 万元素 = 28 MB。如果是字符串或长 ID，更费内存。

### 5.3 布隆过滤器的原理

**布隆过滤器**（Bloom Filter）：用**位数组 + k 个哈希函数**做概率判重。

**结构**：m 位的位数组（每个位置 1 bit）+ k 个哈希函数。

**add(x)**：把 x 的 k 个哈希位置都置 1。

```
add("doc_3"):
  h1("doc_3") % m = 5  →  bit[5] = 1
  h2("doc_3") % m = 17 →  bit[17] = 1
  h3("doc_3") % m = 42 →  bit[42] = 1
```

**contains(x)**：检查 x 的 k 个哈希位置是否全为 1。

```
contains("doc_3"):
  bit[5] == 1 ✓
  bit[17] == 1 ✓
  bit[42] == 1 ✓
  → 可能存在（返回 True）

contains("doc_99"):
  bit[3] == 0 ✗
  → 一定不存在（返回 False）
```

### 5.4 误判率

布隆过滤器的关键性质：

- **假阴性 = 0**：如果 contains 返回 False，则**一定**不存在。
- **假阳性 > 0**：如果 contains 返回 True，则**可能**存在（误判）。

误判率 p ≈ (1 - e^(-kn/m))^k，其中 n 是已插入元素数，m 是位数，k 是哈希数。

**给定 n 和期望误判率 p**，最优参数：

```
m = -n · ln(p) / (ln2)^2    （位数）
k = (m/n) · ln2             （哈希数）
```

例如 n=100 万、p=1%：m ≈ 9.58 Mbit = 1.2 MB，k = 7。对比 set 的 28 MB，省 23 倍。

### 5.5 双哈希法

实际实现不需要 k 个独立哈希函数，用**双哈希法**：

```
h_i(x) = h1(x) + i · h2(x)  (mod m),  i = 0, 1, ..., k-1
```

两个独立哈希 h1、h2 组合出 k 个位置。本 demo 用整数乘法做哈希（比 hashlib 快 50x+）：

```python
def _hashes(self, item):
    h1 = (item * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h2 = (item * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return [((h1 + i * h2) % self._m) for i in range(self._k)]
```

### 5.6 复杂度

- **空间**：O(m) bits，m = -n·ln(p)/(ln2)^2
- **add**：O(k)（k 次哈希 + 位设置）
- **contains**：O(k)（k 次哈希 + 位检查）

k 是常数（通常 3~10），所以 add 和 contains 都是 **O(1)**。

### 5.7 在 RAG 中的角色

多路检索结果合并时，用布隆去重：每个文档 ID 查布隆，没见过就加入结果并 add 到布隆，见过就跳过。O(1) 判重，内存比 set 省几十倍。

**误判的代价**：把没见过的文档误判为「见过」→ 漏掉一个文档。在 RAG 里这不是致命问题（多路召回本来就有冗余），且误判率可控（设 1% 就只有 1% 概率漏）。

---

## 6. 4 种数据结构如何协作

### 6.1 协作流程

```
用户 query："RAG 用了哪些数据结构？"
  │
  ├─→ [1] 语义检索（HNSW）
  │     query → Embedding → query_vec
  │     HNSW.search(query_vec, k=5)
  │     → [doc_12, doc_1, doc_16, doc_18, doc_3]
  │
  ├─→ [2] 关键词检索（倒排索引）
  │     query → 分词 → ["RAG", "数据结构"]
  │     InvertedIndex.search_or(["RAG", "数据结构"])
  │     → [doc_12, doc_6, doc_7, doc_8]
  │
  ├─→ [3] 知识图谱多跳（图）
  │     query → 抽实体 → ["RAG"]
  │     KG.multi_hop("RAG", hops=2)
  │     → RAG →uses→ 向量检索 →is_a→ 语义搜索 (doc_12, doc_14)
  │     → RAG →uses→ 倒排索引 →for→ 搜索引擎 (doc_12, doc_10)
  │     → RAG →uses→ 知识图谱 →uses→ 图 (doc_12, doc_13)
  │     → [doc_12, doc_14, doc_10, doc_13]
  │
  └─→ [4] 融合 + 布隆去重
        all = [12, 1, 16, 18, 3, 12, 6, 7, 8, 12, 14, 10, 13]
        bloom = BloomFilter(expected=13, fp=0.01)
        for id in all:
            if id not in bloom:
                bloom.add(id)
                result.append(id)
        → [12, 1, 16, 18, 3, 6, 7, 8, 14, 10, 13]
```

### 6.2 为什么缺一不可

| 去掉哪路 | 后果 |
|---|---|
| 去掉 HNSW | 「意思相近但用词不同」的文档漏掉（如 query 说「快」，文档写「迅速」） |
| 去掉倒排索引 | 精确匹配的专有名词漏掉（如 query 提「HNSW」，向量不够近） |
| 去掉知识图谱 | 多跳推理问题答不了（如「A 的导师的学生」） |
| 去掉布隆 | 多路召回同一文档重复出现，浪费 LLM context token |

### 6.3 各路的延迟分工

- **HNSW**：O(log n)，通常 1~10 ms（100 万向量）
- **倒排索引**：O(sum posting len)，稀有词 < 1 ms
- **知识图谱**：O(b^h)，2 跳通常 < 5 ms
- **布隆去重**：O(k) per item，< 1 ms

总延迟 = max(三路并行) + 去重 ≈ 10 ms 级别，远低于 LLM 生成延迟（数百 ms）。

### 6.4 融合策略

本 demo 用最简单的**去重合并**（所有路的并集）。生产级 RAG 用更精细的融合：

- **加权融合**：给每路打分（如语义检索的 cosine 分数 + 关键词的 BM25 分数）
- **重排**（rerank）：用 cross-encoder 对融合结果重新排序
- **RRF**（Reciprocal Rank Fusion）：按各路排名取倒数加权

---

## 7. 生产级实现对比

| 组件 | 本 demo | 生产级 |
|---|---|---|
| 向量检索 | 暴力 KNN | FAISS / Milvus / hnswlib / pgvector |
| 倒排索引 | 自建 dict + list | Elasticsearch / Lucene / Tantivy |
| 知识图谱 | 自建邻接表 | Neo4j / NebulaGraph / GraphRAG |
| 布隆过滤器 | 自建 bytearray | RedisBloom / pybloom-live / CuckooFilter |
| RAG 框架 | 自建 pipeline | LangChain / LlamaIndex / Haystack |

本 demo 用 200 行 Python 展示协作原理；生产级用 C++/Rust 实现，处理百万到亿级数据。

---

## 8. 小结

RAG 的 4 种数据结构各司其职：

1. **HNSW** 做语义检索——意思相近但用词不同
2. **倒排索引** 做关键词检索——精确匹配专有名词
3. **知识图谱** 做多跳推理——A 的导师的学生
4. **布隆过滤器** 做去重——O(1) 判重，省内存

它们协作的要点是：**各路检索互补**（覆盖不同的匹配模式）+ **布隆去重**（避免重复 + 省内存）。这就是生产级 RAG 的标准架构。