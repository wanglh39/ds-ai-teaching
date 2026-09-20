"""第 16 章 demo：RAG 与知识检索 × 数据结构协作（HNSW + 倒排索引 + 图 + 布隆过滤器）

扩展专题：一个 AI 应用场景（检索增强生成 RAG）→ 多种数据结构协作
- SimpleHNSW    ：简化 HNSW 向量检索（语义检索，找最相似的文档片段）
- InvertedIndex ：倒排索引（关键词检索，词 → 文档 ID 列表，支持 AND/OR）
- KnowledgeGraph：知识图谱（实体 + 关系，邻接表存储，支持多跳推理）
- BloomFilter   ：布隆过滤器（结果去重，O(1) 判重，可控误判率）

模拟完整 RAG pipeline：
  用户 query → 语义检索（HNSW）+ 关键词检索（倒排）+ 知识图谱多跳（图）
            → 融合结果 → 布隆去重 → 喂给 LLM 的上下文

性能对比：
1. 多路检索 vs 单路（召回率提升）
2. 布隆去重 vs 集合去重（内存 & 速度）
3. 倒排索引 vs 全文扫描（关键词检索延迟）

跑法：
    python 16_rag_retrieval/python/demo.py
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. SimpleHNSW：简化 HNSW 向量检索（语义检索）
# ============================================================================


@dataclass
class VectorDoc:
    """一个可被语义检索的文档片段，带向量。

    id     : 文档 ID
    text   : 文本内容
    vector : 语义向量（Embedding 模型生成，这里用随机模拟）
    """

    id: int
    text: str
    vector: tuple[float, ...]


class SimpleHNSW:
    """简化 HNSW 向量检索，用于语义检索。

    教学简化版：用暴力 KNN 代替完整 HNSW（重点是展示与其它数据结构的协作）。
    完整 HNSW 见第 12 章，这里只保留「建索引 + 查 top-k」的接口。

    - build(docs): 把文档向量存下来（暴力 KNN 不需要预建图）
    - search(query_vec, k): 返回 top-k 相似文档 ID（cosine 相似度）
    - 暴力做法：算 query 与所有文档的相似度，排序取前 k，O(n·d)

    生产级 HNSW 用近邻图 + 跳表入口点，把 O(n) 降到 O(log n)，
    但对 RAG 协作演示来说，暴力 KNN 已足够说明问题。
    """

    def __init__(self) -> None:
        self._docs: list[VectorDoc] = []
        self._id_to_idx: dict[int, int] = {}

    def build(self, docs: list[VectorDoc]) -> None:
        self._docs = docs
        self._id_to_idx = {d.id: i for i, d in enumerate(docs)}

    def __len__(self) -> int:
        return len(self._docs)

    @staticmethod
    def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def search(self, query_vec: tuple[float, ...], k: int = 5) -> list[tuple[int, float]]:
        """返回 top-k 相似文档，[(doc_id, score), ...]，按相似度降序。"""
        scored = [(d.id, self._cosine(query_vec, d.vector)) for d in self._docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# ============================================================================
# 2. InvertedIndex：倒排索引（关键词检索）
# ============================================================================


@dataclass
class TextDoc:
    """一个可被关键词检索的文档。"""

    id: int
    text: str
    tokens: list[str] = field(default_factory=list)


class InvertedIndex:
    """倒排索引，用于关键词检索。

    结构：词 → 文档 ID 列表（按 ID 升序，便于 AND/OR 合并）
    - build(docs): 分词 + 建倒排表
    - search_and(words): 所有词都出现的文档（AND，多路归并）
    - search_or(words): 任一词出现的文档（OR，并集）
    - search_phrase(phrase): 短语查询（连续 token 匹配）

    对比全文扫描：每次查询遍历所有文档做子串匹配，O(n·L)。
    倒排索引查询 O(sum(每个词的文档频次))，对稀有词极快。
    """

    def __init__(self) -> None:
        self._postings: dict[str, list[int]] = {}  # 词 → 文档 ID 列表
        self._doc_tokens: dict[int, list[str]] = {}  # 文档 ID → token 列表
        self._doc_text: dict[int, str] = {}
        self._n_docs: int = 0

    def build(self, docs: list[TextDoc]) -> None:
        self._n_docs = len(docs)
        for d in docs:
            toks = d.tokens if d.tokens else self._tokenize(d.text)
            self._doc_tokens[d.id] = toks
            self._doc_text[d.id] = d.text
            for tok in set(toks):  # 同一文档同一词只记一次
                self._postings.setdefault(tok, []).append(d.id)
        # 每个 posting list 排序，便于 AND 归并
        for word in self._postings:
            self._postings[word].sort()

    def __len__(self) -> int:
        return self._n_docs

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简易分词：英文按空格 + 去标点 + 小写；中文按 bigram（2-gram）。

        中文按 bigram：「排序算法」→ ['排序', '序算', '算法']
        这样查询「排序」能匹配到文档里的「排序」bigram。
        单字也保留，方便单字查询。
        """
        import re

        tokens: list[str] = []
        for part in re.split(r"[\s,.;!?，。；！？、]+", text):
            if not part:
                continue
            if all(ord(c) < 128 for c in part):
                tokens.append(part.lower())
            else:
                # 中文：单字 + bigram
                chars = list(part)
                tokens.extend(chars)
                for i in range(len(chars) - 1):
                    tokens.append(chars[i] + chars[i + 1])
        return tokens

    def search_and(self, words: list[str]) -> list[int]:
        """AND 查询：所有词都出现的文档 ID（多路归并 O(sum L)）。"""
        if not words:
            return []
        postings = [self._postings.get(w, []) for w in words]
        if any(len(p) == 0 for p in postings):
            return []
        # 多路归并求交集
        result = postings[0]
        for p in postings[1:]:
            result = self._intersect(result, p)
            if not result:
                return []
        return result

    def search_or(self, words: list[str]) -> list[int]:
        """OR 查询：任一词出现的文档 ID（并集，去重）。"""
        seen: set[int] = set()
        result: list[int] = []
        for w in words:
            for doc_id in self._postings.get(w, []):
                if doc_id not in seen:
                    seen.add(doc_id)
                    result.append(doc_id)
        result.sort()
        return result

    def search_phrase(self, phrase: str) -> list[int]:
        """短语查询：原文子串匹配（简化版，避免分词歧义）。"""
        if not phrase:
            return []
        result: list[int] = []
        for doc_id, text in self._doc_text.items():
            if phrase in text:
                result.append(doc_id)
        result.sort()
        return result

    @staticmethod
    def _intersect(a: list[int], b: list[int]) -> list[int]:
        """两个有序列表求交集（双指针 O(|a|+|b|)）。"""
        i = j = 0
        out: list[int] = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                out.append(a[i])
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    @staticmethod
    def _contains_subseq(seq: list[str], sub: list[str]) -> bool:
        """seq 中是否包含连续的 sub。"""
        n, m = len(seq), len(sub)
        for i in range(n - m + 1):
            if seq[i : i + m] == sub:
                return True
        return False


class FullTextScan:
    """全文扫描基准：每次查询遍历所有文档做子串匹配，O(n·L)。"""

    def __init__(self) -> None:
        self._docs: list[TextDoc] = []

    def build(self, docs: list[TextDoc]) -> None:
        self._docs = docs

    def __len__(self) -> int:
        return len(self._docs)

    def search_and(self, words: list[str]) -> list[int]:
        result: list[int] = []
        for d in self._docs:
            text_lower = d.text.lower()
            if all(w.lower() in text_lower for w in words):
                result.append(d.id)
        return result

    def search_or(self, words: list[str]) -> list[int]:
        result: list[int] = []
        for d in self._docs:
            text_lower = d.text.lower()
            if any(w.lower() in text_lower for w in words):
                result.append(d.id)
        return result


# ============================================================================
# 3. KnowledgeGraph：知识图谱（实体 + 关系，多跳推理）
# ============================================================================


class KnowledgeGraph:
    """知识图谱，用于多跳推理。

    结构：邻接表，entity → [(relation, target_entity, doc_id), ...]
    - add_entity(name): 加实体
    - add_relation(src, rel, dst, doc_id): 加关系（带来源文档 ID）
    - neighbors(entity): 直接邻居（1 跳）
    - multi_hop(entity, hops, visited): 多跳推理（BFS，避免环）
    - find_path(src, dst): 找 src 到 dst 的路径（BFS 最短路）

    RAG 中的用法：用户问「A 的导师是谁的学生」，单次语义/关键词检索都答不了，
    需要在知识图谱上走 2 跳：A →导师→ B →学生→ C。这就是 GraphRAG 的核心。
    """

    def __init__(self) -> None:
        self._adj: dict[str, list[tuple[str, str, int]]] = {}
        self._entities: set[str] = set()

    def add_entity(self, name: str) -> None:
        self._entities.add(name)
        self._adj.setdefault(name, [])

    def add_relation(self, src: str, rel: str, dst: str, doc_id: int = -1) -> None:
        self.add_entity(src)
        self.add_entity(dst)
        self._adj[src].append((rel, dst, doc_id))

    def neighbors(self, entity: str) -> list[tuple[str, str, int]]:
        """1 跳邻居：[(relation, target, doc_id), ...]。"""
        return list(self._adj.get(entity, []))

    def multi_hop(
        self, start: str, hops: int = 2,
    ) -> list[tuple[list[str], list[int]]]:
        """多跳推理（BFS），返回所有可达路径。

        Returns:
            [(path_entities, doc_ids), ...]
            path_entities: [start, e1, e2, ...] 实体序列
            doc_ids: 路径上经过的文档 ID（去重）
        """
        if start not in self._entities:
            return []
        results: list[tuple[list[str], list[int]]] = []
        # BFS：队列存 (当前实体, 路径实体列表, 路径文档 ID 集合)
        queue: list[tuple[str, list[str], frozenset[int]]] = [
            (start, [start], frozenset())
        ]
        visited_paths: set[tuple[str, ...]] = {(start,)}
        for _ in range(hops):
            next_queue: list[tuple[str, list[str], frozenset[int]]] = []
            for entity, path, doc_ids in queue:
                for rel, dst, doc_id in self._adj.get(entity, []):
                    new_path = path + [dst]
                    key = tuple(new_path)
                    if key in visited_paths:
                        continue
                    visited_paths.add(key)
                    new_doc_ids = doc_ids | ({doc_id} if doc_id >= 0 else set())
                    next_queue.append((dst, new_path, new_doc_ids))
                    if len(new_path) > 1:
                        results.append((new_path, sorted(new_doc_ids)))
            queue = next_queue
        return results

    def find_path(self, src: str, dst: str) -> list[str] | None:
        """BFS 找 src 到 dst 的最短路径。"""
        if src not in self._entities or dst not in self._entities:
            return None
        if src == dst:
            return [src]
        from collections import deque

        queue: deque[list[str]] = deque([[src]])
        visited: set[str] = {src}
        while queue:
            path = queue.popleft()
            entity = path[-1]
            for _, nxt, _ in self._adj.get(entity, []):
                if nxt == dst:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    @property
    def n_relations(self) -> int:
        return sum(len(v) for v in self._adj.values())


# ============================================================================
# 4. BloomFilter：布隆过滤器（结果去重，O(1) 判重）
# ============================================================================


class BloomFilter:
    """布隆过滤器，用于结果去重。

    原理：k 个哈希函数映射到 m 位的位数组。
    - add(x): 把 k 个位置都置 1
    - contains(x): k 个位置全为 1 → 可能存在；任一为 0 → 一定不存在

    误判率 p ≈ (1 - e^(-kn/m))^k。给定 n 和 p，最优 m = -n·ln(p) / (ln2)^2，
    k = (m/n)·ln2。

    RAG 中多路检索结果合并时，用布隆去重：
    - 语义检索召回 doc_1, doc_3, doc_5
    - 关键词检索召回 doc_2, doc_3, doc_6
    - 知识图谱召回 doc_3, doc_7
    - doc_3 被三路都召回，布隆去重只保留一次

    对比 set 去重：set 存完整 ID（4/8 字节），布隆每位 1 bit，
    大规模下省内存；但布隆有误判（把新的当成重复），可控。
    """

    def __init__(self, expected_items: int = 1000, fp_rate: float = 0.01) -> None:
        if expected_items <= 0:
            expected_items = 1
        self._n = expected_items
        self._p = fp_rate
        # 最优位数 m = -n*ln(p) / (ln2)^2
        self._m = max(8, int(-self._n * math.log(self._p) / (math.log(2) ** 2)))
        # 最优哈希数 k = (m/n)*ln2
        self._k = max(1, int((self._m / self._n) * math.log(2)))
        self._bits = bytearray((self._m + 7) // 8)
        self._count = 0

    @property
    def num_bits(self) -> int:
        return self._m

    @property
    def num_hashes(self) -> int:
        return self._k

    @property
    def capacity(self) -> int:
        return self._n

    def _hashes(self, item: int) -> list[int]:
        """k 个哈希位置（双哈希法：h_i = h1 + i*h2 mod m）。

        用 Python 内建 hash() + 种子做双哈希，比 hashlib 快 50x+。
        教学用：确定性够；生产级用 murmur3/xxhash。
        """
        h1 = (item * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        h2 = (item * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        if h2 == 0:
            h2 = 1
        return [((h1 + i * h2) % self._m) for i in range(self._k)]

    def add(self, item: int) -> None:
        for pos in self._hashes(item):
            self._bits[pos >> 3] |= 1 << (pos & 7)
        self._count += 1

    def contains(self, item: int) -> bool:
        for pos in self._hashes(item):
            if not (self._bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def __contains__(self, item: int) -> bool:
        return self.contains(item)

    @property
    def estimated_fp_rate(self) -> float:
        """理论误判率 (1 - e^(-kn/m))^k。"""
        return (1 - math.exp(-self._k * self._count / self._m)) ** self._k


# ============================================================================
# 5. RAG Pipeline：4 种数据结构协作
# ============================================================================


@dataclass
class RagDocument:
    """RAG 文档库中的一条文档。"""

    id: int
    text: str
    vector: tuple[float, ...]
    entities: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """单路检索结果。"""

    source: str  # "semantic" / "keyword" / "graph"
    doc_ids: list[int]
    latency_ms: float


@dataclass
class RagAnswer:
    """RAG pipeline 最终输出。"""

    query: str
    final_doc_ids: list[int]
    per_source: list[RetrievalResult]
    n_before_dedup: int
    n_after_dedup: int
    total_latency_ms: float


class RagPipeline:
    """完整 RAG pipeline：4 种数据结构协作。

    流程：
      query
        ├─→ [HNSW]      语义检索（query 向量化 → top-k 相似文档）
        ├─→ [倒排索引]  关键词检索（query 分词 → AND/OR 查询）
        ├─→ [知识图谱]  多跳推理（抽实体 → 邻居 → 相关文档）
        └─→ [布隆]      融合去重（多路结果合并，O(1) 判重）

    协作要点：
    - 语义检索解决「意思相近但用词不同」（如"快" vs "迅速"）
    - 关键词检索解决「精确匹配」（如专有名词、代码标识符）
    - 知识图谱解决「多跳推理」（如"A 的导师的学生"）
    - 布隆去重解决「多路召回同一文档」（O(1) 判重，省内存）
    """

    def __init__(
        self,
        hnsw: SimpleHNSW,
        inverted: InvertedIndex,
        kg: KnowledgeGraph,
        bloom: BloomFilter,
        embed_dim: int = 16,
    ) -> None:
        self.hnsw = hnsw
        self.inverted = inverted
        self.kg = kg
        self.bloom = bloom
        self.embed_dim = embed_dim

    def retrieve(
        self,
        query: str,
        query_vec: tuple[float, ...],
        query_words: list[str],
        query_entities: list[str],
        k: int = 5,
        hops: int = 2,
    ) -> RagAnswer:
        """执行一次 RAG 检索。

        Args:
            query: 原始查询文本
            query_vec: 查询向量（已 Embedding）
            query_words: 查询关键词（已分词）
            query_entities: 查询中提到的实体
            k: 每路 top-k
            hops: 知识图谱跳数
        """
        t_total = Timer()
        t_total.start()

        # 1. 语义检索（HNSW）
        t1 = Timer(); t1.start()
        sem_results = self.hnsw.search(query_vec, k=k)
        sem_ids = [doc_id for doc_id, _ in sem_results]
        t1.stop()
        r_sem = RetrievalResult("semantic", sem_ids, t1.elapsed_ms)

        # 2. 关键词检索（倒排索引，OR 查询更宽容）
        t2 = Timer(); t2.start()
        kw_ids = self.inverted.search_or(query_words)
        t2.stop()
        r_kw = RetrievalResult("keyword", kw_ids, t2.elapsed_ms)

        # 3. 知识图谱多跳（图）
        t3 = Timer(); t3.start()
        graph_ids: list[int] = []
        for entity in query_entities:
            paths = self.kg.multi_hop(entity, hops=hops)
            for _, doc_ids in paths:
                graph_ids.extend(doc_ids)
        t3.stop()
        r_graph = RetrievalResult("graph", graph_ids, t3.elapsed_ms)

        # 4. 融合 + 布隆去重
        all_ids = sem_ids + kw_ids + graph_ids
        n_before = len(all_ids)
        final_ids: list[int] = []
        # 重置布隆（每次查询独立去重）
        self.bloom = BloomFilter(
            expected_items=max(1, n_before), fp_rate=0.01,
        )
        for doc_id in all_ids:
            if doc_id not in self.bloom:
                self.bloom.add(doc_id)
                final_ids.append(doc_id)
        n_after = len(final_ids)

        t_total.stop()
        return RagAnswer(
            query=query,
            final_doc_ids=final_ids,
            per_source=[r_sem, r_kw, r_graph],
            n_before_dedup=n_before,
            n_after_dedup=n_after,
            total_latency_ms=t_total.elapsed_ms,
        )


# ============================================================================
# 6. 模拟数据 + 完整 RAG 演示
# ============================================================================


def _make_vector(seed: int, dim: int = 16) -> tuple[float, ...]:
    """用种子生成确定性向量（模拟 Embedding）。"""
    rng = random.Random(seed)
    return tuple(rng.gauss(0, 1) for _ in range(dim))


def build_demo_corpus(dim: int = 16) -> list[RagDocument]:
    """构建一个小的 RAG 文档库（教学用）。"""
    raw = [
        (0, "Python 是一种解释型高级编程语言，强调代码可读性", ["Python", "编程语言"]),
        (1, "Java 是一种面向对象的编译型语言，广泛用于企业开发", ["Java", "编程语言"]),
        (2, "C 语言是底层语言，常用于系统编程和嵌入式开发", ["C", "编程语言"]),
        (3, "快速排序是一种分治算法，平均时间复杂度 O(n log n)", ["快速排序", "算法"]),
        (4, "归并排序是稳定的分治排序算法，最坏 O(n log n)", ["归并排序", "算法"]),
        (5, "二分查找要求数组有序，时间复杂度 O(log n)", ["二分查找", "算法"]),
        (6, "哈希表用哈希函数把 key 映射到桶，平均 O(1) 查找", ["哈希表", "数据结构"]),
        (7, "跳表是多层索引链表，期望 O(log n) 搜索", ["跳表", "数据结构"]),
        (8, "图用邻接表或邻接矩阵存储，支持 BFS 和 DFS 遍历", ["图", "数据结构"]),
        (9, "HNSW 是近似最近邻搜索算法，用于向量检索", ["HNSW", "向量检索"]),
        (10, "倒排索引是搜索引擎的核心数据结构，词到文档列表", ["倒排索引", "搜索引擎"]),
        (11, "布隆过滤器用位数组和哈希函数做概率去重", ["布隆过滤器", "去重"]),
        (12, "RAG 是检索增强生成，先检索再生成", ["RAG", "检索增强生成"]),
        (13, "知识图谱用图存储实体关系，支持多跳推理", ["知识图谱", "多跳推理"]),
        (14, "向量检索是语义搜索的基础，找最相似的向量", ["向量检索", "语义搜索"]),
        (15, "Transformer 是大模型的基础架构，用自注意力机制", ["Transformer", "大模型"]),
        (16, "BERT 是基于 Transformer 的预训练语言模型", ["BERT", "预训练"]),
        (17, "GPT 是生成式预训练模型，用 decoder only 结构", ["GPT", "生成式"]),
        (18, "LangChain 是 LLM 应用框架，支持 RAG 和 agent", ["LangChain", "框架"]),
        (19, "LlamaIndex 是专注 RAG 的 LLM 数据框架", ["LlamaIndex", "RAG"]),
    ]
    docs: list[RagDocument] = []
    for doc_id, text, entities in raw:
        vec = _make_vector(doc_id * 7 + 1, dim)
        docs.append(RagDocument(id=doc_id, text=text, vector=vec, entities=entities))
    return docs


def build_demo_kg(docs: list[RagDocument]) -> KnowledgeGraph:
    """从文档库构建知识图谱（实体关系）。"""
    kg = KnowledgeGraph()
    # 手工加一些关系（教学示例）
    relations = [
        ("Python", "is_a", "编程语言", 0),
        ("Java", "is_a", "编程语言", 1),
        ("C", "is_a", "编程语言", 2),
        ("快速排序", "is_a", "算法", 3),
        ("归并排序", "is_a", "算法", 4),
        ("二分查找", "is_a", "算法", 5),
        ("哈希表", "is_a", "数据结构", 6),
        ("跳表", "is_a", "数据结构", 7),
        ("图", "is_a", "数据结构", 8),
        ("HNSW", "uses", "跳表", 9),
        ("HNSW", "uses", "图", 9),
        ("HNSW", "for", "向量检索", 9),
        ("RAG", "uses", "向量检索", 12),
        ("RAG", "uses", "倒排索引", 12),
        ("RAG", "uses", "知识图谱", 12),
        ("RAG", "uses", "布隆过滤器", 12),
        ("知识图谱", "uses", "图", 13),
        ("向量检索", "is_a", "语义搜索", 14),
        ("倒排索引", "for", "搜索引擎", 10),
        ("BERT", "uses", "Transformer", 16),
        ("GPT", "uses", "Transformer", 17),
        ("LangChain", "supports", "RAG", 18),
        ("LlamaIndex", "supports", "RAG", 19),
    ]
    for src, rel, dst, doc_id in relations:
        kg.add_relation(src, rel, dst, doc_id)
    return kg


def simulate_rag_query(
    pipeline: RagPipeline,
    query: str,
    query_vec: tuple[float, ...],
    query_words: list[str],
    query_entities: list[str],
) -> RagAnswer:
    """模拟一次 RAG 查询。"""
    return pipeline.retrieve(
        query=query,
        query_vec=query_vec,
        query_words=query_words,
        query_entities=query_entities,
    )


# ============================================================================
# 7. 性能对比
# ============================================================================


def benchmark_multi_vs_single(
    docs: list[RagDocument], kg: KnowledgeGraph, n_queries: int = 50,
) -> dict:
    """对比：多路检索 vs 单路（召回率）。

    场景：构造 50 个查询，每个查询有一个「正确文档集合」。
    - 单路（仅语义）：只用 HNSW
    - 多路（语义+关键词+图谱）：三路融合
    比较召回率 = 命中正确文档的查询数 / 总查询数。
    """
    rng = random.Random(42)
    dim = len(docs[0].vector) if docs else 16

    hnsw = SimpleHNSW()
    hnsw.build([VectorDoc(d.id, d.text, d.vector) for d in docs])
    inverted = InvertedIndex()
    inverted.build([TextDoc(d.id, d.text) for d in docs])

    # 构造查询：每个查询以某个文档为中心
    # 50% 的查询向量与目标相似（语义检索能命中）
    # 50% 的查询向量与目标不相似（语义检索 miss，靠关键词/图谱补）
    queries = []
    for i in range(n_queries):
        target = rng.choice(docs)
        if i % 2 == 0:
            # 相似查询：目标向量 + 小扰动
            noise = tuple(rng.gauss(0, 0.2) for _ in range(dim))
            qvec = tuple(t + n for t, n in zip(target.vector, noise))
        else:
            # 不相似查询：随机向量（语义检索大概率 miss）
            qvec = tuple(rng.gauss(0, 1) for _ in range(dim))
        # 查询词 = 目标文本里的几个词
        words = InvertedIndex._tokenize(target.text)[:4]
        # 查询实体 = 目标文档的实体
        entities = list(target.entities)
        queries.append((target.id, qvec, words, entities))

    # 单路（仅语义）
    single_hits = 0
    for target_id, qvec, _, _ in queries:
        results = hnsw.search(qvec, k=5)
        if target_id in [doc_id for doc_id, _ in results]:
            single_hits += 1

    # 多路（语义 + 关键词 + 图谱）
    multi_hits = 0
    bloom = BloomFilter(expected_items=100, fp_rate=0.01)
    for target_id, qvec, words, entities in queries:
        sem_ids = [doc_id for doc_id, _ in hnsw.search(qvec, k=5)]
        kw_ids = inverted.search_or(words)
        graph_ids: list[int] = []
        for e in entities:
            for _, doc_ids in kg.multi_hop(e, hops=2):
                graph_ids.extend(doc_ids)
        all_ids = sem_ids + kw_ids + graph_ids
        if target_id in all_ids:
            multi_hits += 1

    return {
        "n_queries": n_queries,
        "single_hits": single_hits,
        "multi_hits": multi_hits,
        "single_recall": single_hits / n_queries,
        "multi_recall": multi_hits / n_queries,
        "improvement": (multi_hits - single_hits) / n_queries,
    }


def benchmark_bloom_vs_set(
    n_items: int = 100000, n_lookups: int = 100000, seed: int = 42,
) -> dict:
    """对比：布隆去重 vs 集合去重（内存 & 速度）。

    场景：100000 个文档 ID 去重，再查 100000 次（一半存在一半不存在）。
    - 布隆：每位 1 bit，省内存；有误判
    - set：每个 int 28 字节（Python int 对象），费内存；零误判
    """
    rng = random.Random(seed)
    ids = list(range(n_items))

    # 布隆
    bloom = BloomFilter(expected_items=n_items, fp_rate=0.01)
    t_bloom_build = Timer(); t_bloom_build.start()
    for i in ids:
        bloom.add(i)
    t_bloom_build.stop()

    # 查询：一半存在一半不存在
    lookups = [rng.randint(0, 2 * n_items) for _ in range(n_lookups)]
    t_bloom_lookup = Timer(); t_bloom_lookup.start()
    bloom_fp = 0
    for x in lookups:
        if x in bloom:
            if x >= n_items:  # 不存在却判为存在 → 误判
                bloom_fp += 1
    t_bloom_lookup.stop()

    # set
    s: set[int] = set()
    t_set_build = Timer(); t_set_build.start()
    for i in ids:
        s.add(i)
    t_set_build.stop()

    t_set_lookup = Timer(); t_set_lookup.start()
    for x in lookups:
        _ = x in s
    t_set_lookup.stop()

    # 内存估算
    bloom_bytes = bloom.num_bits // 8
    set_bytes = n_items * 28  # Python int 约 28 字节 + set 开销

    return {
        "n_items": n_items,
        "n_lookups": n_lookups,
        "bloom_build_ms": t_bloom_build.elapsed_ms,
        "bloom_lookup_ms": t_bloom_lookup.elapsed_ms,
        "bloom_bytes": bloom_bytes,
        "bloom_fp": bloom_fp,
        "bloom_fp_rate": bloom_fp / n_lookups,
        "set_build_ms": t_set_build.elapsed_ms,
        "set_lookup_ms": t_set_lookup.elapsed_ms,
        "set_bytes": set_bytes,
        "memory_ratio": set_bytes / bloom_bytes if bloom_bytes > 0 else 0,
        "lookup_speedup": t_set_lookup.elapsed_ms / t_bloom_lookup.elapsed_ms
        if t_bloom_lookup.elapsed_ms > 0 else float("inf"),
    }


def benchmark_inverted_vs_fullscan(
    docs: list[TextDoc], n_queries: int = 1000, seed: int = 42,
) -> dict:
    """对比：倒排索引 vs 全文扫描（关键词检索延迟）。

    场景：n_queries 次 AND 查询，每次 2 个词。
    - 倒排：O(sum posting list len)
    - 全文：O(n·L) 每次扫所有文档
    """
    rng = random.Random(seed)
    inverted = InvertedIndex()
    inverted.build(docs)
    fullscan = FullTextScan()
    fullscan.build(docs)

    # 构造查询词对
    all_words = list(inverted._postings.keys())
    if len(all_words) < 2:
        return {"n_queries": 0}
    query_pairs = [
        (rng.choice(all_words), rng.choice(all_words))
        for _ in range(n_queries)
    ]

    t_inverted = Timer(); t_inverted.start()
    for w1, w2 in query_pairs:
        inverted.search_and([w1, w2])
    t_inverted.stop()

    t_fullscan = Timer(); t_fullscan.start()
    for w1, w2 in query_pairs:
        fullscan.search_and([w1, w2])
    t_fullscan.stop()

    return {
        "n_docs": len(docs),
        "n_queries": n_queries,
        "inverted_ms": t_inverted.elapsed_ms,
        "fullscan_ms": t_fullscan.elapsed_ms,
        "speedup": t_fullscan.elapsed_ms / t_inverted.elapsed_ms
        if t_inverted.elapsed_ms > 0 else float("inf"),
    }


# ============================================================================
# 8. main
# ============================================================================


def main() -> None:
    print("=" * 76)
    print("第 16 章 demo：RAG 与知识检索 × 数据结构协作")
    print("（HNSW + 倒排索引 + 知识图谱 + 布隆过滤器）")
    print("=" * 76)

    # ------------------------------------------------------------------------
    print("\n[1] 语义检索：SimpleHNSW（向量 top-k 相似）")
    print("-" * 76)
    docs = build_demo_corpus(dim=16)
    hnsw = SimpleHNSW()
    hnsw.build([VectorDoc(d.id, d.text, d.vector) for d in docs])
    qvec = _make_vector(seed=3 * 7 + 1, dim=16)  # 用 doc 3 的向量查询
    results = hnsw.search(qvec, k=3)
    print(f"  文档库大小 = {len(hnsw)}")
    print(f"  查询向量 = doc 3 的向量（找最相似的 3 个）")
    for doc_id, score in results:
        print(f"    doc {doc_id}: cosine = {score:.4f}  「{docs[doc_id].text[:30]}...」")
    print(f"  → HNSW 解决「意思相近」的语义检索")

    # ------------------------------------------------------------------------
    print("\n[2] 关键词检索：InvertedIndex（词 → 文档列表）")
    print("-" * 76)
    inverted = InvertedIndex()
    inverted.build([TextDoc(d.id, d.text) for d in docs])
    print(f"  索引大小 = {len(inverted)} 文档, {len(inverted._postings)} 个词项")
    and_res = inverted.search_and(["排序", "算法"])
    or_res = inverted.search_or(["排序", "算法"])
    print(f"  AND(['排序','算法']) = {and_res}  （两个词都出现）")
    print(f"  OR(['排序','算法'])  = {or_res}  （任一词出现）")
    phrase_res = inverted.search_phrase("向量检索")
    print(f"  phrase('向量检索')   = {phrase_res}  （连续匹配）")
    print(f"  注：中文按 bigram 分词，'排序' 是一个 bigram")
    print(f"  → 倒排索引解决「精确匹配」的关键词检索")

    # ------------------------------------------------------------------------
    print("\n[3] 知识图谱多跳：KnowledgeGraph（实体关系，BFS 多跳）")
    print("-" * 76)
    kg = build_demo_kg(docs)
    print(f"  实体数 = {kg.n_entities}, 关系数 = {kg.n_relations}")
    print(f"  RAG 的 1 跳邻居：")
    for rel, dst, doc_id in kg.neighbors("RAG"):
        print(f"    RAG --{rel}--> {dst}  (来自 doc {doc_id})")
    print(f"  RAG 的 2 跳推理（多跳路径）：")
    paths = kg.multi_hop("RAG", hops=2)
    for path, doc_ids in paths[:8]:
        print(f"    {' → '.join(path)}  (docs: {doc_ids})")
    if len(paths) > 8:
        print(f"    ... 共 {len(paths)} 条 2 跳路径")
    path = kg.find_path("BERT", "Transformer")
    print(f"  BERT → Transformer 最短路：{path}")
    print(f"  → 知识图谱解决「A 的导师的学生」这类多跳推理")

    # ------------------------------------------------------------------------
    print("\n[4] 布隆去重：BloomFilter（O(1) 判重，可控误判率）")
    print("-" * 76)
    bloom = BloomFilter(expected_items=100, fp_rate=0.01)
    print(f"  位数组 = {bloom.num_bits} bits ({bloom.num_bits // 8} bytes), "
          f"哈希数 = {bloom.num_hashes}")
    for i in [0, 1, 2, 3, 5, 8, 13]:
        bloom.add(i)
    print(f"  加入 7 个 ID 后，理论误判率 = {bloom.estimated_fp_rate:.4%}")
    for x in [0, 1, 7, 13, 100]:
        print(f"    contains({x}) = {x in bloom}")
    print(f"  → 布隆过滤器用 k 个哈希 + m 位的位数组，O(1) 判重")

    # ------------------------------------------------------------------------
    print("\n[5] 完整 RAG pipeline：4 种数据结构协作")
    print("-" * 76)
    pipeline = RagPipeline(hnsw, inverted, kg, bloom, embed_dim=16)
    ans = simulate_rag_query(
        pipeline,
        query="RAG 用了哪些数据结构？",
        query_vec=_make_vector(seed=12 * 7 + 1, dim=16),  # doc 12 是 RAG
        query_words=["RAG", "数据结构"],
        query_entities=["RAG"],
    )
    print(f"  查询：「{ans.query}」")
    print(f"  各路检索：")
    for r in ans.per_source:
        print(f"    [{r.source:>8}] {r.latency_ms:.3f} ms, 召回 {len(r.doc_ids)} 个: {r.doc_ids}")
    print(f"  融合前 = {ans.n_before_dedup} 个（含重复）")
    print(f"  布隆去重后 = {ans.n_after_dedup} 个: {ans.final_doc_ids}")
    print(f"  总延迟 = {ans.total_latency_ms:.3f} ms")
    print(f"  → 语义 + 关键词 + 图谱 三路融合，布隆 O(1) 去重")

    # ------------------------------------------------------------------------
    print("\n[6] 性能对比：多路检索 vs 单路（召回率）")
    print("-" * 76)
    res_multi = benchmark_multi_vs_single(docs, kg, n_queries=50)
    print(f"  场景：{res_multi['n_queries']} 次查询")
    print(f"  单路（仅语义）  ：命中 {res_multi['single_hits']}, "
          f"召回率 {res_multi['single_recall'] * 100:.1f}%")
    print(f"  多路（语义+关键词+图谱）：命中 {res_multi['multi_hits']}, "
          f"召回率 {res_multi['multi_recall'] * 100:.1f}%")
    print(f"  → 多路融合召回率提升 {res_multi['improvement'] * 100:.1f} 个百分点")

    # ------------------------------------------------------------------------
    print("\n[7] 性能对比：布隆去重 vs 集合去重（内存 & 速度）")
    print("-" * 76)
    res_bloom = benchmark_bloom_vs_set(n_items=100000, n_lookups=100000)
    print(f"  场景：{res_bloom['n_items']} 个 ID 去重，"
          f"{res_bloom['n_lookups']} 次查询")
    print(f"  布隆    ：build {res_bloom['bloom_build_ms']:.1f} ms, "
          f"lookup {res_bloom['bloom_lookup_ms']:.1f} ms, "
          f"内存 {res_bloom['bloom_bytes'] / 1024:.1f} KB, "
          f"误判率 {res_bloom['bloom_fp_rate'] * 100:.3f}%")
    print(f"  set     ：build {res_bloom['set_build_ms']:.1f} ms, "
          f"lookup {res_bloom['set_lookup_ms']:.1f} ms, "
          f"内存 {res_bloom['set_bytes'] / 1024:.1f} KB")
    print(f"  → 布隆内存省 {res_bloom['memory_ratio']:.1f}x, "
          f"lookup 快 {res_bloom['lookup_speedup']:.2f}x, "
          f"代价是 {res_bloom['bloom_fp_rate'] * 100:.3f}% 误判")

    # ------------------------------------------------------------------------
    print("\n[8] 性能对比：倒排索引 vs 全文扫描（关键词检索）")
    print("-" * 76)
    # 用更大的文档库测延迟差
    big_docs = []
    rng = random.Random(7)
    words_pool = ["算法", "数据", "结构", "检索", "排序", "查找", "图", "树", "哈希", "向量"]
    for i in range(2000):
        toks = [rng.choice(words_pool) for _ in range(rng.randint(5, 15))]
        big_docs.append(TextDoc(id=i, text=" ".join(toks), tokens=toks))
    res_inv = benchmark_inverted_vs_fullscan(big_docs, n_queries=1000)
    print(f"  场景：{res_inv['n_docs']} 文档，{res_inv['n_queries']} 次 AND 查询")
    print(f"  倒排索引 ：{res_inv['inverted_ms']:.2f} ms")
    print(f"  全文扫描 ：{res_inv['fullscan_ms']:.2f} ms")
    print(f"  → 倒排索引快 {res_inv['speedup']:.1f}x")

    # ------------------------------------------------------------------------
    print("\n[9] 保存对比图到 figures/")
    print("-" * 76)
    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    # 图 1：多路 vs 单路 召回率
    save_bar(
        {
            "单路（仅语义）": res_multi["single_recall"] * 100,
            "多路融合": res_multi["multi_recall"] * 100,
        },
        fig_dir / "recall_multi_vs_single.png",
        title=f"多路检索 vs 单路 召回率（{res_multi['n_queries']} 次查询）",
        ylabel="召回率 (%)",
        baseline="单路（仅语义）",
    )
    print(f"  ✓ 保存 figures/recall_multi_vs_single.png")

    # 图 2：布隆 vs set 内存
    save_bar(
        {
            "布隆过滤器": res_bloom["bloom_bytes"] / 1024,
            "set": res_bloom["set_bytes"] / 1024,
        },
        fig_dir / "memory_bloom_vs_set.png",
        title=f"去重内存对比（{res_bloom['n_items']} 个 ID）",
        ylabel="内存 (KB)",
        baseline="set",
    )
    print(f"  ✓ 保存 figures/memory_bloom_vs_set.png")

    # 图 3：布隆 vs set 查询延迟
    save_bar(
        {
            "布隆过滤器": res_bloom["bloom_lookup_ms"],
            "set": res_bloom["set_lookup_ms"],
        },
        fig_dir / "lookup_bloom_vs_set.png",
        title=f"去重查询延迟（{res_bloom['n_lookups']} 次查询）",
        ylabel="延迟 (ms)",
        baseline="set",
    )
    print(f"  ✓ 保存 figures/lookup_bloom_vs_set.png")

    # 图 4：倒排 vs 全文 延迟
    save_bar(
        {
            "倒排索引": res_inv["inverted_ms"],
            "全文扫描": res_inv["fullscan_ms"],
        },
        fig_dir / "latency_inverted_vs_fullscan.png",
        title=f"关键词检索延迟（{res_inv['n_docs']} 文档, {res_inv['n_queries']} 查询）",
        ylabel="延迟 (ms)",
        baseline="全文扫描",
    )
    print(f"  ✓ 保存 figures/latency_inverted_vs_fullscan.png")

    # 图 5：布隆误判率随元素数变化
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    fp_curve: list[float] = []
    for n in sizes:
        bf = BloomFilter(expected_items=n, fp_rate=0.01)
        for i in range(n):
            bf.add(i)
        # 查 10000 个不存在的
        fp = 0
        for x in range(n, n + 10000):
            if x in bf:
                fp += 1
        fp_curve.append(fp / 10000 * 100)
    save_line(
        {"实测误判率(%)": fp_curve},
        fig_dir / "bloom_fp_rate_over_size.png",
        title="布隆过滤器误判率随元素数变化（设计 p=1%）",
        ylabel="误判率 (%)",
        xlabel="元素数 (100 / 500 / ... / 50000)",
    )
    print(f"  ✓ 保存 figures/bloom_fp_rate_over_size.png")

    # 图 6：RAG 各路检索延迟
    latencies = {r.source: r.latency_ms for r in ans.per_source}
    save_bar(
        latencies,
        fig_dir / "rag_per_source_latency.png",
        title="RAG 各路检索延迟",
        ylabel="延迟 (ms)",
    )
    print(f"  ✓ 保存 figures/rag_per_source_latency.png")

    # ------------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("结论：RAG 是 4 种数据结构的协奏——")
    print("  1. 语义检索（HNSW）解决「意思相近但用词不同」")
    print("  2. 关键词检索（倒排索引）解决「精确匹配专有名词」")
    print("  3. 知识图谱（图）解决「A 的导师的学生」多跳推理")
    print("  4. 布隆去重（布隆过滤器）解决「多路召回同一文档」O(1) 判重")
    print("  多路融合召回率显著高于单路；布隆省内存几十倍（代价是可控误判）；")
    print("  倒排索引比全文扫描快数倍；Python 里 set 是 C 实现所以 lookup 快，")
    print("  布隆优势在内存 + 跨语言/跨进程场景（C++/Rust 实现下 lookup 同样 O(1)）。")
    print("=" * 76)


if __name__ == "__main__":
    main()