"""第 07 章 demo：BPE 分词（Trie 最长前缀匹配 vs 朴素哈希查找）。

展示「Trie / 前缀树」在 AI 里的核心应用：BPE（Byte Pair Encoding）分词。
1. 用真实 BPE 训练算法（统计高频 pair 合并）从语料学出子词词表
2. 三种分词实现对比：
   - trie_bpe_tokenize：Trie 最长前缀匹配，O(n * L)，沿 Trie 一次遍历
   - hash_bpe_tokenize：长度分桶哈希，O(n * L)，每个位置切子串查哈希
   - naive_bpe_tokenize：遍历所有子词，O(n * V)，朴素基线（为什么不用哈希表做最长匹配）
3. 验证三种方法分词结果一致
4. 对比不同文本长度和词表大小下的分词速度
5. 内存对比：Trie 前缀共享省了多少节点

跑法：
    python 07_trie/python/demo.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


SAMPLE_CORPUS = (
    "the quick brown fox jumps over the lazy dog and the dog was not amused "
    "natural language processing is a subfield of linguistics and artificial "
    "intelligence and machine learning and deep learning models process text "
    "tokenization is the first step in any nlp pipeline where text is split "
    "into smaller units called tokens which can be words subwords or characters "
    "byte pair encoding is a simple and effective subword tokenization algorithm "
    "that iteratively merges the most frequent pair of adjacent tokens in the "
    "corpus to build a vocabulary of subword units this vocabulary is then used "
    "to tokenize new text by longest prefix matching which is exactly what a "
    "trie does best the transformer architecture uses self attention mechanisms "
    "to process sequences of tokens and large language models like gpt and llama "
    "use byte pair encoding or sentencepiece for tokenization before training "
    "the idea of sharing prefixes across words is fundamental to the trie data "
    "structure and it saves memory because common prefixes like the and ing and "
    "tion are stored only once in the tree rather than repeated in every word "
    "this is why modern tokenizers prefer trie based implementations over hash "
    "tables when doing longest prefix matching for subword tokenization at scale "
    "we will compare the trie approach with naive hash lookup and bucket hash "
    "lookup to understand the performance tradeoffs in real tokenization tasks "
    "the research community has found that efficient tokenization is critical "
    "for both training speed and inference latency in large language model systems "
)


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False


class Trie:
    """前缀树：支持 insert / search / starts_with / longest_prefix_match。

    用 dict 存子节点（支持任意字符，不限 26 字母），适配 BPE 子词。
    """

    def __init__(self) -> None:
        self.root: TrieNode = TrieNode()
        self.word_count: int = 0
        self.node_count: int = 1

    def insert(self, word: str) -> None:
        cur = self.root
        for ch in word:
            nxt = cur.children.get(ch)
            if nxt is None:
                nxt = TrieNode()
                cur.children[ch] = nxt
                self.node_count += 1
            cur = nxt
        if not cur.is_end:
            cur.is_end = True
            self.word_count += 1

    def search(self, word: str) -> bool:
        cur = self.root
        for ch in word:
            nxt = cur.children.get(ch)
            if nxt is None:
                return False
            cur = nxt
        return cur.is_end

    def starts_with(self, prefix: str) -> bool:
        cur = self.root
        for ch in prefix:
            nxt = cur.children.get(ch)
            if nxt is None:
                return False
            cur = nxt
        return True


def build_bpe_vocab(corpus: str, num_merges: int) -> set[str]:
    """用真实 BPE 训练算法从语料学出子词词表。

    1. 初始词表 = 语料里所有单字符
    2. 把语料拆成单词，每个单词表示成字符序列，统计单词频率
    3. 重复 num_merges 次：找当前最高频的相邻字符对，合并成一个新子词加入词表，
       并在所有单词里把该 pair 替换成合并后的子词
    """
    vocab: set[str] = set()
    cleaned_words: list[list[str]] = []
    word_counts: Counter[tuple[str, ...]] = Counter()
    for w in corpus.lower().split():
        chars = [c for c in w if c.isalpha()]
        if chars:
            word_counts[tuple(chars)] += 1
            for c in chars:
                vocab.add(c)

    for _ in range(num_merges):
        pair_counts: Counter[tuple[str, str]] = Counter()
        for word, cnt in word_counts.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += cnt
        if not pair_counts:
            break
        best_pair, _ = pair_counts.most_common(1)[0]
        merged = best_pair[0] + best_pair[1]
        vocab.add(merged)
        new_word_counts: Counter[tuple[str, ...]] = Counter()
        a, b = best_pair
        for word, cnt in word_counts.items():
            new_word: list[str] = []
            i = 0
            wl = len(word)
            while i < wl:
                if i < wl - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] += cnt
        word_counts = new_word_counts

    return vocab


def build_trie(vocab: set[str]) -> Trie:
    trie = Trie()
    for w in vocab:
        trie.insert(w)
    return trie


def build_length_buckets(vocab: set[str]) -> dict[int, set[str]]:
    buckets: dict[int, set[str]] = {}
    for w in vocab:
        buckets.setdefault(len(w), set()).add(w)
    return buckets


def trie_bpe_tokenize(text: str, trie: Trie) -> list[str]:
    """Trie 最长前缀匹配分词，O(n * L)。

    从位置 i 开始沿 Trie 走，记录最后一个 is_end 的位置，
    走到不能走为止，切出最长匹配的子词。
    """
    tokens: list[str] = []
    i = 0
    n = len(text)
    root = trie.root
    while i < n:
        cur = root
        last_end = i
        j = i
        while j < n:
            nxt = cur.children.get(text[j])
            if nxt is None:
                break
            cur = nxt
            j += 1
            if cur.is_end:
                last_end = j
        if last_end == i:
            tokens.append(text[i])
            i += 1
        else:
            tokens.append(text[i:last_end])
            i = last_end
    return tokens


def hash_bpe_tokenize(text: str, buckets: dict[int, set[str]], max_len: int) -> list[str]:
    """长度分桶哈希分词，O(n * L)。

    每个位置从最长子词长度往短试，切出 text[i:i+l] 查对应长度桶的哈希集合。
    和 Trie 同阶，但每个位置要切 L 次子串 + L 次哈希查，无前缀共享。
    """
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        best_len = 0
        limit = min(max_len, n - i)
        for l in range(limit, 0, -1):
            bucket = buckets.get(l)
            if bucket is not None and text[i : i + l] in bucket:
                best_len = l
                break
        if best_len == 0:
            tokens.append(text[i])
            i += 1
        else:
            tokens.append(text[i : i + best_len])
            i += best_len
    return tokens


def naive_bpe_tokenize(text: str, vocab: set[str]) -> list[str]:
    """朴素法：每个位置遍历所有子词找最长匹配，O(n * V)。

    这就是「为什么不用哈希表做最长前缀匹配」——要找最长匹配，
    必须把词表里每个子词都试一遍，V 大时极慢。
    """
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        best_len = 0
        best_word = ""
        for word in vocab:
            wl = len(word)
            if wl > best_len and text.startswith(word, i):
                best_len = wl
                best_word = word
        if best_len == 0:
            tokens.append(text[i])
            i += 1
        else:
            tokens.append(best_word)
            i += best_len
    return tokens


def make_text(length: int) -> str:
    """生成指定长度的英文文本（重复语料）。"""
    base = SAMPLE_CORPUS.lower()
    repeats = (length // len(base)) + 1
    text = (base + " ") * repeats
    return text[:length]


def bench_tokenize(
    text: str,
    trie: Trie,
    buckets: dict[int, set[str]],
    vocab: set[str],
    max_len: int,
    include_naive: bool = True,
    repeat: int = 5,
) -> dict[str, dict[str, float]]:
    cases: dict[str, object] = {
        "trie O(nL)": lambda: trie_bpe_tokenize(text, trie),
        "hash-bucket O(nL)": lambda: hash_bpe_tokenize(text, buckets, max_len),
    }
    if include_naive:
        cases["naive O(nV)"] = lambda: naive_bpe_tokenize(text, vocab)
    return compare(cases, repeat=repeat, warmup=1)


def main() -> None:
    print("=" * 72)
    print("第 07 章 demo：BPE 分词（Trie 最长前缀匹配 vs 朴素哈希查找）")
    print("=" * 72)

    print("\n[1] 训练 BPE 词表（真实 BPE 算法：高频 pair 合并）")
    print("-" * 72)
    vocab_small = build_bpe_vocab(SAMPLE_CORPUS, num_merges=300)
    vocab_mid = build_bpe_vocab(SAMPLE_CORPUS, num_merges=800)
    vocab_large = build_bpe_vocab(SAMPLE_CORPUS, num_merges=2000)
    max_len_small = max(len(w) for w in vocab_small)
    max_len_mid = max(len(w) for w in vocab_mid)
    max_len_large = max(len(w) for w in vocab_large)
    total_chars_mid = sum(len(w) for w in vocab_mid)
    print(f"  词表大小: small={len(vocab_small)} (max_len={max_len_small}), "
          f"mid={len(vocab_mid)} (max_len={max_len_mid}), "
          f"large={len(vocab_large)} (max_len={max_len_large})")
    print(f"  mid 词表总字符数（独立存储）= {total_chars_mid}")

    print("\n[2] 正确性验证：三种方法分词结果一致")
    print("-" * 72)
    trie_mid = build_trie(vocab_mid)
    buckets_mid = build_length_buckets(vocab_mid)
    demo_text = "tokenizationisthefirststepandthetriehelpswithlongestprefixmatching"
    t_trie = trie_bpe_tokenize(demo_text, trie_mid)
    t_hash = hash_bpe_tokenize(demo_text, buckets_mid, max_len_mid)
    t_naive = naive_bpe_tokenize(demo_text, vocab_mid)
    print(f"  text   = {demo_text}")
    print(f"  trie   = {t_trie}")
    print(f"  hash   = {t_hash}")
    print(f"  naive  = {t_naive}")
    assert t_trie == t_hash == t_naive, "三种分词结果不一致"
    print(f"  ✓ 三种方法结果一致：{len(t_trie)} 个 token")

    print("\n[3] 内存对比：Trie 前缀共享 vs 独立存储")
    print("-" * 72)
    for name, vocab in [("small", vocab_small), ("mid", vocab_mid), ("large", vocab_large)]:
        trie = build_trie(vocab)
        total_chars = sum(len(w) for w in vocab)
        nodes = trie.node_count
        saved = total_chars - nodes
        ratio = nodes / total_chars if total_chars else 0
        print(f"  {name:>5}: 词表={len(vocab):>5} 子词, 总字符={total_chars:>6}, "
              f"Trie 节点={nodes:>6}, 节省={saved:>6} ({(1 - ratio) * 100:.1f}%), "
              f"节点/总字符={ratio:.3f}")

    print("\n[4] 性能对比（含朴素 O(nV)）：词表=small, 不同文本长度")
    print("-" * 72)
    trie_s = build_trie(vocab_small)
    buckets_s = build_length_buckets(vocab_small)
    naive_lengths = [2000, 5000]
    naive_trie_means: list[float] = []
    naive_hash_means: list[float] = []
    naive_naive_means: list[float] = []
    for length in naive_lengths:
        text = make_text(length)
        print(f"\n  |vocab|={len(vocab_small)}, text_len={length}, max_len={max_len_small}")
        res = bench_tokenize(text, trie_s, buckets_s, vocab_small, max_len_small,
                             include_naive=True, repeat=3)
        naive_trie_means.append(res["trie O(nL)"]["mean_ms"])
        naive_hash_means.append(res["hash-bucket O(nL)"]["mean_ms"])
        naive_naive_means.append(res["naive O(nV)"]["mean_ms"])
        print(format_table(res, baseline="naive O(nV)"))
        sp = res["naive O(nV)"]["mean_ms"] / res["trie O(nL)"]["mean_ms"]
        print(f"    → Trie 比朴素快 {sp:.1f}x（O(nL) vs O(nV)，L={max_len_small} vs V={len(vocab_small)}）")

    print("\n[5] 性能对比（Trie vs hash-bucket，大规模）：不同词表和文本长度")
    print("-" * 72)
    configs = [
        ("mid", vocab_mid, max_len_mid, [10000, 50000]),
        ("large", vocab_large, max_len_large, [10000, 50000]),
    ]
    big_trie_means: list[float] = []
    big_hash_means: list[float] = []
    big_labels: list[str] = []
    for vname, vocab, max_len, lengths in configs:
        trie = build_trie(vocab)
        buckets = build_length_buckets(vocab)
        for length in lengths:
            text = make_text(length)
            print(f"\n  |vocab|={len(vocab)} ({vname}), text_len={length}, max_len={max_len}")
            res = bench_tokenize(text, trie, buckets, vocab, max_len,
                                 include_naive=False, repeat=5)
            big_trie_means.append(res["trie O(nL)"]["mean_ms"])
            big_hash_means.append(res["hash-bucket O(nL)"]["mean_ms"])
            big_labels.append(f"{vname}\nL={length}")
            print(format_table(res, baseline="hash-bucket O(nL)"))
            sp = res["hash-bucket O(nL)"]["mean_ms"] / res["trie O(nL)"]["mean_ms"]
            print(f"    → Trie 比 hash-bucket 快 {sp:.2f}x（同阶 O(nL)，前缀共享 + 单次遍历优势）")

    print("\n[6] 关键数字：词表=mid, text_len=50000")
    print("-" * 72)
    trie_m = build_trie(vocab_mid)
    buckets_m = build_length_buckets(vocab_mid)
    text_key = make_text(50000)
    res_key = bench_tokenize(text_key, trie_m, buckets_m, vocab_mid, max_len_mid,
                             include_naive=False, repeat=5)
    t_trie_k = res_key["trie O(nL)"]["mean_ms"]
    t_hash_k = res_key["hash-bucket O(nL)"]["mean_ms"]
    total_chars_k = sum(len(w) for w in vocab_mid)
    print(f"  Trie 最长匹配     : {t_trie_k:.3f} ms  (O(n*L), L={max_len_mid})")
    print(f"  hash-bucket 匹配  : {t_hash_k:.3f} ms  (O(n*L), 切子串+哈希查)")
    print(f"  hash/Trie = {t_hash_k / t_trie_k:.2f}x")
    print(f"  词表={len(vocab_mid)}, 总字符={total_chars_k}, "
          f"Trie 节点={trie_m.node_count}, "
          f"前缀共享节省 {(1 - trie_m.node_count / total_chars_k) * 100:.1f}% 内存")
    print(f"  朴素 O(n*V) 在此规模约需 {t_trie_k * len(vocab_mid) / max_len_mid / 1000:.1f} s（外推，太慢不实测）")

    fig_dir = Path(__file__).resolve().parent.parent / "figures"

    save_bar(
        {
            "Trie O(nL)": naive_trie_means[0],
            "hash-bucket O(nL)": naive_hash_means[0],
            "naive O(nV)": naive_naive_means[0],
        },
        fig_dir / "bpe_compare_vocab_small_len2000.png",
        title=f"BPE 分词：|vocab|={len(vocab_small)}, len=2000（Trie vs hash vs naive）",
        ylabel="耗时 (ms)",
        baseline="naive O(nV)",
    )

    save_line(
        {
            "Trie O(nL)": naive_trie_means,
            "hash-bucket O(nL)": naive_hash_means,
            "naive O(nV)": naive_naive_means,
        },
        fig_dir / "bpe_vs_text_len_small.png",
        title=f"BPE 分词耗时 vs 文本长度（|vocab|={len(vocab_small)}）",
        ylabel="耗时 (ms)",
        xlabel="文本长度 (2000 / 5000)",
    )

    save_bar(
        {
            "Trie O(nL)": t_trie_k,
            "hash-bucket O(nL)": t_hash_k,
        },
        fig_dir / "bpe_trie_vs_hash_large.png",
        title=f"BPE 分词：|vocab|={len(vocab_mid)}, len=50000（Trie vs hash）",
        ylabel="耗时 (ms)",
        baseline="hash-bucket O(nL)",
    )

    save_line(
        {
            "Trie O(nL)": big_trie_means,
            "hash-bucket O(nL)": big_hash_means,
        },
        fig_dir / "bpe_trie_vs_hash_vs_scale.png",
        title="BPE 分词：Trie vs hash-bucket（不同词表/文本规模）",
        ylabel="耗时 (ms)",
        xlabel="规模 (mid/large × len 10k/50k)",
    )

    print(f"\n图已保存到: {fig_dir}")
    print("\n结论：")
    print(f"  - 朴素哈希 O(n*V) 最慢：每个位置遍历全部子词，V 大时不可行")
    print(f"  - Trie 和 hash-bucket 同为 O(n*L)，但 Trie 因前缀共享 + 单次遍历更快")
    print(f"  - Trie 前缀共享节省约 {(1 - trie_m.node_count / total_chars_k) * 100:.0f}% 存储（节点数 < 总字符数）")
    print(f"  - BPE 分词的最长前缀匹配天然适配 Trie：沿树走即匹配，无需切子串")
    print(f"  - 生产级：SentencePiece / HuggingFace tokenizers 用 Trie（或 DAG + Trie）做分词")


if __name__ == "__main__":
    main()