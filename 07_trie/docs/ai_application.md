# Trie 的 AI 应用：BPE 分词（最长前缀匹配）

> 本文档讲清「Trie 在 AI 里干什么」。原理见 `principle.md`。核心应用：BPE（Byte Pair Encoding）分词——所有现代 LLM（GPT、LLaMA、BERT）的 tokenization 第一步，用 Trie 做最长前缀匹配。

## 1. 什么是 tokenization：AI 处理文本的第一步

### 1.1 为什么需要 tokenization

神经网络不能直接处理字符串，只能处理数字。**tokenization** 就是把文本切成一个个离散单元（**token**），每个 token 映射到一个整数 ID：

```
文本: "tokenization is the first step"
      ↓ tokenization
tokens: ["tokenization", "is", "the", "first", "step"]
      ↓ 词表查 ID
IDs:   [12976, 318, 262, 1142, 786]
      ↓ 送入模型
```

tokenization 是所有 LLM 的**第一步**，发生在训练和推理的最前端。它的质量直接决定：
- **词表大小** $V$：影响 embedding 表大小和输出层维度
- **序列长度**：切得越细序列越长，attention 是 $O(n^2)$，越长越慢
- **OOV（未登录词）处理**：词表没有的词怎么办

### 1.2 三种 tokenization 粒度

| 粒度 | 示例 | 词表大小 | OOV | 问题 |
|---|---|---|---|---|
| 字符级 | `t o k e n` | ~100 | 无 | 序列太长，语义稀疏 |
| 词级 | `tokenization` | ~100万 | 严重 | 词表爆炸，新词全是 OOV |
| **子词级** | `token ization` | ~3万-13万 | 几乎无 | **折中，现代 LLM 标准** |

**子词分词**（subword tokenization）在字符和词之间折中：常见词整体保留，罕见词切成更小的子词单元。`"tokenization"` 可能切成 `["token", "ization"]`，两个都是词表里的子词。

```
词级:   "unhappiness" → ["unhappiness"] (可能 OOV)
字符级: "unhappiness" → ["u","n","h","a","p","p","i","n","e","s","s"] (太长)
子词级: "unhappiness" → ["un", "happiness"] (都在词表，无 OOV，长度 2)
```

### 1.3 BPE：最流行的子词分词算法

**BPE**（Byte Pair Encoding，字节对编码）是 GPT、GPT-2、GPT-3、GPT-4、LLaMA、BERT 系列用的分词算法。核心思想：**从字符开始，反复合并最高频的相邻字符对，直到词表达到目标大小**。

```
BPE 训练（学词表）:
1. 初始词表 = 语料里所有单字符 {a, b, c, ..., t, h, e, ...}
2. 统计语料里所有相邻字符对频率: (t,h):500, (h,e):480, (e,_):400, ...
3. 合并最高频 pair: 把 "th" 当成一个新子词加入词表
4. 在语料里把所有 "t h" 替换成 "th"
5. 重复 2-4，直到词表达到目标大小（如 50000）

BPE 推理（分词新文本）:
对文本每个位置，用词表里最长的匹配子词切分（最长前缀匹配）
```

**训练阶段**学出词表（一组子词），**推理阶段**用这个词表切新文本。推理的核心操作就是**最长前缀匹配**——这正是 Trie 的拿手好戏。

## 2. BPE 推理：最长前缀匹配

### 2.1 分词算法

给定词表 $V$（一组子词）和输入文本 $s$，BPE 分词（最长匹配贪心版）：

```
i = 0
while i < len(s):
    找词表里最长的、是 s[i:] 前缀的子词 w
    输出 w 作为 token
    i += len(w)
```

```
词表 = { "t", "h", "e", "th", "the", "re", "n", "d", "and", "ing", ... }
输入 = "thereand"

i=0: s[0:]="thereand", 最长前缀匹配 → "the" (3字符)
     tokens=["the"], i=3
i=3: s[3:]="reand", 最长前缀匹配 → "re" (2字符)
     tokens=["the","re"], i=5
i=5: s[5:]="and", 最长前缀匹配 → "and" (3字符)
     tokens=["the","re","and"], i=8
结束: ["the","re","and"]
```

每个位置的核心操作：**从词表里找最长的、匹配当前位置开头的子词**。

### 2.2 用 Trie 实现最长前缀匹配

把词表里所有子词插入 Trie，每个子词结尾标 `is_end`。分词时从位置 $i$ 开始沿 Trie 走，记录最后一个 `is_end` 的位置，走到不能走为止：

```python
def trie_bpe_tokenize(text, trie):
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        cur = trie.root
        last_end = i          # 记录最长匹配位置
        j = i
        while j < n and text[j] in cur.children:
            cur = cur.children[text[j]]
            j += 1
            if cur.is_end:
                last_end = j  # 这个位置是完整子词，更新
        tokens.append(text[i:last_end])
        i = last_end
    return tokens
```

**沿 Trie 走一遍 = 检查所有前缀**：Trie 的路径天然编码了所有以当前位置开头的前缀，走一遍就知道哪些在词表里，取最长的。复杂度 $O(L)$，$L$ = 最长子词长度。

### 2.3 为什么 Trie 是 BPE 的天然选择

| BPE 推理需求 | Trie 怎么满足 |
|---|---|
| 每个位置找最长匹配子词 | `longest_prefix` $O(L)$，沿树走一遍 |
| 词表大（3万-13万） | 查找 $O(L)$ 与词表大小无关 |
| 子词前缀重合多（`the`/`there`/`theory`） | 前缀共享省内存 |
| 每次推理调几百万次 | 单次 $O(L)$ 极快，无切子串开销 |

**核心：BPE 分词 = 反复最长前缀匹配，Trie 的 longest_prefix 是 $O(L)$，而哈希表是 $O(V)$ 或 $O(L^2)$。**

## 3. 为什么不用哈希表：朴素法的问题

### 3.1 朴素法一：遍历所有子词 $O(n \cdot V)$

```python
def naive_bpe_tokenize(text, vocab_set):
    tokens = []
    i = 0
    while i < len(text):
        best_len = 0
        best_word = ""
        for word in vocab_set:                    # 遍历词表所有子词
            if len(word) > best_len and text.startswith(word, i):
                best_len = len(word)
                best_word = word
        tokens.append(best_word)
        i += best_len
    return tokens
```

每个位置遍历词表所有 $V$ 个子词，每个做 `startswith` $O(L)$。总复杂度 $O(n \cdot V \cdot L)$，$n$ = 文本长度。

**问题**：$V = 50000$（LLaMA 词表）、$n = 10000$ 字符时，$5 \times 10^8$ 次 `startswith`，Python 里要几十秒到几分钟。完全不可行。

### 3.2 朴素法二：切子串按长度试 $O(n \cdot L^2)$

```python
def hash_bpe_tokenize(text, buckets, max_len):
    tokens = []
    i = 0
    while i < len(text):
        best_len = 0
        for l in range(max_len, 0, -1):           # 从最长往短试
            if text[i:i+l] in buckets[l]:          # 切子串 + 哈希查
                best_len = l
                break
        tokens.append(text[i:i+best_len])
        i += best_len
    return tokens
```

把词表按子词长度分桶，每个位置从最长 $L$ 往短试，切 `text[i:i+l]` 查对应桶。总复杂度 $O(n \cdot L^2)$（切 $L$ 次子串，每次 $O(L)$）或 $O(n \cdot L)$（如果切片 $O(1)$ 视图 + 哈希 $O(L)$）。

**问题**：
1. **每个位置切 $L$ 次子串**：`text[i:i+l]` 每次分配新字符串对象，开销大
2. **$L$ 次哈希查**：每次哈希要重算子串哈希 $O(l)$
3. **无前缀共享**：`"the"` 和 `"there"` 在哈希表里独立存，`"the"` 的 `t-h-e` 重复
4. **重复扫描**：试 `text[i:i+5]` 和 `text[i:i+4]` 时，前 4 个字符被扫描了两遍

### 3.3 Trie 怎么避免这些问题

```python
def trie_bpe_tokenize(text, trie):
    # ...
    while i < n:
        cur = trie.root
        last_end = i
        j = i
        while j < n and text[j] in cur.children:   # 沿树走，不切子串
            cur = cur.children[text[j]]
            j += 1
            if cur.is_end:
                last_end = j
        # ...
```

1. **不切子串**：逐字符沿 `children` 字典走，只访问单个字符，不分配新字符串
2. **一次遍历检查所有前缀**：沿 Trie 走的路径自动覆盖所有以 $i$ 开头的前缀，走一遍就找到最长匹配
3. **前缀共享**：`"the"` 和 `"there"` 共享 `t→h→e` 路径，内存只存一次
4. **无重复扫描**：每个字符只访问一次，沿树走 $O(L)$

**同是 $O(n \cdot L)$，Trie 比切子串哈希快 5 倍**（本目录实测），因为不切子串 + 前缀共享 + 单次遍历的常数优势。

### 3.4 三种方法复杂度对比

| 方法 | 复杂度 | $n=50000, V=447, L=15$ | 实测 |
|---|---|---|---|
| 朴素遍历所有子词 | $O(n \cdot V \cdot L)$ | $\sim 3.4 \times 10^8$ | ~280 ms（小规模外推） |
| 切子串哈希（分桶） | $O(n \cdot L^2)$ 或 $O(n \cdot L)$ | $\sim 7.5 \times 10^5$ | 48 ms |
| **Trie 最长匹配** | **$O(n \cdot L)$** | $\sim 7.5 \times 10^5$ | **9 ms** |

Trie 比朴素快 60x，比分桶哈希快 5x。

## 4. demo 结果解读

运行 `python 07_trie/python/demo.py` 的关键输出。

### 4.1 BPE 词表训练

```
词表大小: small=326 (max_len=12), mid=447 (max_len=15), large=447 (max_len=15)
```

用真实 BPE 算法（统计高频 pair 合并）从语料训练：
- `small`：300 次合并，词表 326 个子词，最长 12 字符
- `mid` / `large`：800 / 2000 次合并，语料有限后合并饱和，词表 447 个子词，最长 15 字符

子词示例（分词结果）：
```
text = "tokenizationisthefirststepandthetriehelpswithlongestprefixmatching"
trie = ['tokenization', 'is', 'the', 'first', 'step', 'and', 'the', 'trie',
        'h', 'el', 'p', 's', 'with', 'longest', 'prefix', 'matching']
```

`"tokenization"` 作为整体子词在词表里（BPE 学到的），`"helps"` 不在词表切成 `["h","el","p","s"]`。三种方法结果完全一致，验证正确性。

### 4.2 内存对比：前缀共享

```
small: 词表=326, 总字符=1338, Trie 节点=482,   节省 856 (64.0%)
mid:   词表=447, 总字符=1980, Trie 节点=702,   节省 1278 (64.5%)
```

**Trie 比独立存储省 65% 内存**。原因：BPE 子词前缀重合度高——`the`/`there`/`therefore` 共享 `the`，`tion`/`ation`/`ition` 共享 `tion`，`ing`/`thing`/`thing` 共享 `ing`。哈希表把这些重复前缀各存一份，Trie 只存一份。

### 4.3 性能对比（含朴素 O(nV)）

```
|vocab|=326, text_len=2000, max_len=12
实现                     均值(ms)      标准差(ms)        加速比
trie O(nL)             0.4480       0.0128      63.47x
hash-bucket O(nL)       2.3651       0.2002      12.02x
naive O(nV)           28.4326       1.7209       1.00x
→ Trie 比朴素快 63.5x（O(nL) vs O(nV)，L=12 vs V=326）

|vocab|=326, text_len=5000, max_len=12
trie O(nL)             1.2212       0.2593      56.23x
hash-bucket O(nL)       5.2277       0.4808      13.14x
naive O(nV)           68.6746       2.8182       1.00x
→ Trie 比朴素快 56.2x
```

读法（2000 字符，326 子词词表）：
- **Trie**：0.45 ms，$O(n \cdot L)$
- **分桶哈希**：2.4 ms，$O(n \cdot L)$ 但切子串 + 多次哈希
- **朴素遍历**：28.4 ms，$O(n \cdot V)$

**Trie 比朴素快 63 倍**。理论比值 $V/L = 326/12 \approx 27$，实测 63 倍更高——因为朴素法每次 `startswith` 还要逐字符比对，而 Trie 是直接字典查 `children[ch]`，常数更小。

**文本越长优势稳定**：5000 字符时 56x，和 2000 字符的 63x 接近，说明是线性增长（$O(n)$ 主导），比值由 $V/L$ 决定。

### 4.4 Trie vs 分桶哈希（同阶 O(nL)）

```
|vocab|=447 (mid), text_len=50000, max_len=15
trie O(nL)             8.8802 ms    5.56x
hash-bucket O(nL)     49.3911 ms    1.00x
→ Trie 比 hash-bucket 快 5.56x（同阶 O(nL)，前缀共享 + 单次遍历优势）
```

两者**同是 $O(n \cdot L)$**，但 Trie 快 5.5 倍。原因：
1. **不切子串**：Trie 逐字符走 `children` 字典，哈希法每个位置切 $L$ 次 `text[i:i+l]`，每次分配新字符串
2. **一次遍历**：Trie 走一遍检查所有前缀，哈希法从长到短试，前缀字符被重复扫描
3. **字典查 vs 哈希查**：`children[ch]` 是一次 dict 查（$O(1)$），`text[i:i+l] in bucket` 要先切片再算哈希再查

**结论**：即使把哈希表优化到同阶 $O(n \cdot L)$（分桶），Trie 仍因不切子串 + 单次遍历快 5 倍。这是为什么生产级分词器选 Trie 而非哈希表。

### 4.5 关键数字总结

```
词表=447, text_len=50000:
  Trie 最长匹配     : 9.294 ms  (O(n*L), L=15)
  hash-bucket 匹配  : 47.988 ms (O(n*L), 切子串+哈希查)
  hash/Trie = 5.16x
  前缀共享节省 64.5% 内存（节点 702 < 总字符 1980）
  朴素 O(n*V) 在此规模约需 0.3 s（外推，太慢不实测）
```

## 5. 生产级实现：从教学 Trie 到工业分词器

### 5.1 HuggingFace tokenizers

HuggingFace 的 `tokenizers` 库（Rust 实现，Python 绑定）是 LLM tokenization 的事实标准：

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE(vocab, merges))
encoding = tokenizer.encode("tokenization is the first step")
print(encoding.tokens)  # ['tokenization', 'is', 'the', 'first', 'step']
print(encoding.ids)     # [12976, 318, 262, 1142, 786]
```

内部用**双数组 Trie（Double-Array Trie）**存词表：
- 两个一维数组 `base[]` 和 `check[]` 编码整棵 Trie
- 查找是一次数组访问：`next = base[cur] + c; if check[next] == cur: cur = next`
- 极致缓存友好（连续内存），比指针 Trie 快几倍
- 构建后只读，适合词表固定的推理场景

### 5.2 SentencePiece

Google 的 **SentencePiece**（LLaMA、T5 用）支持 BPE 和 Unigram 两种模型：

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
ids = sp.encode("tokenization is the first step")
tokens = sp.id_to_piece(ids)  # ['▁token', 'ization', '▁is', ...]
```

特点：
- **语言无关**：把文本当 Unicode 字节流，不依赖空格分词（`▁` 代表空格）
- **Unigram 模型**：用概率 + 前向 DP 找最优切分（比 BPE 贪心更优），DP 里用 Trie 加速子词查
- **BPE 模型**：最长前缀匹配，用 Trie

### 5.3 GPT-4 的 tiktoken

OpenAI 的 **tiktoken**（GPT-3.5 / GPT-4 用）是 BPE 的优化实现：

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
ids = enc.encode("tokenization is the first step")
# [12976, 318, 262, 1142, 786]
```

特点：
- **正则预切分**：先用正则把文本切成块（数字、字母、空格、标点），每块独立 BPE
- **BPE 合并查表**：用 **hash + rank** 而非 Trie——按合并优先级（rank）贪心合并相邻对
- **Rust / C 实现**：极致优化，每秒处理几百万字符

tiktoken 用哈希 + rank 而非 Trie，是因为它做的是**按合并规则贪心合并**（训练时的 pair 顺序），而非最长前缀匹配。两种 BPE 推理策略：
- **最长前缀匹配**（本目录 demo）：Trie，$O(n \cdot L)$
- **按 rank 贪心合并**（tiktoken）：hash + 优先级，$O(n \cdot L)$ 但策略不同

### 5.4 两种 BPE 推理策略对比

| 策略 | 做法 | 数据结构 | 用在 |
|---|---|---|---|
| **最长前缀匹配** | 每位置找词表最长匹配子词 | **Trie** | SentencePiece BPE、教学 |
| **按 rank 贪心合并** | 反复合并 rank 最小的相邻对 | hash + rank 表 | tiktoken (GPT) |

两者结果**不完全一致**（最长匹配是贪心近似，rank 合并是训练复现），但都 $O(n \cdot L)$。最长前缀匹配更直观、Trie 更自然；rank 合并更贴合训练、实现更紧凑。

### 5.5 为什么生产级仍离不开 Trie 的思想

即使 tiktoken 用 hash + rank，**Trie 的前缀共享思想无处不在**：
1. **双数组 Trie**：HuggingFace tokenizers、Aho-Corasick、中文分词（jieba 词典）都用
2. **Unigram DP**：SentencePiece Unigram 模型的前向 DP 要枚举所有子词，用 Trie 加速子词枚举
3. **AC 自动机**：多模式串匹配（敏感词过滤）= Trie + 失败指针
4. **路由表**：HTTP 路由、IP 路由查找用压缩 Trie（基数树）

**核心思想**：字符串集合的前缀操作（查找、匹配、枚举），Trie 比哈希表更高效，因为前缀共享 + 沿树走。

## 6. Trie 在 AI 里的其他应用

| 应用 | Trie 的作用 | 为什么用 Trie |
|---|---|---|
| **BPE 分词** | 最长前缀匹配子词 | $O(n \cdot L)$，前缀共享省内存 |
| **Unigram 分词** | 枚举所有子词候选（DP 用） | $O(L)$ 枚举当前位置所有子词 |
| **自动补全** | 前缀查所有后续词 | $O(L + K)$，K = 补全候选数 |
| **拼写检查 / 修正** | 找编辑距离 $\le 1$ 的词 | 沿 Trie 走 + 容错分支 |
| **敏感词过滤** | 多模式串一次匹配（AC 自动机） | Trie + 失败指针 $O(n)$ |
| **搜索引擎前缀提示** | 输入框联想 | 前缀子树遍历，按频率排序 |
| **IP 路由查找** | 最长前缀匹配 IP 地址 | 压缩 Trie（基数树），网络路由表 |
| **DNA 序列匹配** | 基因序列子串查找 | 后缀树 / 后缀自动机（Trie 变体） |

### 6.1 自动补全（前缀子树遍历）

```
用户输入 "app"，Trie 走到 "app" 节点，遍历子树收集所有 is_end 节点:
→ ["app", "apple", "application", "apply", "appreciate", ...]
按频率排序返回 top-10
```

哈希表做不了：要遍历所有词看谁以 `"app"` 开头，$O(V \cdot L)$。Trie 走到 `"app"` 节点后遍历子树 $O(K)$，$K$ = 匹配数。

### 6.2 AC 自动机（多模式串匹配）

敏感词过滤：一次扫描文本，找出所有命中的敏感词（词表可能几万个）。

```
朴素: 对每个敏感词做 KMP，O(文本长 × 敏感词数)
AC 自动机: 把敏感词建成 Trie + 失败指针，文本扫一遍 O(文本长 + 命中数)
```

AC 自动机 = Trie + KMP 的失败思想，把多模式串匹配从 $O(n \cdot V)$ 降到 $O(n)$。这是 Trie 在 AI 内容审核里的核心应用。

## 7. 从教学 demo 到生产级：差距与改进

| 维度 | 本目录 demo | 生产级 |
|---|---|---|
| Trie 实现 | Python dict 子节点 | Rust 双数组 Trie |
| 字母表 | 任意字符（dict） | UTF-8 字节 / BPE rank |
| 分词策略 | 最长前缀匹配（贪心） | rank 合并 / Unigram DP |
| 预处理 | 无 | 正则预切分（数字/标点/空格） |
| 词表大小 | 447 | 32000-128256 |
| 吞吐 | ~5 MB/s | ~100 MB/s（tiktoken） |
| 内存 | Python 对象开销 | mmap 双数组，零拷贝 |

**教学 demo 聚焦算法本质**：Trie 最长匹配 vs 哈希查找的复杂度差异。生产级在此基础上做：Rust/C 实现、双数组编码、正则预切分、批量并行、mmap 持久化。

## 8. 小结

BPE 分词为什么用 Trie 而不是哈希表：
1. **最长前缀匹配 $O(L)$**：Trie 沿树走一遍，哈希表要遍历所有子词 $O(V)$ 或切子串 $O(L^2)$
2. **前缀共享省内存**：`the`/`there`/`theory` 共享 `the` 路径，实测 BPE 词表省 65%
3. **不切子串**：逐字符走 `children`，不分配新字符串，比分桶哈希快 5 倍

本目录实测（词表 447，文本 50000 字符）：
- Trie 最长匹配：9 ms
- 分桶哈希（同阶 $O(nL)$）：48 ms（慢 5x）
- 朴素遍历（$O(nV)$）：外推 ~300 ms（慢 60x）

**核心洞察**：BPE 分词 = 反复最长前缀匹配，Trie 的 `longest_prefix` 是 $O(L)$ 且与词表大小无关，这是 Trie 比哈希表的根本优势。所有现代 LLM 分词器（HuggingFace tokenizers、SentencePiece）的内核都基于 Trie 或其变体（双数组 Trie、AC 自动机），因为子词分词的前缀匹配语义天然适配 Trie。