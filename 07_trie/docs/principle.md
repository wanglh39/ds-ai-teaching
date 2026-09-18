# Trie（前缀树 / 字典树）原理

> 本文档讲清「Trie 是什么、前缀共享怎么省内存、插入 / 查找 / 前缀检查 / 最长前缀匹配、与哈希表对比、C 实现逐行讲解、复杂度」。AI 应用（BPE 分词）见 `ai_application.md`。

## 1. Trie 的定义

**Trie**（读作 "try"），又称**前缀树**或**字典树**，是一种用于高效存储和查找**字符串集合**的树形结构。它的核心思想是：**共享公共前缀**。

```
词表 = { "a", "app", "apple", "application", "ban", "banana" }

Trie 结构:
            (root)
           /      \
          a        b
          |        |
          *an      an
         /  \      |
       app  appl  *ication... 
        |    |
        *   ication
       apple   |
        |      *
        *
```

更精确的画法（`*` 表示该节点是一个完整词的结尾）：

```
                 root
                /    \
               a      b
               |      |
               *      an
              / \      |
            pp  ppl   *
            |    |
            *   ication
           / \    |
         le   *   *
          |
          *
```

关键观察：
- `"app"`、`"apple"`、`"application"` 共享前缀 `"app"`，这条路径在树里**只存一次**
- 每个节点不代表一个完整词，而是代表**从根到该节点路径上的字符串**
- `is_end` 标志标记「从根到这个节点的路径是一个完整词」，区分 `"app"`（词）和 `"appl"`（不是词，只是前缀）

| 数据结构 | 精确查找 | 前缀检查 | 最长前缀匹配 | 空间 |
|---|---|---|---|---|
| 哈希表 | $O(L)$ | $O(L)$（遍历所有词） | $O(V \cdot L)$ | $O(\text{总字符数})$ |
| **Trie** | **$O(L)$** | **$O(L)$** | **$O(L)$** | **$O(\text{不同前缀数})$** |
| 排序数组 + 二分 | $O(L \log V)$ | $O(L \log V)$ | 难 | $O(\text{总字符数})$ |

其中 $L$ 是字符串长度，$V$ 是词表大小。Trie 的杀手锏是**最长前缀匹配 $O(L)$**——沿树走一遍即可，而哈希表要么 $O(V)$ 遍历所有词，要么 $O(L^2)$ 试所有前缀切子串。

## 2. 前缀共享 vs 独立存储：Trie 与哈希表的根本区别

### 2.1 哈希表：每个词独立存储

```
哈希表存 { "app", "apple", "application", "ban", "banana" }:

bucket[...] = {
    "app"          → 3 字符
    "apple"        → 5 字符
    "application"  → 11 字符
    "ban"          → 3 字符
    "banana"       → 6 字符
}
总字符 = 3 + 5 + 11 + 3 + 6 = 28
```

每个词独立存一整串，`"app"` 的 `a-p-p` 和 `"apple"` 的 `a-p-p-l-e` 里的 `a-p-p` **重复存了**。

### 2.2 Trie：公共前缀只存一次

```
Trie 存同样 5 个词:

         root
        /     \
       a       b
       |       |
       p       a
       |       |
       p*      n*
      / \       \
     l   i       a
     |   |       |
     e*  c       n*
         |       |
         a       a
         |       |
         t       *
         |
         i
         |
         o
         |
         n*
```

数节点（不含 root）：`a, p, p, l, e, i, c, a, t, i, o, n, b, a, n, a, n, a` = 18 个节点。

| 存储 | 节点 / 字符数 | 节省 |
|---|---|---|
| 哈希表（独立） | 28 | 基准 |
| Trie（共享） | 18 | 10（35.7%） |

`"app"`、`"apple"`、`"application"` 共享 `a→p→p` 这 3 个节点，不重复存。词表越大、前缀重合越多，节省越显著。

### 2.3 实测：BPE 词表的前缀共享节省

本目录 Python demo 用真实 BPE 算法从语料训练词表，实测：

```
词表大小 326（small）: 总字符 1338, Trie 节点 482, 节省 856 (64.0%)
词表大小 447（mid）  : 总字符 1980, Trie 节点 702, 节省 1278 (64.5%)
```

**Trie 比独立存储省约 65% 的节点**。这是因为自然语言子词的前缀重合度很高（`the`、`there`、`theory`、`theorem` 共享 `the`；`tion`、`ation`、`ition` 共享 `tion`）。

### 2.4 为什么哈希表做不了高效的前缀操作

哈希表的查找语义是**精确匹配**：`"apple" in hash_table` 是 $O(L)$（算哈希 + 比对）。但：

- **前缀检查** `starts_with("app")`：哈希表没有原生的前缀查询。要么遍历所有词看谁以 `"app"` 开头 $O(V \cdot L)$，要么额外存所有前缀（空间爆炸）。
- **最长前缀匹配**：给定 `"applesauce"`，找词表里最长的、是它前缀的词。哈希表要遍历所有词 $O(V \cdot L)$，或从长到短切子串试 $O(L^2)$。

Trie 天然支持这些：沿树走即可，$O(L)$。

## 3. 节点结构

### 3.1 子节点表示

本实现假设字母表是小写字母 `a-z`（26 个），用**定长数组**存子节点：

```c
#define TRIE_ALPHA_SIZE 26

typedef struct TrieNode {
    struct TrieNode *children[TRIE_ALPHA_SIZE];  // 26 个子节点指针
    bool is_end;                                 // 从根到本节点是否构成一个完整词
} TrieNode;
```

- `children[i]` 指向下一个字符是 `'a' + i` 的子节点，`NULL` 表示没有
- `is_end` 标志：区分「前缀」和「完整词」。`"app"` 是词（`is_end=true`），`"appl"` 只是前缀（`is_end=false`）

**为什么用 26 数组而不是哈希表 / 链表存子节点？**
1. **$O(1)$ 寻址**：`children[c - 'a']` 一次数组访问，比哈希表查快
2. **缓存友好**：连续内存，prefetcher 友好
3. **字母表小**：26 个指针 = 208 字节（64 位），可接受
4. **简单**：无哈希冲突、无链表遍历

**代价**：字母表大时（如 Unicode）浪费空间——每个节点 26 个指针，大部分是 `NULL`。生产级对大字母表用**哈希表存子节点**或**压缩 Trie（基数树 / Radix Tree）**。

### 3.2 Trie 结构

```c
typedef struct {
    TrieNode *root;       // 根节点（空串对应的位置）
    size_t word_count;    // 已插入的词数（去重）
    size_t node_count;    // 总节点数（含 root，用于算空间）
} Trie;
```

`word_count` 在重复插入同一词时不增加（`is_end` 已为 true）。`node_count` 用于和「独立存储总字符数」对比，量化前缀共享的节省。

### 3.3 字符映射

```c
static int char_index(char c) {
    if (c < 'a' || c > 'z') return -1;  // 非小写字母非法
    return c - 'a';
}
```

返回 `0..25` 对应 `a..z`，非小写字母返回 `-1`（插入 / 查找时视为非法字符）。本实现只支持小写字母，大写、数字、空格会被拒绝。

## 4. 插入：insert

### 4.1 算法

从根开始，逐字符沿树走，走不通就建新节点，最后标记 `is_end`。

```c
bool trie_insert(Trie *t, const char *word) {
    if (!word || word[0] == '\0') return false;   // 拒绝空串
    TrieNode *cur = t->root;
    for (size_t i = 0; word[i] != '\0'; i++) {
        int idx = char_index(word[i]);
        if (idx < 0) return false;                // 非法字符
        if (!cur->children[idx]) {                // 子节点不存在 → 新建
            cur->children[idx] = node_new();
            if (!cur->children[idx]) return false;
            t->node_count++;
        }
        cur = cur->children[idx];                 // 往下走
    }
    if (!cur->is_end) {                           // 首次插入这个词
        cur->is_end = true;
        t->word_count++;
    }
    return true;
}
```

### 4.2 插入示例

```
插入 "app" 到空 Trie:

1. cur = root, i=0, char='a', idx=0
   children[0] 为 NULL → 新建节点 A, cur = A
2. i=1, char='p', idx=15
   A.children[15] 为 NULL → 新建节点 P1, cur = P1
3. i=2, char='p', idx=15
   P1.children[15] 为 NULL → 新建节点 P2, cur = P2
4. 循环结束, P2.is_end = true, word_count = 1

树: root → a → p → p*

插入 "apple" (复用 "app" 前缀):

1-3. 同上，走到 P2（已存在，不新建）
4. i=3, char='l', P2.children[11] 为 NULL → 新建 L, cur = L
5. i=4, char='e', L.children[4] 为 NULL → 新建 E, cur = E
6. E.is_end = true, word_count = 2

树: root → a → p → p* → l → e*
                    (app)    (apple)

只新建了 2 个节点（l, e），"app" 的 3 个节点复用。
```

### 4.3 重复插入

```
再次 insert "apple":
1-5. 沿树走到 E（节点都已存在，不新建）
6. E.is_end 已经是 true → 不增加 word_count
返回 true
```

重复插入同一词不改变树，`word_count` 不增。这让 `word_count` 始终等于**不同词的数量**。

## 5. 精确查找：search

### 5.1 算法

沿树走，走通了且最后节点 `is_end` 为 true 才算找到。

```c
bool trie_search(const Trie *t, const char *word) {
    if (!word) return false;
    TrieNode *cur = t->root;
    for (size_t i = 0; word[i] != '\0'; i++) {
        int idx = char_index(word[i]);
        if (idx < 0) return false;
        cur = cur->children[idx];
        if (!cur) return false;                   // 走不通 → 没这个词
    }
    return cur->is_end;                           // 走通了，看是不是完整词
}
```

### 5.2 search vs starts_with 的区别

```
词表 = { "app", "apple" }

search("app")    → 走 a→p→p, is_end=true  → true   ✓ 是词
search("appl")   → 走 a→p→p→l, is_end=false → false  ✗ 不是词（只是前缀）
search("apple")  → 走 a→p→p→l→e, is_end=true → true  ✓ 是词
search("appx")   → 走 a→p→p, x 子节点 NULL  → false  ✗ 走不通

starts_with("app")   → 走通 → true   ✓ 有词以 app 开头
starts_with("appl")  → 走通 → true   ✓ 有词以 appl 开头（apple）
starts_with("appx")  → 走不通 → false ✗ 没词以 appx 开头
```

- `search` 要求**精确是词**：走通 **且** `is_end`
- `starts_with` 只要求**前缀存在**：走通即可（不管 `is_end`）

这个区别是 Trie 比哈希表强的关键：哈希表只能做 `search`，做不了高效的 `starts_with`。

## 6. 前缀检查：starts_with

```c
bool trie_starts_with(const Trie *t, const char *prefix) {
    if (!prefix) return false;
    TrieNode *cur = t->root;
    for (size_t i = 0; prefix[i] != '\0'; i++) {
        int idx = char_index(prefix[i]);
        if (idx < 0) return false;
        cur = cur->children[idx];
        if (!cur) return false;
    }
    return true;                                  // 走通即可，不看 is_end
}
```

和 `search` 唯一区别：最后 `return true` 而非 `return cur->is_end`。只要前缀路径存在就返回 true。

**空串前缀**：`starts_with("")` 循环不执行，直接返回 `true`——空串是任何串的前缀，这是数学上的正确语义。

## 7. 最长前缀匹配：longest_prefix

这是 Trie 在 AI（BPE 分词）里的核心操作：给定一个字符串 $s$，找词表里**最长的、是 $s$ 前缀的词**。

### 7.1 算法

从位置 0 开始沿 Trie 走，**边走边记录最后一个 `is_end` 的位置**，走到不能走为止。

```c
size_t trie_longest_prefix(const Trie *t, const char *s, char *out, size_t out_cap) {
    if (!s || !t->root) return 0;
    TrieNode *cur = t->root;
    size_t last_end = 0;                          // 最后一个 is_end 的长度
    size_t i = 0;
    for (; s[i] != '\0'; i++) {
        int idx = char_index(s[i]);
        if (idx < 0) break;
        cur = cur->children[idx];
        if (!cur) break;                          // 走不通，停
        if (cur->is_end) {
            last_end = i + 1;                     // 记录这个位置是完整词
        }
    }
    if (out && last_end > 0 && last_end < out_cap) {
        memcpy(out, s, last_end);                 // 拷贝最长前缀到 out
        out[last_end] = '\0';
    }
    return last_end;                              // 返回最长前缀长度
}
```

### 7.2 示例

```
词表 = { "a", "app", "apple", "the", "there" }

longest_prefix("applesauce"):
  i=0 'a' → 走到 A, is_end=true  → last_end=1
  i=1 'p' → 走到 P1, is_end=false
  i=2 'p' → 走到 P2, is_end=true  → last_end=3   ("app")
  i=3 'l' → 走到 L, is_end=false
  i=4 'e' → 走到 E, is_end=true   → last_end=5   ("apple")
  i=5 's' → E.children['s'] 为 NULL → break
  返回 last_end=5, out="apple"

longest_prefix("application"):
  ...走到 P2 时 last_end=3 ("app")
  继续走 l→i→... 但 "appli..." 路径上没有 is_end=true 的节点（"application" 不在词表）
  返回 last_end=3, out="app"

longest_prefix("xyz"):
  i=0 'x' → root.children['x'] 为 NULL → break
  返回 last_end=0（无匹配）
```

### 7.3 为什么是 O(L) 而不是 O(L²) 或 O(V)

- **沿 Trie 走一遍**：每个字符一次数组访问，最多走 $\min(L_s, L_{\max})$ 步（$L_s$ 输入串长，$L_{\max}$ 词表最长词）
- **不切子串**：不像哈希表法要 `s[0:1], s[0:2], ...` 逐个试，Trie 边走边匹配
- **不遍历词表**：不像朴素法要试词表里每个词，Trie 的路径自动筛选出所有前缀

对比：
| 方法 | 最长前缀匹配复杂度 | 说明 |
|---|---|---|
| 朴素哈希（遍历所有词） | $O(V \cdot L)$ | 每个词试一次 `startswith` |
| 切子串哈希（按长度试） | $O(L^2)$ | 切 $L$ 个子串，每个哈希查 $O(L)$ |
| **Trie** | **$O(L)$** | **沿树走一遍** |

$V = 50000$（大词表）、$L = 15$（最长子词）时，Trie 比朴素哈希快 $V/L \approx 3333$ 倍。这正是 BPE 分词用 Trie 的根本原因。

## 8. 空间分析：前缀共享省了多少

### 8.1 节点数 = 不同前缀数

Trie 的节点数等于**所有词的所有不同前缀的数量**（含空串对应的 root）。

```
词表 = { "a", "ab", "abc" }
所有前缀: "", "a", "ab", "abc" → 4 个节点

词表 = { "abc", "abd" }
所有前缀: "", "a", "ab", "abc", "abd" → 5 个节点
"ab" 共享，不重复
```

### 8.2 与独立存储对比

| 存储 | 空间 |
|---|---|
| 哈希表（每词独立） | $\sum_{w \in \text{vocab}} |w|$（总字符数） |
| Trie（前缀共享） | $|\{\text{所有不同前缀}\}|$（节点数） |

**最坏情况**（无前缀共享，如 `"a", "b", "c", ...`）：Trie 节点数 = 总字符数，和哈希表一样。
**最好情况**（全是同一前缀，如 `"a", "aa", "aaa", ...`）：Trie 节点数 = 最长词长度，远小于总字符数。

自然语言子词的前缀重合度高，接近最好情况。本目录实测 BPE 词表节省约 65%。

### 8.3 本实现的实测

C 测试里插入 100000 个 8 字符随机词，节点数 518279：

```
trie size = 100000, nodes = 518279
```

随机词前缀重合度低，节点数约为 $100000 \times 8 \times \text{某系数}$，共享少。而真实 BPE 词表（有大量 `the, there, theory` 类前缀重合）共享显著，实测节省 65%。

## 9. C 实现逐行讲解

### 9.1 节点创建

```c
static TrieNode *node_new(void) {
    TrieNode *n = (TrieNode *)calloc(1, sizeof(TrieNode));
    return n;
}
```

`calloc` 而非 `malloc`：`calloc` 把内存清零，`children[26]` 全是 `NULL`、`is_end` 是 `false`，省去手动初始化。

### 9.2 递归释放

```c
static void node_free(TrieNode *n) {
    if (!n) return;
    for (int i = 0; i < TRIE_ALPHA_SIZE; i++) {
        node_free(n->children[i]);    // 递归释放每个子节点
        n->children[i] = NULL;
    }
    free(n);                          // 释放自己
}
```

后序遍历：先释放所有子树，再释放自己。Trie 是树（无环），递归安全。深度 = 最长词长度，不会栈溢出（除非词长几百万）。

### 9.3 init / free

```c
void trie_init(Trie *t) {
    t->root = node_new();             // 创建 root（代表空串）
    t->word_count = 0;
    t->node_count = 1;                // root 算 1 个节点
}

void trie_free(Trie *t) {
    node_free(t->root);
    t->root = NULL;
    t->word_count = 0;
    t->node_count = 0;
}
```

`init` 预创建 root，代表空串位置。`node_count` 初始为 1（root）。

### 9.4 insert 逐行

```c
bool trie_insert(Trie *t, const char *word) {
    if (!word || word[0] == '\0') return false;   // (1) 拒绝 NULL 和空串
    TrieNode *cur = t->root;
    for (size_t i = 0; word[i] != '\0'; i++) {
        int idx = char_index(word[i]);            // (2) 字符映射 0..25
        if (idx < 0) return false;                // (3) 非法字符，拒绝
        if (!cur->children[idx]) {                // (4) 子节点不存在
            cur->children[idx] = node_new();      //     新建
            if (!cur->children[idx]) return false;//     OOM
            t->node_count++;                      // (5) 节点计数
        }
        cur = cur->children[idx];                 // (6) 往下走
    }
    if (!cur->is_end) {                           // (7) 首次插入这个词
        cur->is_end = true;
        t->word_count++;
    }
    return true;
}
```

关键点：
- **(1)** 拒绝空串：空串插入会让 root 的 `is_end=true`，语义混乱（空串是所有串的前缀）。明确拒绝更清晰。
- **(4)** 懒建节点：只在走不通时才新建，已有前缀直接复用——这是前缀共享的实现。
- **(7)** 去重：重复插入不增 `word_count`，让计数始终是不同词数。

### 9.5 search 逐行

```c
bool trie_search(const Trie *t, const char *word) {
    if (!word) return false;
    TrieNode *cur = t->root;
    for (size_t i = 0; word[i] != '\0'; i++) {
        int idx = char_index(word[i]);
        if (idx < 0) return false;
        cur = cur->children[idx];
        if (!cur) return false;                   // 走不通，词不存在
    }
    return cur->is_end;                           // 走通了，看是不是完整词
}
```

和 insert 的区别：不新建节点，走不通直接返回 false。最后看 `is_end` 区分词与前缀。

### 9.6 longest_prefix 逐行

```c
size_t trie_longest_prefix(const Trie *t, const char *s, char *out, size_t out_cap) {
    if (!s || !t->root) return 0;
    TrieNode *cur = t->root;
    size_t last_end = 0;                          // 记录最后 is_end 的长度
    size_t i = 0;
    for (; s[i] != '\0'; i++) {
        int idx = char_index(s[i]);
        if (idx < 0) break;                       // 非法字符，停
        cur = cur->children[idx];
        if (!cur) break;                          // 走不通，停
        if (cur->is_end) {
            last_end = i + 1;                     // 更新最长匹配位置
        }
    }
    if (out && last_end > 0 && last_end < out_cap) {
        memcpy(out, s, last_end);                 // 拷贝结果
        out[last_end] = '\0';
    }
    return last_end;
}
```

关键：**`last_end` 在循环里不断更新**，每次遇到 `is_end` 就记下当前位置。循环结束后 `last_end` 就是最后一个 `is_end` 的位置 = 最长前缀长度。`break` 而非 `return`：走不通时仍要用之前记录的 `last_end`（可能有更短的前缀匹配）。

## 10. 复杂度分析

### 10.1 各操作复杂度

设 $L$ 为字符串长度，$\Sigma$ 为字母表大小（本实现 26）。

| 操作 | 时间 | 空间 | 说明 |
|---|---|---|---|
| init | $O(1)$ | $O(1)$ | 创建 root |
| insert | $O(L)$ | $O(L)$ 最坏 | 每字符 $O(1)$，可能建 $L$ 个新节点 |
| search | $O(L)$ | $O(1)$ | 沿树走 |
| starts_with | $O(L)$ | $O(1)$ | 沿树走 |
| longest_prefix | $O(L)$ | $O(1)$ | 沿树走 + 记录 |
| free | $O(N)$ | $O(N)$ 栈 | $N$ = 节点数，递归释放 |

**所有核心操作都是 $O(L)$**，与词表大小 $V$ 无关。这是 Trie 的核心优势：查找代价只取决于查询串长度，不随词表增长。

### 10.2 空间复杂度

$$\text{节点数} = |\{ \text{所有词的所有不同前缀} \}| \le \sum_{w \in \text{vocab}} |w|$$

最坏（无共享）等于总字符数，最好（全共享）等于最长词长度。每节点 $|\Sigma|$ 个指针 + 1 个 bool = $26 \times 8 + 1 = 209$ 字节（64 位，有对齐）。

### 10.3 与哈希表对比

| 操作 | Trie | 哈希表 |
|---|---|---|
| insert | $O(L)$ | $O(L)$（算哈希 + 存串） |
| search（精确） | $O(L)$ | $O(L)$（算哈希 + 比对） |
| starts_with（前缀） | $O(L)$ | $O(V \cdot L)$ 或不支持 |
| longest_prefix | $O(L)$ | $O(V \cdot L)$ 或 $O(L^2)$ |
| 空间 | $O(\text{不同前缀})$ | $O(\text{总字符})$ |

**精确查找两者同阶**，但**前缀类操作 Trie 完胜**。BPE 分词要的就是最长前缀匹配，所以用 Trie。

## 11. 与其他结构对比

| 维度 | Trie | 哈希表 | 排序数组 | BST / 平衡树 |
|---|---|---|---|---|
| 精确查找 | $O(L)$ | $O(L)$ | $O(L \log V)$ | $O(L \log V)$ |
| 前缀查找 | $O(L)$ | $O(V \cdot L)$ | $O(L \log V)$ | $O(V \cdot L)$ |
| 最长前缀匹配 | $O(L)$ | $O(V \cdot L)$ | 难 | 难 |
| 范围查询（字典序） | $O(L + K)$ | 不支持 | $O(\log V + K)$ | $O(\log V + K)$ |
| 空间 | $O(\text{前缀数})$ | $O(\text{总字符})$ | $O(\text{总字符})$ | $O(\text{总字符})$ |
| 前缀共享 | 是 | 否 | 否 | 否 |

**选 Trie**：需要前缀操作（前缀检查、最长前缀匹配、自动补全、字典序遍历），且前缀重合度高。
**选哈希表**：只需精确查找，不要前缀操作。
**选排序数组**：一次排序后多次二分，不动态插入。

## 12. 变体与改进方向

1. **压缩 Trie（基数树 / Radix Tree）**：把单字符路径压缩成字符串边，省中间节点。如 `"application"` 不必逐字符建 11 个节点，一条边存 `"application"`。路由表（Linux radix tree）、IP 路由查找用这个。
2. **Ternary Search Tree（TST）**：每个节点只存一个字符 + 三个指针（左、中、右），空间更省（无空指针），适合字母表大时。
3. **DAWG（有向无环词图）**：合并相同子树，比 Trie 更省空间，但只读（不支持动态插入）。拼写检查用。
4. **哈希子节点**：字母表大时（如 Unicode CJK），`children` 用哈希表而非定长数组，空间 $O(\text{实际子节点数})$ 而非 $O(|\Sigma|)$。本目录 Python demo 的 `TrieNode.children: dict` 就是这种。
5. **双数组 Trie（Double-Array Trie）**：用两个一维数组编码 Trie，极致缓存友好，查速接近数组访问。Aho-Corasick、生产级分词器用。
6. **AC 自动机**：Trie + 失败指针，多模式串一次匹配 $O(L)$。敏感词过滤、DNA 序列匹配用。

## 13. 本实现的局限

1. **只支持小写字母 a-z**：`char_index` 拒绝非小写。生产级需支持 Unicode（用哈希子节点或 UTF-8 字节 Trie）。
2. **定长 26 数组**：字母表大时浪费。中文 / Unicode 场景每节点 26 个指针大部分空。
3. **递归释放**：超长词可能栈溢出（实际词长有限，问题不大）。可改迭代 + 显式栈。
4. **无删除**：本实现没有 `delete`。删除需回收节点（子节点全空才 free）并清 `is_end`。
5. **无持久化**：不能保存 / 加载。生产级需序列化（如双数组 Trie 可直接 mmap）。
6. **指针开销**：每节点 26 个指针 = 208 字节，存小词表时哈希表更省。

这些是教学实现的合理取舍。生产级 Trie（如 HuggingFace `tokenizers` 的内部结构）用双数组或压缩 Trie，兼顾速度和空间。

## 14. 小结

Trie 的核心三件套：
1. **前缀共享**：公共前缀只存一次，省内存（实测 BPE 词表省 65%）
2. **沿树走即匹配**：insert / search / starts_with / longest_prefix 都是 $O(L)$，与词表大小无关
3. **is_end 标志**：区分「前缀」和「完整词」，让一棵树同时支持精确查找和前缀查找

理解了这三点，就理解了 Trie。它比哈希表强在**前缀类操作**——最长前缀匹配 $O(L)$ vs 哈希表 $O(V \cdot L)$，这正是 BPE 分词、自动补全、路由查找选 Trie 的根本原因。

下一章 `ai_application.md` 讲 Trie 怎么在 AI 里做 BPE 分词（最长前缀匹配）。