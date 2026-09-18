# 哈希表原理

> 本文档讲清「哈希表是什么、哈希函数怎么选、冲突怎么解决、负载因子与 rehash、C 实现逐行讲解、复杂度分析」。AI 应用见 `ai_application.md`。

## 1. 哈希表的定义

哈希表（Hash Table，又称散列表）是一种通过**哈希函数**把键映射到数组位置，从而实现 $O(1)$ 平均时间查找/插入/删除的数据结构。

```
put("apple", 1)  →  hash("apple") = 3  →  buckets[3] = ("apple", 1)
get("apple")     →  hash("apple") = 3  →  在 buckets[3] 找到 ("apple", 1)
```

核心思想：**用空间换时间**。开一个大小为 $M$ 的数组，用哈希函数 $h(key)$ 把键映射到 $[0, M)$，直接数组下标访问。

| 数据结构 | 查找 | 插入 | 删除 | 说明 |
|---|---|---|---|---|
| 无序数组 | $O(n)$ | $O(1)$ | $O(n)$ | 线性扫描 |
| 排序数组 | $O(\log n)$ | $O(n)$ | $O(n)$ | 二分查找 |
| 平衡树 | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | 红黑树/AVL |
| **哈希表** | **$O(1)$** | **$O(1)$** | **$O(1)$** | **平均情况** |

哈希表的 $O(1)$ 是**平均**复杂度，最坏情况（所有键冲突）退化为 $O(n)$。好的哈希函数 + 合理负载因子能让最坏情况几乎不出现。

## 2. 哈希函数

哈希函数 $h: \text{Key} \to [0, M)$ 是哈希表的核心。它必须满足：

1. **确定性**：同一个键每次算出的哈希值相同
2. **均匀分布**：不同键尽量均匀散布到 $[0, M)$
3. **高效**：计算本身是 $O(1)$

### 2.1 整数键的哈希

**除留余数法**（最简单）：

$$h(k) = k \bmod M$$

要求 $M$ 是质数，否则分布不均。例如 $M = 2^p$ 时只取低 $p$ 位，低位规律性强的键会聚集。

**乘法哈希**（Knuth）：

$$h(k) = \lfloor M \cdot (k \cdot A \bmod 1) \rfloor, \quad A = \frac{\sqrt 5 - 1}{2} \approx 0.618$$

不需要 $M$ 是质数，常取 $M = 2^p$，用位移实现。

### 2.2 字符串键的哈希

字符串是最常见的键类型。本实现用**多项式滚动哈希**（乘法哈希家族）：

```c
static size_t ht_hash(const char *key, size_t capacity) {
    size_t h = 0;
    const size_t A = 31;
    while (*key) {
        h = h * A + (unsigned char)(*key);
        key++;
    }
    return h % capacity;
}
```

数学等价于：

$$h(s) = \left(\sum_{i=0}^{L-1} s_i \cdot A^{L-1-i}\right) \bmod M$$

其中 $s_i$ 是第 $i$ 个字符的 ASCII 值，$A = 31$ 是乘数。

**为什么选 31？**
- 奇数：保证乘法不会丢失信息（偶数会让低位恒为 0）
- 质数：减少周期性模式
- 小：乘法快（编译器可能用移位优化 `31*x = (x<<5) - x`）
- Java `String.hashCode()` 用的就是 31

**为什么选 33 / 37 / 257？** 也是常见选择，257 更大但分布稍好。31 是经验上的最佳平衡点。

### 2.3 其他经典哈希函数

| 哈希函数 | 公式 | 特点 |
|---|---|---|
| DJB2 | `h = h*33 + c` | Bernstein 经典，分布好 |
| FNV-1a | `h ^= c; h *= 1099511628211` | 非加密哈希首选，冲突率极低 |
| MurmurHash3 | 复杂位移混合 | 随机性最强，工业级 |
| CRC32 | 循环冗余校验 | 原用于校验，也可做哈希 |

本实现选多项式滚动哈希是因为**教学清晰**：一眼看出"乘法 + 累加"的结构。生产环境推荐 FNV-1a 或 MurmurHash3。

## 3. 冲突解决

哈希函数把无限可能的键映射到有限的 $M$ 个桶，由鸽巢原理**冲突不可避免**。两种主流解法：

### 3.1 拉链法（Separate Chaining）

每个桶存一个**链表**，哈希到同一桶的键用链表串起来。本实现采用此法。

```
capacity = 5, 已插入: "apple"→1, "banana"→2, "cherry"→3, "date"→4, "egg"→5

假设 hash("apple")=hash("date")=1, hash("banana")=hash("egg")=3, hash("cherry")=0

buckets:
[0] → ("cherry",3) → null
[1] → ("date",4) → ("apple",1) → null
[2] → null
[3] → ("egg",5) → ("banana",2) → null
[4] → null
```

查找 `get("apple")`：
1. `idx = hash("apple") = 1`
2. 遍历 `buckets[1]` 链表，比较 key，找到 `("apple", 1)`

**复杂度**：链表长度 = 负载因子 $\alpha = n/M$，平均查找 $O(1 + \alpha)$。$\alpha$ 控制在 1 以下时近似 $O(1)$。

**优点**：
- 实现简单
- 对负载因子容忍度高（$\alpha > 1$ 也能工作，只是链表变长）
- 删除操作直接（链表删节点）

**缺点**：
- 链表节点不连续，缓存不友好（指针跳转导致 cache miss）
- 需要额外存 next 指针，空间开销大

### 3.2 开放地址法（Open Addressing）

冲突时用**探测序列**找下一个空桶，所有元素存在数组本身，不挂链表。

**线性探测**（Linear Probing）：

$$h_i(k) = (h(k) + i) \bmod M, \quad i = 0, 1, 2, \ldots$$

冲突就往后挨个找空位。

```
插入 "apple"→1, hash=1, buckets[1] 空 → 放 [1]
插入 "date"→4,  hash=1, buckets[1] 占用 → 试 [2] 空 → 放 [2]
插入 "fig"→6,   hash=1, buckets[1] 占用 → [2] 占用 → [3] 空 → 放 [3]
```

**二次探测**（Quadratic Probing）：

$$h_i(k) = (h(k) + c_1 i + c_2 i^2) \bmod M$$

减少聚集（clustering）。

**双重哈希**（Double Hashing）：

$$h_i(k) = (h_1(k) + i \cdot h_2(k)) \bmod M$$

用第二个哈希函数决定步长，聚集最少。

**优点**：
- 无指针，空间紧凑
- 数据连续存储，缓存友好（cache 性能远好于拉链法）
- $\alpha$ 必须严格 $< 1$（通常 $\le 0.7$）

**缺点**：
- 删除复杂（不能直接清空，要用"墓碑标记"tombstone，否则探测链断裂）
- $\alpha$ 接近 1 时性能急剧下降
- 实现比拉链法复杂

### 3.3 对比

| 维度 | 拉链法 | 开放地址法 |
|---|---|---|
| 实现 | 简单 | 较复杂 |
| 缓存 | 差（指针跳转） | 好（连续数组） |
| 负载因子 | 可 $> 1$ | 必须 $< 1$ |
| 删除 | 简单 | 需墓碑 |
| 空间 | 每节点多一个 next 指针 | 紧凑 |
| 适用 | 通用、教学 | 高性能、CPU 密集 |

Python `dict`、Java `HashMap`（Java 8+ 链表转红黑树）用拉链法变体；Ruby `Hash`、CPython 小 dict 用开放地址法。两种都是工业级选择。

## 4. 负载因子与 rehash

### 4.1 负载因子

$$\alpha = \frac{n}{M} = \frac{\text{已存元素数}}{\text{桶数组大小}}$$

负载因子衡量哈希表"装满程度"：
- $\alpha$ 太小：空间浪费（大部分桶空着）
- $\alpha$ 太大：冲突多，链表长，查找慢

拉链法通常 $\alpha \in [0.5, 1.0]$，开放地址法 $\alpha \in [0.5, 0.75]$。

### 4.2 rehash（扩容）

当 $\alpha$ 超过阈值（本实现 0.75），自动扩容：

1. 分配新桶数组，容量翻倍 $M' = 2M$
2. 对每个已存元素重新哈希：$idx' = h(key) \bmod M'$
3. 释放旧桶数组

**为什么不能直接搬？** 桶数变了，$h(key) \bmod M \neq h(key) \bmod M'$，每个元素的位置都要重算。

```c
static bool ht_rehash(HashTable *ht, size_t new_capacity) {
    HTNode **new_buckets = calloc(new_capacity, sizeof(HTNode *));
    for (size_t i = 0; i < ht->capacity; i++) {
        HTNode *node = ht->buckets[i];
        while (node) {
            HTNode *next = node->next;
            size_t idx = ht_hash(node->key, new_capacity);  // 重新哈希
            node->next = new_buckets[idx];                   // 头插到新桶
            new_buckets[idx] = node;
            node = next;
        }
    }
    free(ht->buckets);
    ht->buckets = new_buckets;
    ht->capacity = new_capacity;
    return true;
}
```

**均摊复杂度**：单次 rehash 是 $O(n)$，但 rehash 后能容纳 $n$ 次插入不再扩容。均摊到每次插入是 $O(1)$。

```
容量 4 → 插入 3 个触发 rehash → 容量 8
       → 插入 6 个触发 rehash → 容量 16
              → 插入 12 个触发 rehash → 容量 32
```

每次 rehash 代价翻倍，但间隔也翻倍，几何级数求和 = $O(n)$，均摊 $O(1)$。

## 5. C 实现逐行讲解

### 5.1 数据结构（hash_table.h）

```c
typedef struct HTNode {
    char *key;           // 键（字符串，动态分配副本）
    int64_t value;       // 值（64 位整数，可存 embedding 索引）
    struct HTNode *next; // 同桶下一个节点（拉链）
} HTNode;

typedef struct {
    HTNode **buckets;  // 桶数组，每个元素是链表头指针
    size_t capacity;   // 桶数 M
    size_t size;       // 已存元素数 n
    double max_load;   // 负载因子阈值（默认 0.75）
} HashTable;
```

`buckets[i]` 是一个链表头指针，指向该桶的第一个节点。空桶为 `NULL`。

### 5.2 哈希函数

```c
static size_t ht_hash(const char *key, size_t capacity) {
    size_t h = 0;
    const size_t A = 31;
    while (*key) {
        h = h * A + (unsigned char)(*key);
        key++;
    }
    return h % capacity;
}
```

逐字符遍历，`h = h * 31 + c` 滚动更新。`unsigned char` 转换避免符号位扩展。最后 `% capacity` 映射到桶范围。

### 5.3 插入 ht_put

```c
bool ht_put(HashTable *ht, const char *key, int64_t value) {
    size_t idx = ht_hash(key, ht->capacity);
    HTNode *node = ht->buckets[idx];
    while (node) {                           // 遍历桶链表
        if (strcmp(node->key, key) == 0) {   // 键已存在 → 更新值
            node->value = value;
            return true;
        }
        node = node->next;
    }
    HTNode *new_node = ht_node_new(key, value);  // 新建节点
    new_node->next = ht->buckets[idx];           // 头插法
    ht->buckets[idx] = new_node;
    ht->size++;
    if ((double)ht->size / ht->capacity > ht->max_load) {  // 超阈值 → rehash
        ht_rehash(ht, ht->capacity * 2);
    }
    return true;
}
```

关键步骤：
1. 算哈希定位桶
2. 先查桶链表，键存在则更新（保证键唯一）
3. 不存在则头插新节点（头插 $O(1)$，尾插要遍历）
4. 插完检查负载因子，超阈值就 rehash

**头插 vs 尾插**：头插 $O(1)$ 直接插入，无需遍历到尾。新插入的元素往往被频繁访问，头插让它们在链表前部，查找更快（LRU 效果）。

### 5.4 查找 ht_get

```c
bool ht_get(const HashTable *ht, const char *key, int64_t *out) {
    size_t idx = ht_hash(key, ht->capacity);
    HTNode *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {   // 找到
            if (out) *out = node->value;
            return true;
        }
        node = node->next;
    }
    return false;  // 没找到
}
```

算哈希 → 遍历桶链表 → `strcmp` 比较键。链表长度 = $\alpha$，平均 $O(1)$。

### 5.5 删除 ht_remove

```c
bool ht_remove(HashTable *ht, const char *key) {
    size_t idx = ht_hash(key, ht->capacity);
    HTNode *prev = NULL;
    HTNode *node = ht->buckets[idx];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            if (prev) prev->next = node->next;   // 中间/尾部节点
            else ht->buckets[idx] = node->next;  // 头节点
            ht_node_free(node);
            ht->size--;
            return true;
        }
        prev = node;
        node = node->next;
    }
    return false;
}
```

需要维护 `prev` 指针来跳过被删节点。头节点删除要改 `buckets[idx]`，其余改 `prev->next`。

### 5.6 释放 ht_free

```c
void ht_free(HashTable *ht) {
    for (size_t i = 0; i < ht->capacity; i++) {  // 遍历每个桶
        HTNode *node = ht->buckets[i];
        while (node) {                            // 逐个释放链表节点
            HTNode *next = node->next;
            ht_node_free(node);                   // 释放 key 和 node
            node = next;
        }
    }
    free(ht->buckets);  // 释放桶数组
    ht->buckets = NULL;
    ht->capacity = 0;
    ht->size = 0;
}
```

必须先释放所有链表节点（每个节点的 key 字符串也要释放），再释放桶数组。顺序反了会 use-after-free。

## 6. 复杂度分析

### 6.1 平均情况

假设哈希函数满足**简单均匀哈希假设**（SUHA）：每个键等概率映射到任意桶。

拉链法链表长度期望 = $\alpha = n/M$。

| 操作 | 平均复杂度 | 推导 |
|---|---|---|
| 查找 | $O(1 + \alpha)$ | 算哈希 $O(1)$ + 遍历链表 $O(\alpha)$ |
| 插入 | $O(1 + \alpha)$ | 先查重 $O(\alpha)$ + 头插 $O(1)$ |
| 删除 | $O(1 + \alpha)$ | 同查找 |

$\alpha$ 控制为常数（如 0.75）时，三个操作都是 $O(1)$。

### 6.2 最坏情况

所有键哈希到同一个桶，链表长度 $n$，退化为 $O(n)$。

**对抗**：用好的哈希函数 + rehash 控制负载因子 + 随机化哈希种子（防构造性攻击）。本实现用固定哈希函数，生产环境应加随机种子（如 Python 3 的 hash 随机化 `PYTHONHASHSEED`）。

### 6.3 rehash 的均摊

插入 $n$ 个元素到初始容量 $M_0$ 的表：

- rehash 次数：$\log_2(n/M_0)$
- 第 $i$ 次 rehash 代价：$O(M_0 \cdot 2^i)$
- 总 rehash 代价：$\sum_{i=0}^{\log_2(n/M_0)} M_0 \cdot 2^i = O(n)$
- 均摊到每次插入：$O(n)/n = O(1)$

### 6.4 空间复杂度

- 桶数组：$O(M)$
- 链表节点：$O(n)$（每节点存 key + value + next 指针）
- 总计：$O(M + n)$，$\alpha = n/M$ 时 $O(n)$

## 7. 与其他结构的对比

| 维度 | 哈希表 | 平衡树 | 跳表 |
|---|---|---|---|
| 查找 | $O(1)$ 平均 | $O(\log n)$ | $O(\log n)$ |
| 有序遍历 | 不支持 | 支持 | 支持 |
| 范围查询 | 不支持 | $O(\log n + k)$ | $O(\log n + k)$ |
| 最值 | 不支持 | $O(1)$（首尾） | $O(1)$ |
| 实现复杂度 | 中 | 高 | 中 |
| 缓存友好 | 中（拉链差，开放好） | 差 | 中 |

**选哈希表**：只需等值查找，不要有序遍历（如词嵌入查找、缓存、去重）。
**选树**：需要有序遍历、范围查询、最值（如数据库索引、调度队列）。

## 8. 本实现的局限与改进方向

1. **键类型固定为字符串**：生产级用泛型（C 用 `void*` + 用户提供 hash/eq 函数）
2. **哈希函数固定**：无法换 FNV/Murmur，生产级应可配置
3. **无迭代器**：无法遍历所有键值对
4. **无线程安全**：多线程需加锁或用分段锁
5. **拉链法缓存差**：高频场景可换开放地址法
6. **固定 rehash 策略**：只翻倍，不缩容（删除多时空间浪费）

这些是教学实现的合理取舍，生产级哈希表（如 Google `absl::flat_hash_map`）会逐一解决。

## 9. 小结

哈希表的核心三件套：
1. **哈希函数**：把键映射到 $[0, M)$，要快且均匀
2. **冲突解决**：拉链法（本实现）或开放地址法
3. **负载因子 + rehash**：保持 $\alpha$ 在阈值内，保证 $O(1)$

理解了这三点，就理解了哈希表。下一章 `ai_application.md` 讲它怎么在 AI 里做词嵌入查找和 MoE 路由。