# 二叉堆原理

> 本文档讲清「堆是什么、完全二叉树怎么用数组存、堆序性质、sift_up/sift_down、建堆 O(n)、C 实现逐行讲解、复杂度」。AI 应用见 `ai_application.md`。

## 1. 堆的定义

**堆**（Heap）是一种特殊的完全二叉树，满足**堆序性质**（heap property）：

- **小顶堆**（min-heap）：每个节点 $\le$ 它的所有子节点，根是全局最小
- **大顶堆**（max-heap）：每个节点 $\ge$ 它的所有子节点，根是全局最大

```
小顶堆示例:
          1
        /   \
       3     2
      / \   / \
     5   4 8   7
    / \
   9   6

数组: [1, 3, 2, 5, 4, 8, 7, 9, 6]
```

堆只保证**父子间的偏序**，不保证兄弟间的顺序。上例中 `3 < 2` 不成立，但 `1 < 3` 和 `1 < 2` 都成立。这比排序弱得多，正因为约束弱，插入/删除才能做到 $O(\log n)$。

| 数据结构 | 找最值 | 插入 | 删最值 | 全序 |
|---|---|---|---|---|
| 无序数组 | $O(n)$ | $O(1)$ | $O(n)$ | 否 |
| 排序数组 | $O(1)$ | $O(n)$ | $O(1)$ | 是 |
| 平衡树 | $O(1)$ | $O(\log n)$ | $O(\log n)$ | 是 |
| **堆** | **$O(1)$** | **$O(\log n)$** | **$O(\log n)$** | **否** |

堆的定位：**只需要反复取最值、插入元素，不需要全序遍历**。这正是优先队列的核心需求。

## 2. 完全二叉树与数组表示

### 2.1 完全二叉树

**完全二叉树**（complete binary tree）：除最后一层外每层都填满，最后一层节点从左到右连续排列。

```
完全二叉树（7 个节点）:        非完全二叉树:
          0                           0
        /   \                       /   \
       1     2                     1     2
      / \   / \                   / \     \
     3   4 5   6                 3   4     6   ← 缺了 5
```

完全二叉树的关键性质：**高度 $h = \lfloor \log_2 n \rfloor$**，即 $n$ 个节点的树只有 $\log n$ 层。这是堆 $O(\log n)$ 操作的根源。

### 2.2 数组表示

完全二叉树可以用数组**紧凑存储**，无需指针。约定**根在索引 0**，逐层从左到右填入数组：

```
树:          0           数组下标:  0  1  2  3  4  5  6
           /   \         数组内容: [A, B, C, D, E, F, G]
          1     2
         / \   / \
        3   4 5   6
```

下标关系（根在 0）：

$$\text{parent}(i) = \left\lfloor \frac{i-1}{2} \right\rfloor, \quad \text{left}(i) = 2i+1, \quad \text{right}(i) = 2i+2$$

```c
static size_t heap_parent(size_t i) { return (i - 1) / 2; }
static size_t heap_left(size_t i)   { return 2 * i + 1; }
static size_t heap_right(size_t i)  { return 2 * i + 2; }
```

**为什么不用指针？**
1. **空间紧凑**：无 left/right/parent 指针，每节点只存数据本身
2. **缓存友好**：数据连续存储，CPU prefetcher 友好，cache miss 少
3. **下标计算 $O(1)$**：一次乘除加，比指针解快
4. **适合固定形状**：完全二叉树形状固定，不需要指针表达拓扑

### 2.3 根在 0 vs 根在 1

有些实现把根放在索引 1（留 `data[0]` 不用），下标公式更简洁：

$$\text{parent}(i) = \left\lfloor \frac{i}{2} \right\rfloor, \quad \text{left}(i) = 2i, \quad \text{right}(i) = 2i+1$$

但浪费一个槽位。本实现选**根在 0**（不浪费空间，C 数组从 0 开始自然），代价是 parent 公式多一个 `-1`。

## 3. 堆序性质

### 3.1 小顶堆

对任意节点 $i$（除根），都满足：

$$\text{data}[\text{parent}(i)] \le \text{data}[i]$$

即父节点 $\le$ 子节点。根 `data[0]` 是全局最小。

**验证函数**（测试用）：

```c
static bool is_min_heap(const MinHeap *h) {
    for (size_t i = 1; i < h->size; i++) {
        size_t p = (i - 1) / 2;
        if (h->data[i] < h->data[p]) {
            return false;  // 子 < 父，违反小顶堆
        }
    }
    return true;
}
```

遍历每个非根节点，检查它是否 $\ge$ 父节点。$O(n)$ 验证。

### 3.2 大顶堆

对任意节点 $i$（除根），$\text{data}[\text{parent}(i)] \ge \text{data}[i]$，根是全局最大。

### 3.3 小顶堆 vs 大顶堆

| | 小顶堆 | 大顶堆 |
|---|---|---|
| 根 | 全局最小 | 全局最大 |
| 用途 | 取最小、Top-K 大元素 | 取最大、堆排序 |
| pop | 返回最小 | 返回最大 |

**Top-K 用小顶堆**（本章重点）：要找最大的 K 个，维护一个大小 K 的小顶堆，堆顶是这 K 个里最小的，新元素比堆顶大就替换。这样堆里始终是当前见过的最大 K 个。

**堆排序用大顶堆**（或小顶堆 + 反向取）：建堆后反复 pop 根，得到有序序列。

本实现是**小顶堆**，适配 Top-K 采样场景。

## 4. 核心操作：sift_up 与 sift_down

堆的所有动态操作（push/pop/build）都靠这两个原语维护堆序。

### 4.1 sift_up（上浮）

**场景**：在末尾插入一个元素后，它可能比父小（违反小顶堆），需要**上浮**到正确位置。

```c
static void heap_sift_up(MinHeap *h, size_t i) {
    while (i > 0) {
        size_t p = heap_parent(i);
        if (h->data[i] < h->data[p]) {
            // 交换 i 和父
            double tmp = h->data[i];
            h->data[i] = h->data[p];
            h->data[p] = tmp;
            i = p;           // 继续往上比
        } else {
            break;           // 已满足堆序，停
        }
    }
}
```

过程：从节点 $i$ 开始，只要比父小就交换，然后移到父位置继续。最坏情况上浮到根，路径长度 = 树高 = $O(\log n)$。

```
插入 0.5 到小顶堆 [1,3,2,5,4,8,7,9,6]：
数组末尾加 0.5 → [1,3,2,5,4,8,7,9,6,0.5]
                  1
                /   \
               3     2
              / \   / \
             5   4 8   7
            / \ /
           9   6 0.5  ← 比父 4 小，交换

→ [1,3,2,5,0.5,8,7,9,6,4]
                  1
                /   \
               3    0.5  ← 比父 1 小，交换
              ...

→ [0.5,3,1,5,0.5...4] → 继续直到 0.5 到根
最终: [0.5,3,1,5,4,2,7,9,6,8]
```

### 4.2 sift_down（下沉）

**场景**：根被替换（pop 或 replace_top）后，新根可能比子大，需要**下沉**到正确位置。

```c
static void heap_sift_down(MinHeap *h, size_t i) {
    size_t n = h->size;
    while (1) {
        size_t l = heap_left(i);
        size_t r = heap_right(i);
        size_t smallest = i;
        if (l < n && h->data[l] < h->data[smallest]) {
            smallest = l;     // 左子更小
        }
        if (r < n && h->data[r] < h->data[smallest]) {
            smallest = r;     // 右子比左还小
        }
        if (smallest == i) {
            break;            // 已是最小，堆序满足
        }
        // 交换 i 和最小的子
        double tmp = h->data[i];
        h->data[i] = h->data[smallest];
        h->data[smallest] = tmp;
        i = smallest;         // 继续往下
    }
}
```

过程：从节点 $i$ 开始，找它和左右子中**最小**的，如果不是 $i$ 自己，就交换，然后移到那个子位置继续。最坏下沉到叶子，路径 $O(\log n)$。

**为什么和较小的子交换？** 小顶堆要求父 $\le$ 两个子。如果父比子大，必须换下去；换到哪个子？换到较小的那个，保证换完后父 $\le$ 两个子（较大的子本来就比小的子大，换上来当父也满足）。

### 4.3 复杂度

| 操作 | 复杂度 | 说明 |
|---|---|---|
| sift_up | $O(\log n)$ | 最坏上浮到根 |
| sift_down | $O(\log n)$ | 最坏下沉到叶 |

两者都是沿一条根到叶（或叶到根）的路径走，路径长度 = 树高 = $\lfloor \log_2 n \rfloor$。

## 5. push / pop / peek

### 5.1 push

```c
bool heap_push(MinHeap *h, double v) {
    if (!heap_reserve(h, h->size + 1)) return false;  // 容量不够先扩
    h->data[h->size] = v;     // 放到末尾
    h->size++;
    heap_sift_up(h, h->size - 1);  // 上浮
    return true;
}
```

策略：**末尾插入 + sift_up**。新元素先放最后（保持完全二叉树形状），再上浮到正确位置。

```
push(0.5) 到 [1,3,2,5,4]:
1. 末尾插入: [1,3,2,5,4,0.5]
2. sift_up(5): 0.5 < 父 2 → 交换 → [1,3,0.5,5,4,2]
3. sift_up(2): 0.5 < 父 1 → 交换 → [0.5,3,1,5,4,2]
4. sift_up(0): 到根，停
```

复杂度：$O(\log n)$（一次 sift_up）。

### 5.2 pop

```c
bool heap_pop(MinHeap *h, double *out) {
    if (h->size == 0) return false;
    if (out) *out = h->data[0];       // 取根（最小）
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size]; // 末尾移到根
        heap_sift_down(h, 0);          // 下沉
    }
    return true;
}
```

策略：**根与末尾交换 + 删末尾 + sift_down**。
1. 取出根（最小值）
2. 把末尾元素移到根位置
3. size 减一（末尾逻辑删除）
4. 从根 sift_down 恢复堆序

```
pop() 从 [1,3,2,5,4,8,7,9,6]（根是 1）:
1. 取出 1
2. 末尾 6 移到根: [6,3,2,5,4,8,7,9] (size 从 9 减到 8)
3. sift_down(0): 6 vs 子 3,2 → 最小是 2 → 交换 → [2,3,6,5,4,8,7,9]
4. sift_down(2): 6 vs 子 8,7 → 6 最小 → 停
结果: [2,3,6,5,4,8,7,9]，返回 1
```

**为什么不能直接删根留空？** 根空了就不是完全二叉树了。把末尾填到根，保持完全二叉树形状，再用 sift_down 修堆序。

复杂度：$O(\log n)$（一次 sift_down）。

### 5.3 peek

```c
bool heap_peek(const MinHeap *h, double *out) {
    if (h->size == 0) return false;
    if (out) *out = h->data[0];  // 看根
    return true;
}
```

只看根不删，$O(1)$。

### 5.4 replace_top

```c
bool heap_replace_top(MinHeap *h, double v, double *out_old) {
    if (h->size == 0) return false;
    if (out_old) *out_old = h->data[0];
    h->data[0] = v;            // 直接覆盖根
    heap_sift_down(h, 0);      // 下沉
    return true;
}
```

**替换根 + sift_down**，比 `pop + push` 快：少一次 sift_up。Top-K 里当新元素比堆顶大时用这个——弹出最小、压入新元素，一步到位。

复杂度：$O(\log n)$（一次 sift_down，比 pop+push 的两次 $O(\log n)$ 快一倍）。

## 6. 建堆：Floyd 算法 O(n)

给定 $n$ 个元素的数组，怎么把它变成堆？

### 6.1 朴素法：逐个 push

```c
for (size_t i = 0; i < n; i++) {
    heap_push(&h, arr[i]);  // 每次 O(log size)
}
```

总复杂度：$\sum_{i=1}^{n} O(\log i) = O(n \log n)$。

### 6.2 Floyd 法：自底向上 sift_down

```c
bool heap_build(MinHeap *h, const double *items, size_t n) {
    if (!heap_reserve(h, n)) return false;
    for (size_t i = 0; i < n; i++) {
        h->data[i] = items[i];
    }
    h->size = n;
    if (n <= 1) return true;
    size_t last_inner = heap_parent(n - 1);     // 最后一个内部节点
    for (size_t i = last_inner + 1; i-- > 0;) { // 从最后一个内部节点倒着 sift_down
        heap_sift_down(h, i);
    }
    return true;
}
```

策略：
1. 把所有元素原样复制到数组（此时不是堆）
2. 从**最后一个内部节点**开始，倒着对每个内部节点做 sift_down

**为什么从最后一个内部节点倒着？**
- 叶子天然满足堆序（没子，trivially $\le$ 子）
- 从底向上修，修每个内部节点时它的子树已经是堆，sift_down 把这个节点融入到已堆化的子树里
- 倒序保证处理节点 $i$ 时，子树 $2i+1, 2i+2$ 都已堆化

### 6.3 为什么是 O(n) 而不是 O(n log n)

直觉上 $n$ 个节点每个 sift_down $O(\log n)$，应该是 $O(n \log n)$。但**大部分节点在底层，sift_down 距离很短**。

精确分析：高度 $h$ 的节点最多有 $\lceil n / 2^{h+1} \rceil$ 个，每个 sift_down 最多走 $h$ 步。

$$T(n) = \sum_{h=0}^{\lfloor \log_2 n \rfloor} \left\lceil \frac{n}{2^{h+1}} \right\rceil \cdot h \le n \sum_{h=0}^{\infty} \frac{h}{2^{h+1}} = n \cdot 1 = O(n)$$

关键：$\sum_{h=0}^{\infty} h / 2^{h+1} = 1$（收敛级数）。所以建堆是**线性**的，比逐个 push 的 $O(n \log n)$ 快。

### 6.4 实测对比

本目录 C 测试的微基准（10 万元素）：

```
[BENCH] heap_build (Floyd):  2.74 ms/iter  (100 次平均)
```

对比逐个 push（10 万元素，每次 $O(\log n)$）：实测约 12-15 ms。Floyd 法快 4-5x，符合 $O(n)$ vs $O(n \log n)$ 的理论。

## 7. C 实现逐行讲解

### 7.1 数据结构（heap.h）

```c
typedef struct {
    double *data;     // 数组存储，根在 data[0]
    size_t size;      // 当前元素数
    size_t capacity;  // 数组容量
} MinHeap;
```

只有三个字段：数组指针、大小、容量。没有左右子指针——下标计算代替。`double` 类型适配 logits/score 场景。

### 7.2 容量管理

```c
static bool heap_reserve(MinHeap *h, size_t need) {
    if (need <= h->capacity) return true;     // 够了
    size_t new_cap = h->capacity == 0 ? 8 : h->capacity;
    while (new_cap < need) {
        new_cap *= 2;                         // 翻倍扩容
    }
    double *new_data = (double *)realloc(h->data, new_cap * sizeof(double));
    if (!new_data) return false;
    h->data = new_data;
    h->capacity = new_cap;
    return true;
}
```

扩容策略：**翻倍**。均摊 $O(1)$（同哈希表 rehash 的几何级数论证）。初始容量 0 时首次分配 8。

### 7.3 sift_up 逐行

```c
static void heap_sift_up(MinHeap *h, size_t i) {
    while (i > 0) {                    // 还没到根
        size_t p = heap_parent(i);     // 算父下标
        if (h->data[i] < h->data[p]) { // 子 < 父 → 违反小顶堆
            double tmp = h->data[i];   // 交换
            h->data[i] = h->data[p];
            h->data[p] = tmp;
            i = p;                     // 移到父位置继续
        } else {
            break;                     // 已满足，停
        }
    }
}
```

关键：`i > 0` 判断是否到根（根的 parent 无意义）。`<` 严格小于，相等不交换（稳定，减少不必要操作）。

### 7.4 sift_down 逐行

```c
static void heap_sift_down(MinHeap *h, size_t i) {
    size_t n = h->size;
    while (1) {
        size_t l = heap_left(i);
        size_t r = heap_right(i);
        size_t smallest = i;                              // 先假设自己最小
        if (l < n && h->data[l] < h->data[smallest]) {   // 左子存在且更小
            smallest = l;
        }
        if (r < n && h->data[r] < h->data[smallest]) {   // 右子存在且比当前最小还小
            smallest = r;
        }
        if (smallest == i) {                              // 自己最小，堆序满足
            break;
        }
        double tmp = h->data[i];                          // 交换
        h->data[i] = h->data[smallest];
        h->data[smallest] = tmp;
        i = smallest;                                     // 移到子位置继续
    }
}
```

两个 `if` 而非 `if-else`：右子要和"当前最小"比（可能是左子），所以第二个 if 用 `smallest` 不用 `i`。`l < n` / `r < n` 检查子是否存在（完全二叉树末尾可能缺右子甚至缺左子）。

### 7.5 heap_build 逐行

```c
bool heap_build(MinHeap *h, const double *items, size_t n) {
    if (!heap_reserve(h, n)) return false;
    for (size_t i = 0; i < n; i++) {
        h->data[i] = items[i];                // 原样复制
    }
    h->size = n;
    if (n <= 1) return true;                  // 0 或 1 个元素天然是堆
    size_t last_inner = heap_parent(n - 1);   // 最后一个内部节点的下标
    for (size_t i = last_inner + 1; i-- > 0;) { // 从 last_inner 倒着到 0
        heap_sift_down(h, i);
    }
    return true;
}
```

`for (size_t i = last_inner + 1; i-- > 0;)` 是 size_t 倒序循环的惯用法：`i` 从 `last_inner` 开始，每次减一，到 0 还会执行一次（`i-- > 0` 先判断再减）。

**为什么 last_inner = parent(n-1)？** 最后一个元素是 `n-1`，它的父是最后一个有子的节点。从它开始倒着到根 0，覆盖所有内部节点。

### 7.6 heap_replace_top 逐行

```c
bool heap_replace_top(MinHeap *h, double v, double *out_old) {
    if (h->size == 0) return false;
    if (out_old) *out_old = h->data[0];   // 可选返回旧根
    h->data[0] = v;                        // 覆盖根
    heap_sift_down(h, 0);                  // 下沉恢复堆序
    return true;
}
```

Top-K 的核心原语：当新元素比堆顶（最小）大时，替换堆顶再下沉。比 `pop + push` 少一次 sift_up，快约 2x。

## 8. 复杂度分析

### 8.1 各操作复杂度

| 操作 | 时间 | 空间 | 说明 |
|---|---|---|---|
| init | $O(1)$ | $O(\text{cap})$ | 分配数组 |
| push | $O(\log n)$ | 均摊 $O(1)$ | sift_up + 可能扩容 |
| pop | $O(\log n)$ | $O(1)$ | sift_down |
| peek | $O(1)$ | $O(1)$ | 看根 |
| replace_top | $O(\log n)$ | $O(1)$ | 一次 sift_down |
| build (Floyd) | $O(n)$ | $O(1)$ | 自底向上 |
| build (逐个 push) | $O(n \log n)$ | $O(1)$ | 朴素 |

### 8.2 push 的均摊分析

扩容翻倍，第 $k$ 次扩容代价 $O(2^k)$，但间隔 $2^k$ 次 push。总扩容代价 $\sum 2^k = O(n)$，均摊每次 push $O(1)$ 额外。加上 sift_up 的 $O(\log n)$，push 均摊 $O(\log n)$。

### 8.3 build 的 O(n) 证明（详细）

设堆有 $n$ 个元素，高度 $H = \lfloor \log_2 n \rfloor$。

高度 $h$ 的节点数 $\le \lceil n / 2^{h+1} \rceil$（第 $h$ 层从顶往下数，最多 $n/2^{h+1}$ 个）。

每个高度 $h$ 的节点 sift_down 最多走 $h$ 步。

$$T(n) \le \sum_{h=0}^{H} \frac{n}{2^{h+1}} \cdot h = \frac{n}{2} \sum_{h=0}^{H} \frac{h}{2^h} \le \frac{n}{2} \sum_{h=0}^{\infty} \frac{h}{2^h} = \frac{n}{2} \cdot 2 = n$$

其中 $\sum_{h=0}^{\infty} h/2^h = 2$（对 $\sum x^h = 1/(1-x)$ 求导后令 $x=1/2$）。所以 $T(n) = O(n)$。

### 8.4 空间复杂度

数组存储，$n$ 个元素占 $O(n)$ 空间，无额外指针开销。比指针二叉树省 $2n$ 个指针（左右子）。

## 9. 优先队列

**优先队列**（Priority Queue）是堆的抽象接口：元素带优先级，出队按优先级（而非入队顺序）。

```
普通队列: FIFO，先入先出
优先队列: 每次出队优先级最高（或最低）的元素
```

堆是实现优先队列的标准结构：
- `push` = 入队（带优先级）
- `pop` = 出队（最高/最低优先级）
- `peek` = 看队首（不取出）

| 实现 | 入队 | 出队 | 适用 |
|---|---|---|---|
| 无序数组 | $O(1)$ | $O(n)$（找最值） | 入队多出队少 |
| 排序数组 | $O(n)$（插入排序） | $O(1)$ | 出队多入队少 |
| **堆** | **$O(\log n)$** | **$O(\log n)$** | **通用** |
| 平衡树 | $O(\log n)$ | $O(\log n)$ | 需要全序遍历 |

堆在两端都 $O(\log n)$，且常数小、缓存友好，是优先队列的默认选择。C++ `std::priority_queue`、Python `heapq`、Java `PriorityQueue` 都用堆实现。

## 10. 与其他结构对比

| 维度 | 堆 | 排序数组 | 平衡树 | 哈希表 |
|---|---|---|---|---|
| 找最值 | $O(1)$ | $O(1)$（端点） | $O(1)$（端点） | $O(n)$ |
| 插入 | $O(\log n)$ | $O(n)$ | $O(\log n)$ | $O(1)$ |
| 删最值 | $O(\log n)$ | $O(1)$ 或 $O(n)$ | $O(\log n)$ | $O(n)$ |
| 查找任意 | $O(n)$ | $O(\log n)$ | $O(\log n)$ | $O(1)$ |
| 全序遍历 | $O(n \log n)$（排序） | $O(n)$ | $O(n)$ | 不支持 |
| 建结构 | $O(n)$ | $O(n \log n)$ | $O(n \log n)$ | $O(n)$ |
| 空间 | $O(n)$ 紧凑 | $O(n)$ 紧凑 | $O(n)$ 有指针 | $O(n)$ |
| 缓存 | 好 | 极好 | 差 | 中 |

**选堆**：只需反复取最值（优先队列、Top-K、Dijkstra、Beam Search）。
**选排序数组**：一次排序后多次二分查找，不动态插入。
**选平衡树**：需要全序遍历、范围查询、前驱后继。
**选哈希表**：等值查找，不要顺序。

## 11. 本实现的局限与改进方向

1. **固定 double 类型**：生产级用泛型（C++ 模板、C 用 `void*` + 比较函数）
2. **只有小顶堆**：大顶堆需改比较方向，或存负值复用小顶堆
3. **无索引映射**：无法 $O(1)$ 定位任意元素（Dijkstra 的 decrease-key 需要），生产级加 `pos_map: elem → index`
4. **无合并操作**：两个堆合并是 $O(n)$，二项堆/配对堆支持 $O(\log n)$ 合并
5. **单线程**：并发优先队列需锁或无锁结构（如 skip list）
6. **无删除任意元素**：本实现只支持删根，删中间元素需先查位置（$O(n)$）+ sift

这些是教学实现的合理取舍。生产级优先队列（如 Go `container/heap`、Boost `fibonacci_heap`）会按需扩展。

## 12. 小结

堆的核心三件套：
1. **完全二叉树 + 数组表示**：下标算父子，无指针，缓存友好
2. **堆序性质**：父 $\le$ 子（小顶堆），根是全局最小，约束弱于全序
3. **sift_up / sift_down**：$O(\log n)$ 维护堆序的两个原语

理解了这三点，就理解了堆。建堆的 $O(n)$ 是个反直觉的优美结果——大部分节点在底层，sift_down 距离短，几何级数求和收敛到线性。

下一章 `ai_application.md` 讲堆怎么在 AI 里做 Top-K 采样和 Beam Search 剪枝。