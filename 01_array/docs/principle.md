# 动态数组原理

> 本文档讲清「动态数组是什么、怎么实现、复杂度多少」。配套的 AI 应用文档见 `ai_application.md`。

## 1. 数组的数学定义

数组是最简单的线性表，数学上是一个从下标到元素的映射：

$$
A: \{0, 1, \dots, n-1\} \to V
$$

其中 $n$ 是数组长度，$V$ 是元素集合。支持两个核心操作：

- **随机访问**：给定 $i$，返回 $A(i)$，时间 $O(1)$
- **存储**：给定 $i$ 和 $v$，令 $A(i) := v$，时间 $O(1)$

这两个 $O(1)$ 是数组一切优势的根源，而它之所以能做到 $O(1)$，是因为**元素在内存里连续存放**。

## 2. 内存布局：为什么连续存放就能 O(1) 访问

### 2.1 内存的地址模型

内存可以看作一个巨大的字节数组 `M[0], M[1], ..., M[2^k - 1]`，其中 $k$ 是地址总线宽度（64 位机器上 $k=64$，但实际可用远小于此）。

每个内存地址指向一个字节。一个 `int`（通常 4 字节）占用 4 个连续地址：

```
地址:  100   101   102   103   104   105   106   107
内容:  [  int A[0]  ]    [  int A[1]  ]
```

### 2.2 数组的寻址公式

如果数组 `A` 的首元素放在地址 `base`，每个元素占 `s` 字节，则 `A[i]` 的地址是：

$$
\text{addr}(A[i]) = \text{base} + i \times s
$$

这个公式是一次乘法 + 一次加法，**与 $n$ 无关**，所以是 $O(1)$。

**关键**：这个 $O(1)$ 完全依赖「元素连续存放」。如果元素散落在内存各处（如链表），就必须顺着指针逐个找，变成 $O(n)$。

### 2.3 缓存友好性：连续存放的隐藏收益

现代 CPU 有多级缓存（L1/L2/L3），访问缓存比访问主存快 100 倍以上：

| 层级 | 容量 | 延迟 |
|---|---|---|
| 寄存器 | 几十个 | <1ns |
| L1 缓存 | 32~64 KB | ~1ns |
| L2 缓存 | 256KB~1MB | ~4ns |
| L3 缓存 | 几 MB~几十 MB | ~12ns |
| 主存 | GB 级 | ~100ns |

CPU 加载数据到缓存时，不是按字节加载，而是按**缓存行**（通常 64 字节）加载。这意味着：

> **访问 `A[0]` 时，`A[1]`、`A[2]`、...、`A[15]`（假设 int 4 字节）已经被免费加载到缓存里了。**

这就是连续存放的隐藏收益：**顺序访问数组时，几乎每次都命中缓存**。链表则相反，每个节点的下一个节点在哪完全不确定，几乎每次都 miss。

### 2.4 实测：顺序访问 vs 随机访问

一个经典实验：同样大小的数组，顺序遍历 vs 随机跳跃遍历，测速差。

```python
import numpy as np
import time

n = 10_000_000
arr = np.arange(n, dtype=np.int64)

# 顺序访问
t = time.perf_counter()
s = arr.sum()
t_seq = time.perf_counter() - t

# 随机跳跃访问（每次跨一个大步，破坏缓存）
indices = np.arange(0, n, 4096, dtype=np.int64)  # 跨 32KB
t = time.perf_counter()
s = arr[indices].sum()
t_rand = time.perf_counter() - t

print(f"顺序: {t_seq:.3f}s, 跳跃: {t_rand:.3f}s")
```

顺序访问通常快 5~10 倍，这就是缓存的力量，也是数组相对链表的核心优势之一。

## 3. 静态数组 vs 动态数组

### 3.1 静态数组

C 里的 `int arr[100]` 是静态数组：

```c
int arr[100];  // 编译时就定死大小，不能变
```

- 优点：分配在栈上，快；没有扩容开销
- 缺点：大小固定，不能变；大了浪费、小了溢出

### 3.2 动态数组

动态数组在堆上分配，可以按需扩容：

```c
typedef struct {
    int *data;       // 指向堆上的连续内存
    size_t size;     // 当前元素个数
    size_t capacity; // 已分配的容量
} DynArray;
```

- `size`：实际存了多少元素
- `capacity`：底层数组能存多少元素
- 不变式：`size <= capacity`
- 当 `size == capacity` 时再 append，先扩容（`capacity` 翻倍），再存

### 3.3 为什么不一开始就分配很大

- 浪费内存：你不知道最终要存多少
- 分配大块内存慢：`malloc(1GB)` 比 `malloc(1MB)` 慢得多
- 按需增长才能适应从 0 到任意大小的场景

## 4. 扩容策略：为什么是倍增

### 4.1 倍增策略

当 `size == capacity` 时，把 `capacity` 翻倍：

```c
size_t new_cap = a->capacity == 0 ? 8 : a->capacity * 2;
da_resize(a, new_cap);
```

### 4.2 为什么倍增而不是 +1 或 +k

如果每次扩容只 +1：

```
容量: 0 → 1 → 2 → 3 → 4 → ... → n
每次扩容要复制旧元素: 0 + 1 + 2 + 3 + ... + n-1 = n(n-1)/2 = O(n^2)
```

n 次 append 的总开销是 $O(n^2)$，单次 append 摊还 $O(n)$，不可接受。

如果每次倍增：

```
容量: 0 → 1 → 2 → 4 → 8 → 16 → ... → 2^k (其中 2^k >= n)
每次扩容复制: 1 + 2 + 4 + 8 + ... + 2^k = 2^(k+1) - 1 = O(n)
```

n 次 append 的总开销是 $O(n)$，单次 append 摊还 $O(1)$。

### 4.3 摊还分析

「摊还」的意思是：单次 append 可能触发扩容（慢，$O(n)$），但大多数 append 不触发（快，$O(1)$）。把扩容的开销摊到所有 append 上，平均每次 $O(1)$。

形式化：考虑从空数组开始连续 n 次 append。

- 扩容发生在 `size = 1, 2, 4, 8, ..., 2^k`（其中 $2^k \le n < 2^{k+1}$）
- 每次扩容到 `capacity = 2, 4, 8, ..., 2^{k+1}`，复制开销 `1, 2, 4, ..., 2^k`
- 总复制开销：$1 + 2 + 4 + \dots + 2^k = 2^{k+1} - 1 < 2n$
- 加上 n 次 append 本身的 $O(1)$ 写入：总开销 $O(n)$
- 摊还到每次 append：$O(n) / n = O(1)$

### 4.4 倍增因子为什么是 2

倍增因子 $\alpha$ 的选择是时间和空间的权衡：

- $\alpha = 2$（倍增）：摊还 $O(1)$，空间利用率最坏 50%（刚扩容后 size = capacity/2）
- $\alpha = 1.5$（增长 50%）：摊还 $O(1)$，空间利用率最坏 66%
- $\alpha$ 越大：扩容次数越少，但空间浪费越多
- $\alpha$ 越小：空间利用率越高，但扩容次数越多

实践中 1.5~2 都常见。Python list 用的是 `new_cap = old_cap + (old_cap >> 3) + 3`（约 1.125 倍 + 常数），偏向空间利用率。Java ArrayList 用 1.5 倍。C++ vector 通常用 2 倍。

本项目用 2 倍，最简单清晰。

## 5. C 实现逐行讲解

### 5.1 结构定义

```c
typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} DynArray;
```

- `data`：指向堆上连续内存的指针，这是数组的核心
- `size`：当前元素个数，初始 0
- `capacity`：已分配容量，初始 0

### 5.2 初始化

```c
void da_init(DynArray *a) {
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}
```

初始化时不分配内存（懒分配）。第一次 append 时才分配 `DA_INIT_CAP = 8` 个元素。这样空数组的开销是 0。

### 5.3 释放

```c
void da_free(DynArray *a) {
    free(a->data);
    a->data = NULL;
    a->size = 0;
    a->capacity = 0;
}
```

释放底层数组，重置字段。注意 `free(NULL)` 是安全的（标准保证），所以即使没分配过也能调 `da_free`。

### 5.4 扩容

```c
bool da_resize(DynArray *a, size_t new_cap) {
    if (new_cap == 0) {
        free(a->data);
        a->data = NULL;
        a->size = 0;
        a->capacity = 0;
        return true;
    }
    int *p = realloc(a->data, new_cap * sizeof(int));
    if (!p) return false;
    a->data = p;
    a->capacity = new_cap;
    if (a->size > new_cap) a->size = new_cap;
    return true;
}
```

- `realloc`：尝试原地扩容；不行就分配新块、复制旧数据、释放旧块
- `realloc` 可能返回新地址，所以必须用返回值更新 `a->data`
- `realloc` 可能失败（返回 NULL），必须检查
- 如果 `new_cap < size`，截断 `size`

### 5.5 追加

```c
bool da_append(DynArray *a, int v) {
    if (a->size == a->capacity) {
        size_t new_cap = a->capacity == 0 ? DA_INIT_CAP : a->capacity * 2;
        if (!da_resize(a, new_cap)) return false;
    }
    a->data[a->size++] = v;
    return true;
}
```

- 检查是否满：`size == capacity`
- 满了就扩容：空数组给初始容量 8，否则翻倍
- 扩容后保证 `size < capacity`，可以安全写入
- 写入 `data[size]`，然后 `size++`

### 5.6 其他操作

```c
int da_get(const DynArray *a, size_t i) {
    return a->data[i];
}
```

`da_get` 就是数组下标访问，$O(1)$。注意没有越界检查（生产代码应该有，但教学代码省略以突出核心）。

```c
bool da_pop(DynArray *a, int *out) {
    if (a->size == 0) return false;
    if (out) *out = a->data[--a->size];
    else a->size--;
    return true;
}
```

`da_pop` 删除末尾元素，$O(1)$。不缩容（缩容的策略更复杂，通常也不做）。

## 6. 复杂度汇总

| 操作 | 时间复杂度 | 备注 |
|---|---|---|
| `da_get(i)` | $O(1)$ | 下标访问 |
| `da_set(i, v)` | $O(1)$ | 下标访问 |
| `da_append(v)` | $O(1)$ 摊还 | 偶尔 $O(n)$ 触发扩容 |
| `da_pop()` | $O(1)$ | 删末尾 |
| `da_insert(i, v)` | $O(n)$ | 要移动 i 后所有元素 |
| `da_remove(i)` | $O(n)$ | 要移动 i 后所有元素 |
| 空间 | $O(\text{capacity})$ | $\text{size} \le \text{capacity} < 2 \times \text{size}$ |

**注意**：在中间插入/删除是 $O(n)$，因为要保持连续存放，必须移动后面所有元素。这是数组相对链表的主要劣势。

## 7. 数组 vs 链表

| 操作 | 数组 | 链表 |
|---|---|---|
| 随机访问 `A[i]` | $O(1)$ | $O(n)$ |
| 头部插入 | $O(n)$ | $O(1)$ |
| 尾部插入 | $O(1)$ 摊还 | $O(1)$ |
| 中间插入 | $O(n)$ | $O(1)$（已知节点） |
| 遍历 | 缓存友好 | 缓存不友好 |
| 内存开销 | 仅元素 | 元素 + 指针 |
| 空间局部性 | 好 | 差 |

**数组的优势在「随机访问 + 遍历」，链表的优势在「频繁中间插入删除」**。

AI 里几乎全用数组（张量就是数组），因为：
1. 矩阵乘法要随机访问
2. 大规模遍历要缓存友好
3. GPU 要求连续内存才能并行

链表在 AI 里几乎只用于 LRU 缓存（第 02 章）等少数场景。

## 8. 与 Python list 的关系

Python 的 `list` 本质就是动态数组，但存的是指针（每个元素是一个 `PyObject*`）：

```python
# Python list 的 CPython 实现简化
typedef struct {
    PyObject **data;  // 指针数组
    Py_ssize_t size;
    Py_ssize_t capacity;
} PyListObject;
```

- `data` 是 `PyObject*` 的数组，每个元素是指向真实对象的指针
- 所以 Python list 里存的是「指针」，不是「值」
- 这就是为什么 `[1, "a", [1,2]]` 能存混合类型：每个元素只是个指针
- 代价：每次访问要多一次指针解引用，缓存也不如纯 int 数组友好

numpy 的 `ndarray` 更接近 C 数组：直接存值，不存指针，所以快得多。

## 9. 与 numpy ndarray 的关系

numpy 的 `ndarray` 是多维数组，但底层也是一块连续内存：

```python
import numpy as np
a = np.array([[1, 2], [3, 4]], dtype=np.int32)
# 内存里就是 [1, 2, 3, 4]，连续 16 字节
```

- `dtype` 决定每个元素的大小（`int32` 是 4 字节）
- `shape` 决定怎么解释这块内存（`(2, 2)` 就是 2x2 矩阵）
- `strides` 决定每个维度的步长（行优先时 `strides = (8, 4)`，即跨行 8 字节、跨列 4 字节）

numpy 比 Python list 快的根本原因：
1. **存值不存指针**：缓存友好
2. **连续内存**：SIMD 可向量化
3. **C 实现**：无解释器开销
4. **dtype 固定**：无类型检查开销

## 10. 小结

动态数组的核心是：
1. **连续内存** → $O(1)$ 随机访问 + 缓存友好
2. **倍增扩容** → 摊还 $O(1)$ 追加
3. **size/capacity 分离** → 按需增长，不浪费

这三个特性让数组成为 AI 里最常用的数据结构——张量、矩阵、向量，底层全是数组。

下一篇 `ai_application.md` 讲：**数组的连续性怎么被矩阵乘法利用，分块矩阵乘法怎么进一步压榨缓存**。