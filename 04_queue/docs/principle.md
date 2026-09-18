# 循环队列原理

> 本文档讲清「循环队列是什么、为什么要循环、怎么判满判空、C 实现细节」。AI 应用见 `ai_application.md`。

## 1. 队列的定义

队列是先进先出（FIFO, First In First Out）的线性表：

```
入队: enqueue(1), enqueue(2), enqueue(3)
队列: [1, 2, 3]  ← 队头(出)    队尾(入) →
出队: dequeue() → 1, dequeue() → 2, dequeue() → 3
```

两个核心操作：
- `enqueue(v)`：在队尾插入，$O(1)$
- `dequeue()`：从队头删除，$O(1)$

**FIFO 是队列的本质**：最先放进去的最先出来。这与栈的 LIFO 恰好相反。

| 场景 | 为什么 FIFO |
|---|---|
| 排队取号 | 先来的先服务 |
| 打印队列 | 先提交的先打印 |
| BFS 广度优先搜索 | 先发现的节点先扩展 |
| 操作系统任务调度 | 先到的任务先调度（FCFS） |
| **分布式通信** | **梯度按顺序聚合，环形传递** |

## 2. 数组队列的假溢出

最朴素的数组队列：用一个数组，`head` 指向队头，`tail` 指向队尾下一个空位。

```c
enqueue: data[tail++] = v;
dequeue: v = data[head++];
```

问题：`head` 和 `tail` 只增不减，前面出队腾出的空间无法复用。

```
容量 5 的数组队列：
[1, 2, 3, 4, 5]   head=0, tail=5
dequeue 3 次 → head=3, tail=5
[_, _, _, 4, 5]   剩余 2 个元素，但 tail 已到末尾
enqueue(6) → tail=6 越界！明明前面有 3 个空位却用不了
```

这就是**假溢出**：数组前端有空位，但 `tail` 指针已经走到数组末尾，无法继续入队。

### 2.1 解决方案一：搬移

每次出队后把所有元素前移，保持 `head=0`：

```
dequeue → 所有元素前移一位，tail--
```

- 代价：每次 `dequeue` 变成 $O(n)$（要搬移 $n-1$ 个元素）
- 不可接受：队列的核心优势就是 $O(1)$ 入队出队

### 2.2 解决方案二：循环队列（回绕）

让 `head` 和 `tail` 在到数组末尾时**回绕到开头**，把数组看成一个环：

```
容量 5 的循环队列：
[1, 2, 3, 4, 5]   head=0, tail=0（判满见下文）
dequeue 3 次 → head=3
[_, _, _, 4, 5]
enqueue(6) → data[0]=6, tail=(5+1)%5=0? 不对...
```

关键：用**取模运算**实现回绕：

```c
tail = (tail + 1) % capacity;  // 到末尾后回到 0
head = (head + 1) % capacity;
```

这样数组前端的空间就能被复用，入队出队都是 $O(1)$。

## 3. 循环队列的定义

循环队列 = 固定大小数组 + 头尾指针 + 取模回绕。

```
逻辑视图（环）:
        data[0]
       /        \
   data[4]      data[1]
       \        /
       data[3] - data[2]

物理视图（数组）:
[data[0], data[1], data[2], data[3], data[4]]
  ↑ tail                ↑ head
  （回绕后 tail 可在 head 左边）
```

- `head`：队头下标，`dequeue` 从这里取
- `tail`：队尾下一个空位下标，`enqueue` 写到这里
- `capacity`：数组物理大小
- 入队：`data[tail] = v; tail = (tail+1) % capacity;`
- 出队：`v = data[head]; head = (head+1) % capacity;`

## 4. 判满判空：两种方案

回绕带来一个麻烦：**队空和队满时，`head == tail` 都成立**。

```
队空: head == tail（没有元素）
队满: head == tail（元素填满一圈，tail 追上了 head）
```

光看 `head == tail` 无法区分空和满。两种解决方案：

### 4.1 方案一：浪费一个槽位（本章实现）

让 `tail` 永远指向「下一个空位」，队满时 `tail` 停在 `head` 前一格，即：

```c
判满: (tail + 1) % capacity == head
判空: head == tail
```

- 容量 `capacity` 的数组最多存 `capacity - 1` 个元素
- 代价：浪费 1 个槽位
- 好处：无需额外字段，$O(1)$ 判断

```
capacity=5 的数组，最多存 4 个：
[_, 2, 3, 4, 5]   head=1, tail=0
判满: (0+1)%5 == 1 == head → 满
判空: head=1 != tail=0 → 非空
```

### 4.2 方案二：额外 size 字段

维护一个 `size` 字段记录当前元素数：

```c
判满: size == capacity
判空: size == 0
```

- 不浪费槽位，可存 `capacity` 个
- 代价：每次入队出队都要更新 `size`（一次整数加减，几乎无开销）
- 好处：`size` 查询也是 $O(1)$，且语义清晰

### 4.3 两种方案对比

| 维度 | 浪费一个槽位 | 额外 size 字段 |
|---|---|---|
| 实际容量 | capacity - 1 | capacity |
| 额外空间 | 1 个槽位 | 1 个 size_t（8 字节） |
| 判满判空 | 比较 head/tail | 比较 size |
| size 查询 | `(tail-head+cap)%cap` | 直接返回 size |
| 实现复杂度 | 略高（取模算 size） | 略低 |

本章采用**方案一**（浪费一个槽位），因为它更经典，且能展示取模回绕的核心思想。生产代码里两种都常见。

## 5. C 实现逐行讲解

### 5.1 结构体（queue.h）

```c
typedef struct {
    double *data;    // 固定大小数组
    size_t head;     // 队头下标
    size_t tail;     // 队尾下一个空位下标
    size_t capacity; // 数组物理大小（含浪费的槽位）
} DQueue;
```

- `data` 是动态分配的数组（`init` 时分配，`free` 时释放）
- `capacity` 是数组物理大小，用户期望存 `n` 个元素时实际分配 `n+1`
- `head`/`tail` 是环形下标，取模回绕

### 5.2 初始化（queue.c）

```c
void dsq_init(DQueue *q, size_t capacity) {
    if (capacity == 0) capacity = 1;        // 防御：容量至少 1
    q->capacity = capacity + 1;             // 多分配 1 个槽位（浪费法）
    q->data = (double *)malloc(q->capacity * sizeof(double));
    q->head = 0;
    q->tail = 0;
}
```

- 用户传 `capacity=5`，实际 `q->capacity=6`，可存 5 个元素
- `head == tail == 0` 表示空队
- 若 `malloc` 失败返回 `NULL`，后续操作会段错误；生产代码应检查（这里为简洁省略）

### 5.3 入队（enqueue）

```c
bool dsq_enqueue(DQueue *q, double v) {
    if ((q->tail + 1) % q->capacity == q->head) return false;  // 满
    q->data[q->tail] = v;                                       // 写入队尾
    q->tail = (q->tail + 1) % q->capacity;                      // tail 回绕前进
    return true;
}
```

- 先判满：`(tail+1) % capacity == head` 表示再写一个就追上 head
- 写入 `data[tail]`，然后 `tail` 取模前进
- 返回 `false` 表示队满，调用方可选择丢弃或等待

### 5.4 出队（dequeue）

```c
bool dsq_dequeue(DQueue *q, double *out) {
    if (q->head == q->tail) return false;                       // 空
    if (out) *out = q->data[q->head];                           // 读出队头
    q->head = (q->head + 1) % q->capacity;                      // head 回绕前进
    return true;
}
```

- 先判空：`head == tail`
- 读出 `data[head]`（若 `out` 非 NULL），然后 `head` 取模前进
- 返回 `false` 表示队空

### 5.5 查询操作

```c
bool dsq_empty(const DQueue *q) {
    return q->head == q->tail;
}

size_t dsq_size(const DQueue *q) {
    return (q->tail - q->head + q->capacity) % q->capacity;
}

size_t dsq_capacity(const DQueue *q) {
    return q->capacity > 0 ? q->capacity - 1 : 0;  // 扣除浪费的槽位
}
```

- `size` 的计算：`(tail - head + capacity) % capacity`
  - `tail >= head` 时：`tail - head`
  - `tail < head` 时（回绕过）：`tail - head + capacity`
  - 加 `capacity` 再取模统一了两种情况
- `capacity` 对外报告的是「可存元素数」，即物理大小减 1

### 5.6 释放

```c
void dsq_free(DQueue *q) {
    free(q->data);
    q->data = NULL;
    q->head = 0;
    q->tail = 0;
    q->capacity = 0;
}
```

- 释放数组，指针置 NULL 防悬空
- 所有字段归零，结构体回到安全状态

## 6. 回绕的正确性：一次完整追踪

以 `capacity=4`（用户期望存 4 个，物理 `q->capacity=5`）为例：

```
操作              data[]          head  tail  size  判满判空
init(4)           [_,_,_,_,_]      0     0     0    空(head==tail)
enqueue(1)        [1,_,_,_,_]      0     1     1
enqueue(2)        [1,2,_,_,_]      0     2     2
enqueue(3)        [1,2,3,_,_]      0     3     3
enqueue(4)        [1,2,3,4,_]      0     4     4    (4+1)%5=0==head → 满
enqueue(5) → 失败（满）
dequeue → 1       [_,2,3,4,_]      1     4     3
enqueue(5)        [_,2,3,4,5]      1     0     4    tail 回绕到 0，(0+1)%5=1==head → 满
dequeue → 2       [_,_,3,4,5]      2     0     3
dequeue → 3       [_,_,_,4,5]      3     0     2
enqueue(6)        [6,_,_,4,5]      3     1     3    tail 回绕，写到 data[0]
dequeue → 4       [6,_,_,_,5]      4     1     2
dequeue → 5       [6,_,_,_,_]      0     1     1    head 回绕到 0
dequeue → 6       [_,_,_,_,_]      1     1     0    空(head==tail)
```

关键观察：
1. `head` 和 `tail` 都能在数组里「转圈」
2. 元素 5、6 都写到了数组前端（复用了出队腾出的空间）
3. 出队顺序 1,2,3,4,5,6 严格 FIFO
4. 满时恰好存了 4 个（= 用户期望容量），空时 `head==tail`

## 7. 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|---|---|---|
| `init` | $O(n)$（分配并零初始化可选） | $O(n)$ |
| `enqueue` | $O(1)$ | $O(1)$ |
| `dequeue` | $O(1)$ | $O(1)$ |
| `peek` | $O(1)$ | $O(1)$ |
| `empty` / `size` | $O(1)$ | $O(1)$ |
| `free` | $O(1)$ | — |

- 所有操作都是 $O(1)$，这是循环队列的核心优势
- 空间是 $O(n)$，$n$ 是容量，固定不变
- 取模运算 `% capacity` 在 `capacity` 是 2 的幂时可优化为位运算 `& (capacity-1)`（很多生产实现要求容量是 2 的幂）

## 8. 与链表队列的对比

### 8.1 链表队列

用带头尾指针的单链表：

```c
typedef struct Node {
    double val;
    struct Node *next;
} Node;

typedef struct {
    Node *head;  // 队头
    Node *tail;  // 队尾
} LQueue;
```

- `enqueue`：在 `tail` 后插入节点，`tail` 后移，$O(1)$
- `dequeue`：取 `head` 节点，`head` 后移，释放节点，$O(1)$
- 容量无上限（受内存限制）

### 8.2 对比表

| 维度 | 循环队列（数组） | 链表队列 |
|---|---|---|
| 入队出队 | $O(1)$ | $O(1)$ |
| 容量 | 固定（预分配） | 动态（按需分配） |
| 内存 | 连续，缓存友好 | 每节点一次 malloc，缓存差 |
| 内存开销 | 仅数据本身 | 每节点多一个指针（8 字节） |
| 扩容 | 不支持（固定大小） | 天然支持 |
| 假溢出 | 回绕解决 | 无此问题 |
| 判满 | 有容量限制 | 无（除非内存耗尽） |
| 适用场景 | 容量已知、高频操作 | 容量未知、稀疏 |

### 8.3 选择建议

- **容量已知 + 高频**：循环队列（缓存友好、无 malloc 开销）
  - 例：批处理调度（一批 N 个任务）、环形缓冲区（音频/视频流）
- **容量未知 + 稀疏**：链表队列
  - 例：BFS 的待访问队列（规模不确定）
- **既要容量灵活又要缓存友好**：动态数组队列（扩容时搬移，均摊 $O(1)$）

## 9. 常见陷阱

### 9.1 判满判空混淆

用 `head == tail` 判满会误判空队为满。必须区分两种方案，本章用「浪费一个槽位」法：

```c
判满: (tail + 1) % capacity == head   // 不是 head == tail
判空: head == tail
```

### 9.2 size 计算的负数问题

```c
size = (tail - head + capacity) % capacity;
```

若直接写 `(tail - head) % capacity`，当 `tail < head` 时（回绕过），`tail - head` 是负数（无符号下是大正数），取模结果错误。加 `capacity` 再取模保证非负。

### 9.3 容量为 0 或 1

- `capacity=0`：取模 `% 0` 是未定义行为，`init` 里做了防御（至少 1）
- `capacity=1`：浪费一个槽位后实际存 0 个，是无用队列；`init` 里 `capacity+1=2`，可存 1 个

### 9.4 取模性能

`%` 运算比加减法慢。优化：让 `capacity` 是 2 的幂，用 `& (capacity-1)` 替代 `% capacity`。很多生产实现（如 Linux 内核的 `kfifo`）要求容量是 2 的幂。

## 10. 小结

循环队列的核心思想：

1. **固定数组 + 头尾指针**：空间连续，缓存友好
2. **取模回绕**：解决假溢出，复用前端空间，入队出队 $O(1)$
3. **浪费一个槽位**：区分空和满，无需额外字段
4. **FIFO 语义**：先进先出，匹配排队、调度、环形通信等场景

下一章的 AI 应用将展示：循环队列的「环形」结构如何天然对应到分布式训练的 Ring-AllReduce 通信拓扑，实现带宽最优的梯度同步。