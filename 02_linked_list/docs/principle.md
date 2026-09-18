# 链表原理

> 本文档讲清「链表是什么、为什么需要它、怎么实现、复杂度多少」。AI 应用见 `ai_application.md`。

## 1. 链表的定义

链表是一种线性表，但**元素在内存里不连续**，每个节点通过指针指向下一个节点：

```
head → [v1|next] → [v2|next] → [v3|next] → NULL
```

每个节点存两样东西：
- `value`：数据
- `next`：指向下一个节点的指针

这是单链表。双向链表再加一个 `prev` 指针：

```
NULL ← [prev|v1|next] ⇄ [prev|v2|next] ⇄ [prev|v3|next] → NULL
```

## 2. 为什么需要链表：数组的短板

数组（第 01 章）在**头部插入**和**中间插入**是 $O(n)$，因为要保持连续存放，必须移动后面所有元素：

```
插入前: [a, b, c, d, _]
插入 x 到位置 1: [a, x, b, c, d]  ← b, c, d 都要右移
```

如果频繁在头部/中间插入删除，数组的 $O(n)$ 移累成 $O(n^2)$，不可接受。

链表的优势正在此：**插入删除只需改指针，$O(1)$**：

```
插入前: ... → [a] → [c] → ...
插入 x: ... → [a] → [x] → [c] → ...  ← 只改 a.next 和 x.next
```

## 3. 链表的代价：随机访问 O(n)

链表的代价是**随机访问变慢**：要找第 $i$ 个元素，必须从头顺着指针走 $i$ 步，$O(n)$。

| 操作 | 数组 | 链表 |
|---|---|---|
| `A[i]` 随机访问 | $O(1)$ | $O(n)$ |
| 头部插入 | $O(n)$ | $O(1)$ |
| 尾部插入 | $O(1)$ 摊还 | $O(1)$（有尾指针） |
| 已知节点后插入 | $O(n)$ | $O(1)$ |
| 已知节点删除 | $O(n)$ | $O(1)$（双向链表） |
| 遍历 | 缓存友好 | 缓存不友好 |

**核心权衡**：数组用连续内存换 $O(1)$ 访问；链表用指针换 $O(1)$ 插入删除。

## 4. 单链表 vs 双向链表

### 4.1 单链表

只有 `next` 指针：
- 插入：$O(1)$（已知前驱节点）
- 删除：$O(n)$（要找前驱，因为要改前驱的 `next`）
- 空间：每个节点 1 个指针

### 4.2 双向链表

有 `prev` 和 `next`：
- 插入：$O(1)$
- 删除：$O(1)$（已知节点，直接改 `prev.next` 和 `next.prev`）
- 空间：每个节点 2 个指针

**LRU 缓存用双向链表**，因为淘汰时要从尾部删节点（$O(1)$），访问后要把节点移到头部（$O(1)$，需要 `prev` 才能脱链）。

## 5. C 实现逐行讲解

### 5.1 节点定义

```c
typedef struct DListNode {
    int key;
    int value;
    struct DListNode *prev;
    struct DListNode *next;
} DListNode;
```

- `key`：用于 LRU 的键
- `value`：存储的值
- `prev` / `next`：双向指针

### 5.2 链表结构

```c
typedef struct {
    DListNode *head;
    DListNode *tail;
    size_t size;
} DList;
```

- `head`：头指针（最近访问）
- `tail`：尾指针（最久未访问，LRU 淘汰对象）
- `size`：节点数

### 5.3 头部插入

```c
DListNode *dl_push_front(DList *l, int key, int value) {
    DListNode *n = make_node(key, value);
    if (!n) return NULL;
    if (!l->head) {
        l->head = l->tail = n;
    } else {
        n->next = l->head;
        l->head->prev = n;
        l->head = n;
    }
    l->size++;
    return n;
}
```

- 创建新节点
- 空链表：head 和 tail 都指向新节点
- 非空：新节点的 next 指向旧 head，旧 head 的 prev 指向新节点，head 更新
- 返回节点指针（LRU 需要把它存进哈希表）

### 5.4 脱链（核心辅助操作）

```c
void dl_unlink(DList *l, DListNode *node) {
    if (node->prev) node->prev->next = node->next;
    else l->head = node->next;
    if (node->next) node->next->prev = node->prev;
    else l->tail = node->prev;
    node->prev = NULL;
    node->next = NULL;
    l->size--;
}
```

把节点从链表里摘出来，但不释放内存：
- 如果 node 有前驱，前驱的 next 跳过 node
- 否则 node 是 head，head 后移
- 同理处理 next 和 tail
- 清空 node 的指针，size 减 1

**这是 LRU 的核心**：访问一个节点时，先 unlink 再 push_front，就把它移到了头部。

### 5.5 移到头部

```c
void dl_move_to_front(DList *l, DListNode *node) {
    if (node == l->head) return;
    dl_unlink(l, node);
    node->next = l->head;
    if (l->head) l->head->prev = node;
    l->head = node;
    if (!l->tail) l->tail = node;
    l->size++;
}
```

- 已经是 head 就不用动
- 否则先脱链，再插到头部
- 全程 $O(1)$，只改几个指针

## 6. 链表的缓存不友好

链表最大的弱点是**遍历时缓存不友好**：

- 数组：元素连续，访问 `A[i]` 时 `A[i+1]` 已在缓存
- 链表：每个节点的下一个节点在哪完全不确定，几乎每次都 miss

实测：遍历同样多元素，链表比数组慢 5~10 倍（即使操作数相同）。

**这就是 AI 里几乎不用链表存数据的原因**——AI 要大规模遍历（矩阵、张量），链表的缓存不友好是致命的。

链表在 AI 里只用于**少量、频繁插入删除**的场景，最典型的就是 LRU 缓存（第 02 章）。

## 7. 与其他数据结构的关系

- **数组**（第 01 章）：连续内存，$O(1)$ 访问，但插入删除 $O(n)$
- **链表**（本章）：指针连接，$O(1)$ 插入删除，但访问 $O(n)$
- **哈希表**（第 05 章）：通常用链表做桶（拉链法）
- **跳表**（第 12 章）：多层链表，给链表加索引让它 $O(\log n)$ 访问

## 8. 小结

链表的核心：
1. **指针连接** → $O(1)$ 插入删除
2. **不连续** → $O(n)$ 随机访问 + 缓存不友好
3. **双向链表** → 已知节点 $O(1)$ 删除（LRU 的关键）

链表不是 AI 的主力数据结构（数组才是），但在 LRU 缓存等「频繁插入删除 + 顺序维护」场景不可替代。

下一篇 `ai_application.md` 讲：**LRU 缓存怎么用哈希表 + 双向链表实现 $O(1)$ 淘汰，KV Cache 为什么需要它**。