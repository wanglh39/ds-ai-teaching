# 并查集（Disjoint Set）原理：父指针 + 路径压缩 + 按秩合并

> 本文档讲清「并查集是什么、父指针表示、find 和路径压缩、union 和按秩合并、为什么近 O(1)（Ackermann 反函数 α(n)）、C 实现逐行讲解、与 BFS/DFS 判连通对比」。AI 应用（层次聚类 HAC）见 `ai_application.md`。

## 1. 并查集是什么：维护"属于同一集合"的数据结构

**并查集**（Disjoint Set Union, DSU）是维护**若干不相交集合**的数据结构，支持两个核心操作：

- **find(x)**：返回元素 x 所在集合的代表元（根）
- **union(x, y)**：把 x 和 y 所在的两个集合合并成一个

并查集不关心集合里有什么元素，只关心**哪些元素属于同一集合**。这是很多算法的核心需求：

- **连通分量**：图中两个顶点是否连通？合并就是加边
- **Kruskal 最小生成树**：加边时判断两端是否已连通（避免成环）
- **等价类**：哪些变量相等？合并就是加等式约束
- **层次聚类**：哪些点属于同一簇？合并就是两个簇合体

```
初始: {0}, {1}, {2}, {3}, {4}     每个元素自成一集

union(0, 1): {0,1}, {2}, {3}, {4}
union(2, 3): {0,1}, {2,3}, {4}
union(0, 2): {0,1,2,3}, {4}       0 和 2 合并，连带 1 和 3

find(1) == find(3)?  →  是（都在 {0,1,2,3}）
find(1) == find(4)?  →  否
```

### 1.1 为什么不用别的数据结构

| 需求 | 哈希表 | 链表 | 并查集 |
|---|---|---|---|
| 查"x 和 y 同集？" | O(1)（存标签） | O(1)（存标签） | O(α) ≈ O(1) |
| 合并两个集合 | O(n)（改所有标签） | O(1)（接链表尾） | O(α) ≈ O(1) |
| 增量加边判连通 | O(n) 每次 | O(1) 但要存链表 | O(α) 且不存邻接表 |

哈希表查同集快，但合并要改一个集合所有元素的标签 O(n)。链表合并快（接尾），但要存链表 O(n) 内存。**并查集两个操作都近 O(1)**，且只存 parent 数组 O(n) 内存——这是它的核心优势。

## 2. 父指针表示：森林

并查集用**森林**（forest）表示集合：每棵树是一个集合，树根是代表元。每个元素存一个 `parent` 指针指向父节点，根的 parent 指向自己。

```
集合 {0,1,2,3} 和 {4,5}:

    树表示:          parent 数组:
       0              index: 0  1  2  3  4  5
      / \             parent:0  0  0  1  4  4
     1   2
     |
     3

    4
     \
      5

  find(x) = 沿 parent 走到根
  find(3) = 3 → 1 → 0  (根是 0)
  find(5) = 5 → 4      (根是 4)
```

**为什么用树？** 因为合并极快：union(x,y) 只需把一棵树的根挂到另一棵树的根下——改一个 parent 指针。

```
union(3, 5):  把 4 挂到 0 下
  before:        after:
     0              0
    / \            /|\
   1   2          1 2 4
   |              |   |
   3              3   5
   (根 0)         (根 0)
   4
    \
     5
   (根 4)
```

合并只需 `parent[4] = 0`，一个赋值。这是并查集比哈希表合并快的原因。

### 2.1 朴素实现的退化

如果每次 union 都把一棵树挂到另一棵树下，最坏情况会退化成**链表**：

```
union(0,1), union(1,2), union(2,3), ..., union(n-2, n-1):

  0 ← 1 ← 2 ← 3 ← ... ← n-1   (一条链)

  find(n-1) 要走 n 步 → O(n)
  n 次 find 总 O(n²)
```

树高 O(n)，find 退化成 O(n)。两个优化让树高降到近 O(1)：**路径压缩**和**按秩合并**。

## 3. find 和路径压缩

**路径压缩**（Path Compression）：find(x) 沿 parent 走到根后，把路径上所有节点**直接挂到根下**——下次 find 这些节点就是 O(1)。

```
find(3) 前的链:        find(3) 后:
  0                     0
  |                    /|\
  1                   1 2 3   ← 路径上所有节点直接挂到根
  |                   
  2                   
  |                   
  3                   

  下次 find(3) = O(1)（parent[3] 直接是根 0）
```

### 3.1 递归实现

```c
int dset_find(DSet *ds, int x) {
    if (ds->parent[x] != x) {
        ds->parent[x] = dset_find(ds, ds->parent[x]);  // 递归找根，回溯时压缩
    }
    return ds->parent[x];
}
```

递归版最简洁：`parent[x] = find(parent[x])` 一行同时做"找根"和"压缩"。递归到根后，回溯时把路径上每个节点的 parent 直接改成根。

### 3.2 迭代实现（两趟）

递归版在 n 很大时可能栈溢出（链长 > 栈深度）。迭代版用两趟：

```c
int dset_find(DSet *ds, int x) {
    // 第一趟：找根
    int root = x;
    while (ds->parent[root] != root) {
        root = ds->parent[root];
    }
    // 第二趟：路径压缩，把路径上所有节点直接挂到根
    while (ds->parent[x] != root) {
        ds->parent[x], x = root, ds->parent[x];  // 保存下一个，把当前挂到根，走到下一个
    }
    return root;
}
```

第一趟沿 parent 走到根，第二趟再走一遍把每个节点直接挂到根。本 C 实现用递归版（简洁），Python demo 用迭代版（避免栈溢出）。

### 3.3 路径压缩的效果

路径压缩后，树变得**非常扁平**——大部分节点直接挂在根下。经过一次 find，后续 find 几乎都是 O(1)。但路径压缩单独用还不够：最坏情况下第一次 find 仍可能 O(n)。需要配合按秩合并保证树高始终小。

## 4. union 和按秩合并

**按秩合并**（Union by Rank）：union 时把**矮树挂到高树下**，避免树高增长。rank 是树高的上界（路径压缩后不精确，但够用）。

```c
int dset_union_sets(DSet *ds, int x, int y) {
    int rx = dset_find(ds, x);
    int ry = dset_find(ds, y);
    if (rx == ry) return 0;                    // 已在同一集合，不合并
    if (ds->rank[rx] < ds->rank[ry]) {         // 保证 rx 是 rank 大的
        int tmp = rx; rx = ry; ry = tmp;
    }
    ds->parent[ry] = rx;                       // 矮树挂到高树下
    if (ds->rank[rx] == ds->rank[ry]) {        // 等高时高树 rank +1
        ds->rank[rx]++;
    }
    ds->n_sets--;
    return 1;
}
```

### 4.1 rank 的含义

rank 是**树高的上界**（不是精确高度，因为路径压缩会降高但不更新 rank）。关键性质：

- 矮树挂到高树下，树高不变
- 等高树合并，树高 +1
- rank 为 k 的树至少有 $2^k$ 个节点 → 树高最多 $\log_2 n$

```
rank 0:  单个节点        rank 1:  两节点合并     rank 2:  两个 rank 1 合并
    x                   x                    x
                        |                   / \
                        y                  y   z
                                           |   |
                                          ...  ...
```

### 4.2 按秩合并的效果

只按秩合并（不路径压缩），树高最多 $\log_2 n$，find 是 $O(\log n)$。n=10⁶ 时树高最多 20——已经很快。但配合路径压缩后，能做到近 O(1)。

### 4.3 两个优化必须一起用

| 优化 | 树高 | find | union |
|---|---|---|---|
| 都不用 | O(n) | O(n) | O(1) |
| 只路径压缩 | 摊还 O(α) | 摊还 O(α) | 摊还 O(α) |
| 只按秩合并 | O(log n) | O(log n) | O(log n) |
| **两者都用** | **近 O(1)** | **摊还 O(α)** | **摊还 O(α)** |

路径压缩单独用已经是摊还 O(α)，但按秩合并让常数更小、实际更快。两者一起用是标准做法。

## 5. 复杂度：Ackermann 反函数 α(n) ≈ 近 O(1)

两个优化一起用，find 和 union 的**摊还复杂度**是 $O(\alpha(n))$，其中 $\alpha(n)$ 是 **Ackermann 反函数**。

### 5.1 Ackermann 函数

Ackermann 函数 $A(m, n)$ 是一个增长**比任何原始递归函数都快**的函数：

```
A(0, n)   = n + 1
A(m+1, 0) = A(m, 1)
A(m+1, n+1) = A(m, A(m+1, n))

A(0, n) = n+1
A(1, n) = n+2
A(2, n) = 2n+3
A(3, n) = 2^(n+3) - 3
A(4, n) = 2^2^...^2 - 3   (n+3 个 2 的幂塔)
```

$A(4, 1)$ 已经是 $2^{65536} - 3$，远超宇宙原子数。

### 5.2 Ackermann 反函数

$\alpha(n)$ 是 $A$ 的反函数：$\alpha(n) = \min\{m : A(m, m) \geq n\}$。

因为 $A$ 增长极快，$\alpha$ 增长极慢：

| $n$ | $\alpha(n)$ |
|---|---|
| $n \leq 3$ | 0 |
| $n \leq 7$ | 1 |
| $n \leq 2^{65536}$ | 4 |

**对任何实际数据量**，$\alpha(n) \leq 4$。宇宙原子数 $\approx 10^{80} < 2^{65536}$，所以 $\alpha(10^{80}) = 4$。

### 5.3 实际含义

并查集的 find/union 是 $O(\alpha(n))$，对任何实际问题 $\alpha(n) \leq 4$——**常数时间**。所以并查集被称为"近 O(1)"。

n 次 union + m 次 find 的总成本是 $O((n+m) \cdot \alpha(n))$，实际就是 $O(n+m)$。这是并查集的强大之处：**几乎所有操作都是常数时间**。

### 5.4 为什么不是真的 O(1)

理论上 $\alpha(n)$ 不是常数——它会随 n 增长，只是增长极慢。所以严格说是 $O(\alpha(n))$ 不是 $O(1)$。但实际中 $\alpha(n) = 4$ 对任何可想象的 n 都成立，所以工程上当作 O(1) 完全合理。

## 6. C 实现逐行讲解

本实现文件 `10_disjoint_set/c/dset.h` + `dset.c`。

### 6.1 数据结构（dset.h）

```c
typedef struct {
    int *parent;    // 父指针数组，parent[x] 是 x 的父节点
    int *rank;      // 秩数组，rank[x] 是以 x 为根的树高上界
    size_t n;       // 元素总数
    size_t n_sets;  // 当前集合数（初始 n，每次 union 减 1）
} DSet;
```

只有两个数组 `parent` 和 `rank`，内存 $O(n)$。`n_sets` 维护集合数，union 时减 1，O(1) 查询。

### 6.2 初始化（dset_init）

```c
void dset_init(DSet *ds, size_t n) {
    ds->n = n;
    ds->n_sets = n;                          // 初始 n 个单元素集合
    ds->parent = (int *)malloc(n * sizeof(int));
    ds->rank = (int *)calloc(n, sizeof(int));  // rank 初始全 0
    for (size_t i = 0; i < n; i++) {
        ds->parent[i] = (int)i;              // 每个元素自成一集，parent 指向自己
    }
}
```

初始每个元素是一个单节点树，parent[i] = i（根指向自己），rank 全 0。

### 6.3 find 带路径压缩（dset_find）

```c
int dset_find(DSet *ds, int x) {
    if (ds->parent[x] != x) {
        ds->parent[x] = dset_find(ds, ds->parent[x]);  // 递归找根，回溯时压缩
    }
    return ds->parent[x];
}
```

`parent[x] != x` 说明 x 不是根，递归找根。回溯时 `parent[x] = root` 把 x 直接挂到根下。一行同时做"找根"和"路径压缩"。

### 6.4 union 按秩合并（dset_union_sets）

```c
int dset_union_sets(DSet *ds, int x, int y) {
    int rx = dset_find(ds, x);               // 找 x 的根
    int ry = dset_find(ds, y);               // 找 y 的根
    if (rx == ry) return 0;                  // 已同集，不合并，返回 0
    if (ds->rank[rx] < ds->rank[ry]) {       // 保证 rx 是 rank 大的
        int tmp = rx; rx = ry; ry = tmp;
    }
    ds->parent[ry] = rx;                     // 矮树（ry）挂到高树（rx）下
    if (ds->rank[rx] == ds->rank[ry]) {      // 等高时高树 rank +1
        ds->rank[rx]++;
    }
    ds->n_sets--;                            // 集合数减 1
    return 1;                                // 合并成功，返回 1
}
```

关键三步：
1. find 两边根（同时路径压缩）
2. 矮树挂高树下（保证树高不增）
3. 等高时 rank+1（树高增 1）

返回值：1 表示合并了，0 表示已同集。`n_sets` 维护集合数。

### 6.5 connected 和 count_sets

```c
int dset_connected(DSet *ds, int x, int y) {
    return dset_find(ds, x) == dset_find(ds, y);  // 同根即连通
}

size_t dset_count_sets(DSet *ds) {
    return ds->n_sets;                       // O(1) 查询，不用数根
}
```

`connected` 两次 find 比根，$O(\alpha)$。`count_sets` 直接返回维护的 `n_sets`，O(1)。

## 7. 与 BFS/DFS 判连通对比

判"两个顶点是否连通"有两种做法：

| 方法 | 数据结构 | 判连通 | 加边 | 内存 | 增量加边 |
|---|---|---|---|---|---|
| **BFS/DFS** | 邻接表 | O(V+E) | O(1) | O(V+E) | O(V+E) 重建 |
| **并查集** | parent[] | O(α) | O(α) | O(V) | O(α) |

### 7.1 一次性判连通

给定一个图，判"u 和 v 是否连通"：

- **BFS/DFS**：从 u 出发遍历整个连通分量，看能否到 v。O(V+E) 时间，O(V+E) 内存（存邻接表）
- **并查集**：对所有边 union 一遍，然后 find(u) == find(v)。O(E·α) 时间，O(V) 内存

并查集时间多一个 α 因子（但 α≈4），内存省 O(E)（不存邻接表）。

### 7.2 增量加边判连通

**核心差异**：如果图在不断加边，每次加边后要判连通：

- **BFS/DFS**：每次加边后，连通分量可能变，要重新 BFS O(V+E)。加 k 条边总 O(k·(V+E))
- **并查集**：每次加边就 union O(α)，判连通就 find O(α)。加 k 条边总 O(k·α)

**这是层次聚类的核心场景**：逐步降低距离阈值，每次新边出现，两个簇合并。并查集每次 O(α)，BFS 每次重建 O(V+E)。

### 7.3 具体数字

n=10000 点，E≈2000000 边，10 次阈值更新：

| 方法 | 10 次更新总成本 | 实测 |
|---|---|---|
| 并查集 | 10 × ΔE × α ≈ 10 × 200000 × 4 = 8M | ~6.5 秒 |
| BFS 重建 | 10 × (V+E) = 10 × 2010000 = 20M | ~24 秒 |

实测并查集快 3.7x（纯 Python 开销拉低了理论优势，C 实现差距更大）。内存上并查集 O(V) = 80KB，BFS O(V+E) = 30MB，省 400x。

### 7.4 什么时候用 BFS/DFS

并查集只能判"是否连通"，**不能**：

- 找最短路径（BFS 的本职工作）
- 遍历连通分量里的所有节点（BFS/DFS 可以，并查集要扫全图数根）
- 处理带权图的最短路（Dijkstra/Bellman-Ford）

所以并查集用于**只需判连通/维护连通分量**的场景。需要路径信息时用 BFS/DFS。

## 8. 复杂度汇总

| 操作 | 时间 | 空间 |
|---|---|---|
| init(n) | O(n) | O(n) |
| find(x) | O(α(n)) 摊还 | O(1) |
| union(x,y) | O(α(n)) 摊还 | O(1) |
| connected(x,y) | O(α(n)) 摊还 | O(1) |
| count_sets() | O(1) | O(1) |
| n 次 union + m 次 find | O((n+m)·α(n)) | O(n) |

$\alpha(n) \leq 4$ 对任何实际 n，所以工程上所有操作都是 O(1)。这是并查集被称为"最接近魔法的数据结构"的原因。

## 9. 总结

并查集的三个关键思想：

1. **父指针表示**：用森林表示集合，合并只需改一个指针
2. **路径压缩**：find 时把路径压平，后续 find 近 O(1)
3. **按秩合并**：矮树挂高树下，树高最多 O(log n)

三者配合，得到 $O(\alpha(n))$ 摊还复杂度——对任何实际数据量都是常数时间。内存只 O(V)（不存邻接表），增量加边 O(α)（不重建）。这让并查集成为**层次聚类、Kruskal MST、连通分量维护**的首选数据结构。