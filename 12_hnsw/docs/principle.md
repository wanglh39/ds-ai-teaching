# HNSW 原理：跳表 + 近邻图 = 亚线性向量检索

> 本文档讲清「为什么跳表和近邻图单独都不够，组合起来才是 HNSW」的数据结构原理：跳表的多层索引链表、抛硬币定层、O(log n) 搜索；近邻图的贪心搜索、入口点问题；HNSW 如何用跳表的思想给图提供分层入口点。C 实现逐行讲解。AI 应用（向量检索）见 `ai_application.md`。

## 1. 问题：高维向量检索为什么难

给定 n 个 d 维向量和一个查询向量 q，找离 q 最近的 k 个向量。这是 **k-近邻搜索（k-NN）**。

**暴力做法**：算 q 到所有 n 个向量的距离，排序取前 k。复杂度 O(n·d)——数据量一大就崩。100 万向量、50 维，一次查询要算 5000 万次浮点运算，约 20ms。看起来不多，但 RAG 系统一次请求要查几十次，延迟就上去了。

**空间数据结构**（KD-tree、Ball-tree）在低维（d ≤ 10）很快，但高维（d ≥ 20）退化到 O(n)——「维度灾难」。向量检索的维度通常是 50~1536（BERT 是 768，OpenAI text-embedding-3 是 1536），KD-tree 完全没用。

**HNSW**（Hierarchical Navigable Small World）的答案：不切空间，建一张**近邻图**，在图上贪心搜索。配合**跳表的分层思想**选入口点，实现 O(log n) 的亚线性检索。这是 2018 年 Malkov 提出的算法，现在 FAISS、Milvus、hnswlib 都用它。

本文档讲清两个组件——跳表和近邻图——以及它们为什么单独都不够、组合起来才够。

## 2. 跳表：多层索引链表

### 2.1 从有序链表说起

有序链表插入/删除/搜索都是 O(n)——必须从头遍历找位置。即使数据有序，也用不上二分。

**跳表的想法**：在链表之上加几层「索引链表」。第 0 层是完整链表，第 1 层每隔一个节点留一个，第 2 层更稀疏……搜索时从最高层开始，快速跳过大段数据，再逐层下降精确定位。

```
第 3 层:  1 ──────────────────────────────→ 9
第 2 层:  1 ──────────→ 5 ──────────→ 9
第 1 层:  1 ────→ 3 ────→ 5 ────→ 7 ────→ 9
第 0 层:  1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
```

搜索 6：从第 3 层 1 出发，next 是 9 > 6，下降到第 2 层；next 是 5 < 6，走到 5，next 是 9 > 6，下降到第 1 层；next 是 7 > 6，下降到第 0 层；next 是 6，找到。总共走了 4 步，而非从头遍历 6 步。

### 2.2 抛硬币定层

每个节点该出现在哪些层？**抛硬币**：插入时，从第 0 层开始，每次以概率 p（通常 0.5）决定是否升一层。抛到反面就停。

```
插入节点 6:
  第 0 层：一定有
  第 1 层：抛硬币 → 正面，有
  第 2 层：抛硬币 → 反面，停
  → 6 出现在第 0、1 层
```

期望层数 = 1/(1-p) = 2（p=0.5 时）。第 k 层的节点数期望 = n·p^k。最高层期望 = log_{1/p}(n)。

**关键性质**：层数服从几何分布，期望最高层 O(log n)。搜索时每层期望走 O(1/p) 步，总共 O(log n / p) = O(log n)。

### 2.3 跳表搜索

```
search(key):
  x = head
  for i = top_level down to 0:
    while x.next[i] != null and x.next[i].key < key:
      x = x.next[i]       # 在第 i 层向右走
  x = x.next[0]            # 下降到第 0 层的候选
  return x != null and x.key == key
```

从最高层开始，每层向右走到最后一个 < key 的节点，然后下降。第 0 层的 next 就是目标位置。

**复杂度**：每层期望走 1/p 步（p=0.5 时 2 步），共 O(log n) 层，总共 O(log n)。

### 2.4 跳表 vs 平衡树

| 特性 | 跳表 | 红黑树 / AVL |
|---|---|---|
| 期望复杂度 | O(log n) | O(log n) |
| 实现难度 | 简单（抛硬币） | 复杂（旋转） |
| 并发友好 | 好（局部锁） | 差（旋转动多处） |
| 范围查询 | 天然支持（第 0 层链表） | 需要中序遍历 |
| 内存 | O(n) + 索引指针 | O(n) |
| 最坏情况 | O(n)（概率极低） | O(log n) |

跳表用概率换简单性。实践中跳表常用于并发结构（Redis 的 ZSET、LevelDB 的 MemTable）。

### 2.5 C 实现讲解

```c
// skiplist.h
#define SKIPLIST_MAX_LEVEL 16

typedef struct SkipNode {
    int key;
    int value;
    int level;
    struct SkipNode *next[];   // 柔性数组，next[0..level-1]
} SkipNode;
```

`next[]` 是**柔性数组**——每个节点的层数不同，按实际层数分配内存。`sizeof(SkipNode) + sizeof(SkipNode*) * level`。

```c
// 抛硬币定层
int skiplist_random_level(void) {
    int level = 1;
    while ((rand() & 1) && level < SKIPLIST_MAX_LEVEL) {
        level++;
    }
    return level;
}
```

`rand() & 1` 取随机数的最低位，等概率 0/1。连续抛到 1 就升层，抛到 0 就停。最高 16 层（够 2^16 = 65536 个节点，实际 log n 远小于此）。

```c
// 插入
void skiplist_insert(SkipList *sl, int key, int value) {
    SkipNode *update[SKIPLIST_MAX_LEVEL];   // 记录每层要插入位置的前驱
    SkipNode *x = sl->head;
    for (int i = sl->level - 1; i >= 0; i--) {
        while (x->next[i] && x->next[i]->key < key)
            x = x->next[i];
        update[i] = x;                      // 第 i 层的前驱
    }
    int lvl = skiplist_random_level();      // 抛硬币定层
    if (lvl > sl->level) {
        for (int i = sl->level; i < lvl; i++)
            update[i] = sl->head;           // 新层的前驱是 head
        sl->level = lvl;
    }
    SkipNode *node = skipnode_alloc(lvl, key, value);
    for (int i = 0; i < lvl; i++) {
        node->next[i] = update[i]->next[i]; // 标准链表插入
        update[i]->next[i] = node;
    }
    sl->count++;
}
```

`update[]` 记录每层的插入前驱——这是跳表插入的关键。搜索时顺便记录每层停在哪，插入时直接用。

```c
// find_closest：找 key 最接近的节点（HNSW 入口点选择用）
bool skiplist_find_closest(const SkipList *sl, int key, int *out_key, int *out_value) {
    int le_key, le_val;
    bool has_le = skiplist_find_le(sl, key, &le_key, &le_val);   // ≤ key 的最大
    int ge_key, ge_val;
    bool has_ge = skiplist_find_ge(sl, key, &ge_key, &ge_val);   // ≥ key 的最小
    if (!has_le && !has_ge) return false;
    if (!has_le) { *out_key = ge_key; *out_value = ge_val; return true; }
    if (!has_ge) { *out_key = le_key; *out_value = le_val; return true; }
    if (key - le_key <= ge_key - key) {   // 选更近的
        *out_key = le_key; *out_value = le_val;
    } else {
        *out_key = ge_key; *out_value = ge_val;
    }
    return true;
}
```

`find_closest` 是 HNSW 入口点选择的核心：给定查询的「投影值」，O(log n) 找到投影最接近的节点作为图搜索的起点。

## 3. 近邻图：每节点连最近邻

### 3.1 图结构

**近邻图**：每个向量是一个节点，每个节点连到它的 M 个最近邻。搜索时从某个入口点出发，贪心走向查询点。

```
向量分布:                    近邻图 (M=3):
  A    B                        A ── B
   \  / \                       |    | \
    C    D                      C    D──E
     \  /                        \  /
      E                           E
```

### 3.2 贪心搜索

```
greedy_search(query, entry):
  current = entry
  while True:
    best_nb = None
    best_dist = dist(current, query)
    for nb in neighbors(current):
      if dist(nb, query) < best_dist:
        best_dist = dist(nb, query)
        best_nb = nb
    if best_nb == None: break    # 局部最优，停
    current = best_nb            # 走到更近的邻居
  return current
```

每次看当前节点的所有邻居，走到离 query 最近的那个。重复直到没有邻居比当前更近——到达**局部最优**。

**复杂度**：每步看 M 个邻居，算 M 次距离。走多少步取决于图的结构——如果图是「可导航的」（Small World 性质），期望步数 O(log n)。总共 O(M·log n·d)。

### 3.3 best-first search（HNSW 实际用的）

纯贪心只走一条路径，容易陷局部最优。HNSW 用 **best-first search**：维护一个大小 ef 的候选集，每次扩展最近的候选，而非只走最近的一个。

```
best_first_search(query, ef):
  candidates = priority_queue(entry)    # 按 dist 排序
  results = priority_queue(entry)       # 保留 ef 个最近
  while candidates not empty:
    c = candidates.pop_nearest()
    if dist(c) > results.farthest(): break   # 候选都比结果远，停
    for nb in neighbors(c):
      if nb not visited:
        visited.add(nb)
        if dist(nb) < results.farthest() or len(results) < ef:
          candidates.push(nb)
          results.push(nb)
          if len(results) > ef: results.pop_farthest()
  return results
```

ef 越大，搜索越宽，召回率越高，但越慢。ef=1 退化为纯贪心。

### 3.4 图的入口点问题

贪心搜索需要一个**入口点**。如果入口点离 query 很远，搜索可能走很多步，甚至走不到 query 附近（陷在远处的局部最优）。

**朴素方案**：固定一个入口点（比如第一个插入的节点）。问题：如果 query 在图的另一端，搜索要走很远。

**随机方案**：随机选入口点。问题：质量不稳定。

**HNSW 的方案**：用跳表的分层思想——上层图是下层的抽样，搜索从顶层（稀疏）开始，能跨大距离；逐层下降到第 0 层（密集）精确定位。

## 4. 为什么单独都不够

### 4.1 纯跳表在高维不够

跳表是**一维**结构——按 key 排序。高维向量没有天然的「排序顺序」。

可以把高维向量投影到一维（取第一维坐标、或到原点距离），用跳表排序。但**不同向量可能投影到同一点**（投影冲突），且**投影近不等于原向量近**。

```
二维向量:                    投影到 x 轴:
  A(1, 10)                   A: 1
  B(2, 0)                    B: 2     ← A、B 投影接近，但实际距离远
  C(3, 10)                   C: 3
```

跳表能 O(log n) 找到投影接近的候选，但候选可能不是真正的近邻。高维下投影冲突严重，跳表退化。

### 4.2 纯近邻图入口点选择慢

近邻图的贪心搜索很快（O(log n)），但**入口点选择**是瓶颈：

- 固定入口点：query 离入口远时搜索慢
- 暴力找入口点：O(n)，失去亚线性的意义
- 随机入口点：质量不稳定

没有好的入口点选择机制，近邻图的优势（贪心搜索快）发挥不出来。

### 4.3 互相补足

| 组件 | 强项 | 弱项 |
|---|---|---|
| 跳表 | O(log n) 查找、有序 | 高维投影冲突 |
| 近邻图 | 高维近邻搜索 | 入口点选择 |
| **HNSW** | **跳表给入口点，图给搜索路径** | — |

跳表解决「从哪开始搜」（O(log n) 选入口点），图解决「怎么走最快」（贪心搜索 O(log n)）。组合起来 O(log n) + O(log n) = O(log n)。

## 5. HNSW：跳表思想 + 近邻图

### 5.1 分层结构

HNSW 借鉴跳表的**分层思想**：不是建一张图，而是建多层图。

- **第 0 层**：所有节点，每节点连 M 个最近邻（密集图）
- **第 1 层**：约一半节点（抛硬币选），连 M 个最近邻（稀疏图）
- **第 L 层**：更少节点（顶层最稀疏）

```
第 2 层:  A ──────────→ D              (稀疏，跨大距离)
第 1 层:  A ───→ C ───→ D ───→ F       (中等)
第 0 层:  A → B → C → D → E → F → G    (密集，精确定位)
```

节点出现在第 l 层的概率 = p^l（抛硬币，和跳表一样）。顶层期望 log_{1/p}(n) 层。

### 5.2 搜索：从顶层贪心下降

```
hnsw_search(query, k):
  # 从顶层开始，贪心走到最近的节点
  entry = top_layer.entry_point
  for l = top_layer down to 1:
    entry = greedy_search(layer[l], query, entry, ef=1)   # 每层找最近
  # 第 0 层用 best-first 精确搜索
  results = best_first_search(layer[0], query, entry, ef=k)
  return top_k(results)
```

**上层**：图稀疏，节点间距离大，贪心搜索能快速跨大距离（几步走到 query 附近）。
**下层**：图密集，节点间距离小，best-first 精确定位近邻。

这和跳表搜索**完全一样**：从最高层粗定位，逐层下降精定位。跳表是「按 key 排序的分层链表」，HNSW 是「按近邻关系的分层图」。

### 5.3 插入：抛硬币定层 + 搜索找近邻

```
hnsw_insert(vector):
  l = random_level()           # 抛硬币决定出现在哪些层
  # 从顶层到 l+1 层：贪心走，找最近的入口点
  entry = top_layer.entry_point
  for layer = top down to l+1:
    entry = greedy_search(layer, query=vector, entry, ef=1)
  # 从 l 层到第 0 层：best-first 找 ef_construction 个候选，选 M 个连接
  for layer = l down to 0:
    candidates = best_first_search(layer, vector, entry, ef_construction)
    neighbors = select_M_nearest(candidates, vector)
    add_edges(vector, neighbors)
    entry = nearest(candidates)   # 下一层的入口点
```

插入时先抛硬币定层（和跳表一样），再用搜索找近邻连接。建图是增量的——每个新节点用已有图搜索找候选近邻，不需要全扫。

### 5.4 启发式邻居选择

选 M 个候选近邻时，不只选最近的，还要选**多样的**（方向不同）。避免所有邻居挤在一个方向，导致搜索路径单一。

```
select_neighbors_heuristic(candidates, M):
  result = []
  for c in candidates sorted by distance:
    if all(dist(c, r) > dist(c, query) for r in result):
      result.append(c)         # c 离 query 比离已选邻居都近 → 多样
      if len(result) == M: break
  return result
```

这保证邻居在各个方向都有，搜索时能从多方向接近 query。

## 6. C 实现逐行讲解

我们的 C 实现是**教学简化版**——没有分层结构，只有一层近邻图 + 跳表入口点选择。完整 HNSW 的分层逻辑见上节。

### 6.1 HNSW 结构

```c
// hnsw.h
typedef struct {
    HnswVector *vectors;        // 所有向量
    int **neighbors;            // 邻接表 neighbors[i][0..M-1]
    int *neighbor_count;        // 每节点的邻居数
    int n_nodes;
    int capacity;
    int dim;
    int M;                      // 最大连接数
    int entry_point;            // 入口点索引
    SkipList dist_index;        // 跳表：key=量化距离, value=节点索引
} HNSW;
```

`dist_index` 是跳表，key 是「向量到原点距离的量化值」，value 是节点索引。搜索时用 query 到原点距离查跳表，O(log n) 找到投影最接近的节点作为入口点。

### 6.2 插入

```c
int hnsw_insert(HNSW *h, const float *coords) {
    int idx = h->n_nodes;
    // 存向量
    for (int i = 0; i < h->dim; i++)
        h->vectors[idx].coords[i] = coords[i];

    // 算到原点距离，量化为 int，插入跳表
    float norm_sq = dist_sq_raw(coords, coords, h->dim);
    int key = quantize_dist(norm_sq);
    skiplist_insert(&h->dist_index, key, idx);

    if (h->entry_point == -1) { h->entry_point = idx; return idx; }

    // 暴力找最近邻（教学简化，O(n)）
    // 生产级 HNSW 用 best-first 搜索找候选，O(log n)
    int *cand = ...;  // 所有已有节点
    int *selected = ...;  // 选 M 个最近的
    for (int s = 0; s < M; s++) {
        add_edge(h, idx, selected[s]);      // 正向连接
        add_edge(h, selected[s], idx);      // 反向连接
    }
    return idx;
}
```

教学版用暴力找近邻（O(n)），保证图质量。生产级用搜索找候选（O(log n)），但图质量稍降。

### 6.3 搜索

```c
int hnsw_search(const HNSW *h, const float *query, int k, int *result) {
    // 1. 跳表找入口点：O(log n)
    float q_norm_sq = dist_sq_raw(query, query, h->dim);
    int q_key = quantize_dist(q_norm_sq);
    int entry = h->entry_point;
    int found_key, found_value;
    if (skiplist_find_closest(&h->dist_index, q_key, &found_key, &found_value))
        entry = found_value;     // 投影最接近的节点

    // 2. 贪心搜索：从入口点走向 query
    bool *visited = calloc(h->n_nodes, sizeof(bool));
    int current = entry;
    float current_dist = hnsw_distance_sq(h, current, query);
    while (true) {
        visited[current] = true;
        int best_neighbor = -1;
        float best_dist = current_dist;
        for (int i = 0; i < h->neighbor_count[current]; i++) {
            int nb = h->neighbors[current][i];
            if (visited[nb]) continue;
            float d = hnsw_distance_sq(h, nb, query);
            if (d < best_dist) { best_dist = d; best_neighbor = nb; }
        }
        if (best_neighbor == -1) break;   // 局部最优
        current = best_neighbor;
        current_dist = best_dist;
    }

    // 3. 收集 current 的 2 跳邻居，取 top-k
    ...
    return n_result;
}
```

**第 1 步**：跳表 O(log n) 找入口点。这是跳表给图的贡献——不用暴力找入口。
**第 2 步**：贪心搜索，每步看 M 个邻居。这是图给搜索的贡献——不用全扫。
**第 3 步**：从局部最优的 2 跳邻居里选 top-k，补充贪心可能漏掉的近邻。

### 6.4 跳表入口点选择的作用

```c
// 没有跳表：入口点固定，query 远时搜索慢
int entry = h->entry_point;  // 固定入口

// 有跳表：O(log n) 找投影最近的入口
int q_key = quantize_dist(q_norm_sq);
skiplist_find_closest(&h->dist_index, q_key, &found_key, &found_value);
int entry = found_value;  // 自适应入口
```

跳表让入口点**自适应**——query 在哪，入口点就在哪附近。这把贪心搜索的起始距离从 O(n) 降到 O(√n)（投影距离），大幅减少搜索步数。

## 7. 复杂度分析

| 操作 | 暴力 KNN | 简化 HNSW | 完整 HNSW |
|---|---|---|---|
| 建图 | — | O(n²·d) 暴力 | O(n·log n·M·d) 搜索建图 |
| 搜索 | O(n·d) | O(log n + M·log n·d) | O(log n·M·d) |
| 内存 | O(n·d) | O(n·d + n·M) | O(n·d + n·M) |

完整 HNSW 的搜索 O(log n·M·d)：跳表入口点 O(log n) + 贪心走 O(log n) 步 × 每步 M 邻居 × d 维距离。

**对比暴力**：n=1M, d=50, M=16 时，暴力 50M 次运算，HNSW 约 16·20·50 = 16000 次——**3000 倍加速**。实际受缓存、分支预测影响，约 100~1000 倍。

## 8. 与其他近邻搜索结构对比

| 结构 | 低维 (d≤10) | 高维 (d≥20) | 建图 | 适合场景 |
|---|---|---|---|---|
| KD-tree | O(log n) | O(n) 退化 | O(n log n) | 低维精确 |
| Ball-tree | O(log n) | O(n) 退化 | O(n log n) | 低中维 |
| LSH | O(n^ρ) ρ<1 | O(n^ρ) | O(n·L) | 高维近似 |
| **HNSW** | O(log n) | **O(log n)** | O(n log n) | **高维近似，召回率高** |

HNSW 在高维下保持 O(log n)（常数大些），且召回率通常 95%+，是当前向量检索的主流选择。

## 9. 总结

HNSW 的核心洞察：**跳表和近邻图是同一个思想的不同实例**——分层 + 贪心下降。

- 跳表：按 key 排序的分层链表，搜索从顶层粗定位到第 0 层精定位
- HNSW：按近邻关系的分层图，搜索从顶层（稀疏图）跨大距离到第 0 层（密集图）精定位

单独用跳表：高维投影冲突，找不到真正近邻。单独用近邻图：入口点选择慢。组合起来：跳表给 O(log n) 入口点，图给 O(log n) 搜索路径——亚线性高维向量检索。

下一文档 `ai_application.md` 讲 HNSW 在 AI 里的应用：RAG、推荐、搜索，以及 FAISS/Milvus/hnswlib 等生产级实现。