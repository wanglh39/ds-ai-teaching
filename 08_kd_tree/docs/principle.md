# KD-Tree（k 维树）原理

> 本文档讲清「KD-Tree 是什么、怎么把二分搜索推广到 k 维、按轴交替切分 + median 平衡、最近邻搜索 + 剪枝、为什么高维退化（curse of dimensionality）、C 实现逐行讲解、复杂度」。AI 应用（KNN 加速）见 `ai_application.md`。

## 1. KD-Tree 的定义：二分搜索的 k 维推广

**KD-Tree**（k-Dimensional Tree）是一种对 **k 维空间**中的点集进行**二分切分**的树形结构。它由 Bentley 在 1975 年提出，核心思想一句话：**把一维二分搜索的「按中位数切分」推广到 k 维，每次换一个轴切**。

先回顾一维二分搜索为什么快：n 个排好序的数，找目标值每次折半，$O(\log n)$ 次比较就够。关键在于**每次切分都把搜索空间一分为二**，且查询时能根据「目标在左还是在右」只搜一侧。

到了 k 维，每个点有 k 个坐标，没有天然的「大小顺序」。KD-Tree 的做法是：

- **第 0 层**按第 0 轴（x 轴）切分：选 x 坐标的中位数点做根，x 比它小的进左子树，大的进右子树
- **第 1 层**按第 1 轴（y 轴）切分：在左右子树里分别按 y 坐标中位数再切
- **第 2 层**按第 2 轴（z 轴）切分（如果 k=3）
- ……
- **第 k 层**回到第 0 轴（轴交替：`axis = depth % k`）

这样每 k 层就把所有轴都切了一遍，保证树在各个维度上都「均衡」，树高 $O(\log n)$。

```
2D 点集 (按 x、y 交替切分):

      第0层: 按 x 轴切分（竖线）
      ┌─────────────┬─────────────┐
      │  x < median │  x > median │
      │   (左子树)  │   (右子树)  │
      └──────┬──────┴──────┬──────┘
             │             │
      第1层: 按 y 轴切分（横线）
      ┌─────┬─────┐ ┌─────┬─────┐
      │y<med│y>med│ │y<med│y>med│
      └─────┴─────┘ └─────┴─────┘
```

直观上，KD-Tree 把空间递归切成越来越小的「超矩形」区域，每个叶子区域里只剩很少的点。查询最近邻时，先定位查询点落在哪个区域，再回溯检查相邻区域有没有更近的点——这就是剪枝。

## 2. 从一维二分到 k 维：按轴交替切分

### 2.1 一维：排序数组 + 二分

一维空间（数轴）上的 n 个点，排序后用二分查找：

```
点集: [1, 3, 5, 7, 9, 11, 13, 15]
查询: 8 → 落在 7 和 9 之间 → 最近邻是 7 或 9
树高: log2(8) = 3，3 次比较搞定
```

这本质上就是一棵「按中位数切分」的平衡二叉搜索树（BST）。

### 2.2 二维：KD-Tree 按 x、y 交替切

到了二维，点没有全序，但我们可以**轮流**按 x 和 y 排序切分：

```
点集: (2,3) (5,4) (9,6) (4,7) (8,1) (7,2) (1,8)

构建过程（depth=0 切 x，depth=1 切 y，交替）:

depth=0, 切 x 轴:
  按 x 排序: (1,8)(2,3)(4,7)(5,4)(7,2)(8,1)(9,6)
  中位数点: (5,4)  → 根节点，split_axis=0
  左子树(x<5): (1,8)(2,3)(4,7)
  右子树(x>5): (7,2)(8,1)(9,6)

depth=1, 左子树切 y 轴:
  按 y 排序: (2,3)(4,7)(1,8)
  中位数点: (4,7)  → split_axis=1
  左(y<7): (2,3)
  右(y>7): (1,8)

depth=1, 右子树切 y 轴:
  按 y 排序: (8,1)(7,2)(9,6)
  中位数点: (7,2)  → split_axis=1
  左(y<2): (8,1)
  右(y>2): (9,6)

最终 KD-Tree:
                  (5,4) [axis=x]
                 /        \
            (4,7)[y]      (7,2)[y]
            /    \        /    \
        (2,3)  (1,8)  (8,1)  (9,6)
```

注意每个节点存了**它当时切分用的轴** `split_axis`，搜索时靠这个判断查询点在分裂超平面的哪一侧。

### 2.3 为什么要「交替」而不是「总切同一个轴」

如果总切 x 轴，那 y 方向的点永远不被分离，所有 y 相近的点会挤在同一棵子树里，查询时剪枝失效。**交替切分**保证每个维度都被均匀划分，树在所有方向上都「细长」不起来。

更严格地说，每 k 层切完一圈所有轴，任意一个超矩形区域在经过 $O(\log n)$ 层后体积缩到 $1/n$，所以树高 $O(\log n)$，查询效率有保证。

### 2.4 median 选取：为什么用中位数而不是随便切

切分点选**中位数**是为了**平衡**：左右子树点数最多差 1，树高严格 $O(\log n)$。

如果随便选一个点切分，最坏情况退化成链表（每次都选到最值点），树高 $O(n)$，查询退化成线性扫描。中位数切分是 KD-Tree 高效的根基。

工程上找中位数有三种做法：

| 方法 | 时间 | 本实现 | 说明 |
|---|---|---|---|
| 排序取中间 | $O(n \log n)$ 每层 | ✓ 本实现 | 简单，用 qsort |
| 快速选择 nth_element | $O(n)$ 每层 | C++ 标准库 | 最优，总构建 $O(n \log n)$ |
| 排序一次 + 按轴分桶 | $O(n \log n)$ 总 | 复杂 | 预处理贵但构建快 |

本实现用 qsort 每层排序，构建总复杂度 $O(n \log^2 n)$，简单清晰，适合教学。

## 3. 最近邻搜索 + 剪枝：KD-Tree 的核心算法

最近邻搜索（1-NN）：给查询点 $q$，在树里找距离 $q$ 最近的点。算法是**递归 + 剪枝**。

### 3.1 朴素递归（不剪枝）：遍历所有点

最朴素的递归就是「左右子树都搜」，等价于遍历所有点，$O(n)$，没意义。KD-Tree 的威力在于**剪枝**：很多子树可以证明「不可能含更近的点」，直接跳过。

### 3.2 剪枝原理：分裂超平面距离

关键观察：节点 $v$ 按轴 $a$ 切分，分裂超平面是 $x_a = v.point[a]$ 这条线（2D 是线，3D 是面，kD 是超平面）。查询点 $q$ 到这个超平面的**垂直距离**是：

$$
\text{轴距离} = |q[a] - v.point[a]|
$$

如果当前已找到的最近邻距离 $d_{\text{best}}$ **小于**这个轴距离，那么超平面另一侧的所有点都不可能比当前最优更近——因为它们到 $q$ 的距离**至少**是轴距离（垂直距离是两点距离的下界）。

```
剪枝示意（2D，切 x 轴）:

  分裂超平面 x = v.x
        │
   左  │  右
  ─────┼──────
        │
        │     q (查询点在右侧)
        │
   左侧所有点到 q 的距离 ≥ |q.x - v.x|
   若 |q.x - v.x| > d_best，左侧整棵子树可剪掉
```

### 3.3 完整搜索流程

```
nearest(node, q, best, best_dist):
    if node == NULL: return
    # 1. 检查当前节点
    d = dist(node.point, q)
    if d < best_dist:
        best = node; best_dist = d
    # 2. 决定先搜哪侧：q 离哪侧近先搜哪侧
    axis = node.split_axis
    diff = q[axis] - node.point[axis]
    if diff < 0:
        near, far = node.left, node.right
    else:
        near, far = node.right, node.left
    # 3. 先搜近侧（更可能找到更近的点，缩小 best_dist）
    nearest(near, q, best, best_dist)
    # 4. 剪枝：轴距离² < best_dist² 才搜远侧
    if diff * diff < best_dist:
        nearest(far, q, best, best_dist)
```

几个要点：

1. **先搜近侧**：近侧更可能有更近的点，先搜能尽快缩小 `best_dist`，让远侧剪枝更狠
2. **剪枝用距离平方**：避免开方，`diff*diff < best_dist_sq` 等价但更快
3. **剪枝条件是 `<` 不是 `≤`**：等于时远侧理论上可能有等距点，但最近邻只要任选一个，用 `<` 不影响正确性
4. **best_dist 在递归中是引用/全局**：近侧搜索后 best_dist 可能变小，远侧剪枝用的是**更新后**的值，这是剪枝有效的关键

### 3.4 KNN（k 近邻）扩展

求前 K 个最近邻只需把「单个 best」换成「大小为 K 的最大堆」：

- 堆未满（$< K$）：直接插入
- 堆满且新距离 < 堆顶（当前第 K 远）：替换堆顶
- 剪枝条件：堆满时，轴距离² < 堆顶距离² 才搜远侧

本实现的 C 版做 1-NN（K=1 特例），Python 版做通用 KNN。

## 4. 为什么高维退化：curse of dimensionality

KD-Tree 在低维（k=2,3）极快，但维度一高就**退化**，甚至比暴力还慢。这就是著名的**维度灾难**（curse of dimensionality）。

### 4.1 现象：高维下剪枝失效

看 demo 实测的**访问节点数**（n=10000，K=5）：

| 维度 | 访问节点数 | 占比 | 剪枝效果 |
|---|---|---|---|
| 2 | 35 | 0.3% | 极强，几乎只搜一条路径 |
| 10 | 7087 | 70.9% | 大部分子树都得搜 |
| 50 | 10000 | 100% | 完全失效，全遍历 |
| 100 | 10000 | 100% | 完全失效，全遍历 |

dim=2 时只访问 0.3% 的节点，剪枝把 99.7% 的子树砍掉了；dim=100 时一个都砍不掉，KD-Tree 退化成 $O(n)$ 全遍历，还比暴力多了递归开销。

### 4.2 根本原因：高维下所有点距离趋近

数学上，高维空间有个反直觉性质：**随机采样的点之间距离会高度集中**，所有点离查询点「差不多远」。

具体来说，n 个点在 d 维单位超立方体里均匀采样，任意两点距离的期望约 $\sqrt{d/6}$，而距离的**方差随 d 增大相对缩小**——意思是 d 大时，最近邻和最远邻的距离差别相对于绝对距离变得很小。

```
低维 (2D):                      高维 (100D):
最近邻距离 0.1                  最近邻距离 3.2
最远邻距离 5.0                  最远邻距离 3.8
比值 0.02（差异大）             比值 0.84（差异小）
→ 剪枝能砍掉大片区域            → 几乎所有点都"可能更近"，剪不掉
```

剪枝条件是「轴距离² < best_dist²」。高维下 best_dist 和所有点到查询点的距离都差不多，而轴距离只是某一维的分量，平方后比 best_dist² 小的概率很高——**剪枝条件几乎总成立**，远侧子树几乎总得搜。

### 4.3 退化阈值：k 约 10-20

经验上，KD-Tree 在 $k \le 10$ 时还有优势，$k > 20$ 基本退化。精确阈值取决于数据分布和 n，但「KD-Tree 只适合低维」是公认结论。这也是为什么：

- 2D/3D 几何问题（最近点对、碰撞检测）用 KD-Tree 很好
- AI 里的高维向量（embedding 100-768 维）**不用** KD-Tree，改用 HNSW、FAISS 等

### 4.4 纯 Python KD-Tree 的额外开销

本 demo 的 Python KD-Tree 在小数据量低维下甚至比 numpy 向量化暴力**慢**（n=1000, dim=2：KD-Tree 18.9ms vs brute 6.9ms）。这是工程层的常数开销：

- numpy 暴力是 C 层向量化的 `einsum`，算 n 个距离一次完成，极快
- 纯 Python KD-Tree 每次递归都有解释器开销、属性访问、堆操作

所以**算法优势**要看访问节点数（dim=2 只 0.3%），**绝对耗时**要看实现语言。C 实现里 KD-Tree 比 C 暴力快 2180x（无向量化暴力，KD-Tree 剪枝威力全显）。生产环境要用 C/C++/Rust 实现的 KD-Tree 才能发挥低维优势。

## 5. C 实现逐行讲解

本实现固定 k=2（2D），文件 `08_kd_tree/c/kd_tree.h` + `kd_tree.c`。

### 5.1 数据结构（kd_tree.h）

```c
#define KD_DIM 2

typedef struct KDNode {
    double point[KD_DIM];   // 该点的坐标
    int split_axis;          // 切分轴（0=x, 1=y）
    struct KDNode *left;
    struct KDNode *right;
} KDNode;

typedef struct {
    KDNode *root;
    size_t size;
} KDTree;
```

每个节点存**点坐标 + 切分轴 + 左右孩子**。`split_axis` 是搜索时判断走哪侧的关键。

### 5.2 构建（kd_build → build_rec）

```c
static KDNode *build_rec(KDPoint *pts, size_t n, int depth) {
    if (n == 0) return NULL;
    int axis = depth % KD_DIM;                    // 轴交替
    qsort(pts, n, sizeof(KDPoint),                // 按当前轴排序
          axis == 0 ? cmp_axis0 : cmp_axis1);
    size_t mid = n / 2;                           // 中位数下标
    KDNode *node = malloc(sizeof(KDNode));
    for (int i = 0; i < KD_DIM; i++)
        node->point[i] = pts[mid].p[i];           // median 点做节点
    node->split_axis = axis;
    node->left  = build_rec(pts, mid, depth + 1); // 左：median 之前
    node->right = build_rec(pts + mid + 1,        // 右：median 之后
                            n - mid - 1, depth + 1);
    return node;
}
```

逐行：

1. `axis = depth % KD_DIM`：轴交替，depth=0 切 x，depth=1 切 y，depth=2 又切 x…
2. `qsort` 按当前轴排序，让中位数落在中间。`cmp_axis0`/`cmp_axis1` 是两个比较函数
3. `mid = n / 2`：排序后中间那个就是中位数，保证左右子树大小差 ≤ 1
4. 递归构建左右子树，`pts + mid + 1` 跳过 median 点本身

构建前 `kd_build` 把二维数组拷到 `KDPoint` 工作数组（避免改原数组），构建后释放。

### 5.3 最近邻搜索（kd_nearest → nearest_rec）

```c
static void nearest_rec(const KDNode *node, const double query[KD_DIM],
                        const KDNode **best, double *best_dist_sq) {
    if (node == NULL) return;
    double d = kd_dist_sq(node->point, query);    // 当前节点距离²
    if (d < *best_dist_sq) {                      // 更优则更新
        *best_dist_sq = d;
        *best = node;
    }
    int axis = node->split_axis;
    double diff = query[axis] - node->point[axis]; // 轴距离（带符号）
    const KDNode *near_child = (diff < 0.0) ? node->left : node->right;
    const KDNode *far_child  = (diff < 0.0) ? node->right : node->left;
    nearest_rec(near_child, query, best, best_dist_sq);  // 先搜近侧
    if (diff * diff < *best_dist_sq) {                   // 剪枝
        nearest_rec(far_child, query, best, best_dist_sq);
    }
}
```

逐行：

1. `kd_dist_sq` 算距离平方（不开方，省一次 sqrt）
2. `diff = query[axis] - node->point[axis]`：查询点在切分轴上离超平面的距离（带符号，符号决定哪侧近）
3. `near_child`/`far_child`：根据 diff 符号选近侧远侧。`diff < 0` 说明查询点在切分轴上比节点小，进左子树（左子树存的是 axis 值更小的点）
4. **先搜近侧**：`nearest_rec(near_child, ...)`，这一步可能缩小 `*best_dist_sq`
5. **剪枝**：`diff * diff < *best_dist_sq` 才搜远侧。注意这里用的是**近侧搜完更新后**的 `*best_dist_sq`，这是剪枝有效的关键——近侧找到更近的点后，远侧更可能被剪掉

`best` 和 `best_dist_sq` 用指针传，让递归各层共享同一份最优状态。

### 5.4 测试（test.c）

测试覆盖：

- `test_empty`：空树 nearest 返回 NULL
- `test_single`：单点树
- `test_known_2d`：5 个已知点，5 个查询点，验证最近邻坐标
- `test_query_is_point`：查询点恰是树中点，距离应为 0
- `test_duplicate_points`：重复点
- `test_dist_sq`：距离平方函数
- `test_correctness_vs_brute`：200 个随机点，500 个随机查询，对比 KD-Tree 和暴力法结果
- `test_stress_correctness`：2000 个点，1000 个查询，对比暴力法
- `test_axis_alternation`：验证根切 x 轴、深度 1 切 y 轴

正确性测试的核心是「和暴力法对比」——暴力法一定对，KD-Tree 要和它一致。

## 6. 复杂度

| 操作 | 平均 | 最坏 | 说明 |
|---|---|---|---|
| 构建 | $O(n \log n)$¹ | $O(n \log n)$¹ | 本实现 qsort 每层 $O(n \log n)$，共 $\log n$ 层 |
| 最近邻查询 | $O(\log n)$（低维） | $O(n)$（高维退化） | 低维剪枝有效，高维退化全遍历 |
| KNN 查询 | $O(\log n + K)$（低维） | $O(n)$ | 维护大小 K 堆 |
| 空间 | $O(n)$ | $O(n)$ | 每个点一个节点 |

¹ 用 `nth_element`（快速选择）找中位数可降到 $O(n \log n)$ 总构建；本实现用 qsort 是 $O(n \log^2 n)$。

**查询复杂度的维度依赖**更精确的形式是 $O(2^k + \log n)$，其中 $k$ 是维度。$k$ 小时 $2^k$ 是常数，$O(\log n)$ 主导；$k$ 大时 $2^k$ 爆炸，退化成 $O(n)$。这就是「低维 $O(\log n)$、高维 $O(n)$」的数学根源。

## 7. KD-Tree vs 其他空间数据结构

| 结构 | 适合维度 | 构建 | 查询 | 增删 | 典型场景 |
|---|---|---|---|---|---|
| **KD-Tree** | 低（≤10） | $O(n \log n)$ | $O(\log n)$ | 难（可能要重建） | 2D/3D 几何、低维 KNN |
| Ball Tree | 中（≤30） | $O(n \log n)$ | $O(\log n)$ | 难 | 中维 KNN，比 KD-Tree 抗高维 |
| R-Tree | 低（2D/3D） | $O(n \log n)$ | $O(\log n)$ | 支持 | 空间数据库、GIS |
| **HNSW** | 高（100+） | $O(n \log n)$ | $O(\log n)$ | 支持 | 高维 ANN（embedding 检索） |
| 暴力 | 任意 | $O(1)$ | $O(n)$ | 支持 | 基线、小数据 |

KD-Tree 的定位是**低维精确最近邻**。高维近似最近邻（ANN）请用 HNSW（第 12 章）或 FAISS。

## 8. 小结

- KD-Tree = **二分搜索的 k 维推广**：按轴交替切分，median 保证平衡
- 构建：每层选 `depth % k` 轴，排序取中位数切分，递归构建左右子树
- 查询：递归搜近侧 + 用分裂超平面距离剪枝远侧，低维 $O(\log n)$
- 高维退化：维度灾难使所有点距离趋近，剪枝条件几乎总成立，退化成 $O(n)$
- 实践：低维（≤10）用 KD-Tree，高维用 HNSW/FAISS