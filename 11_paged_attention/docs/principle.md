# 分页 KV Cache 原理：页表 + 固定块 + 空闲池 + 引用计数

> 本文档讲清「把操作系统的分页机制搬到 KV Cache 管理」的数据结构原理：虚拟内存与页表回顾、固定大小块 + 空闲块池、逻辑页号 → 物理块号映射、引用计数实现块共享、C 实现逐行讲解、复杂度分析。AI 应用（vLLM PagedAttention）见 `ai_application.md`。

## 1. 为什么 KV Cache 需要分页

大模型推理时，每个序列的 KV Cache 随 token 数线性增长。一个 batch 里多个序列长度各异，且在不断生成新 token。传统做法给每个序列**连续分配**一段显存，带来两个致命问题：

- **内部碎片**：预分配 max_len 的空间，序列实际只用了很短一部分，浪费严重
- **外部碎片**：序列长度可变，释放后显存里留下大小不一的空洞，新序列可能放不进去

这和操作系统早期遇到的内存管理问题**完全一样**。OS 的答案是**分页**——把内存切成等大的页框，按需分配，用页表映射逻辑地址到物理地址。vLLM 的 PagedAttention 把这套机制原封不动搬到了 KV Cache。

本文档讲清这套机制的数据结构原理。核心三件套：

| 数据结构 | 作用 | 对应 OS 概念 |
|---|---|---|
| **Block（固定块）** | 显存的最小分配单位，每块 block_size 个 slot | 物理页框（page frame） |
| **BlockPool（空闲块池）** | 管理所有块的分配/回收，free list 串联空闲块 | 伙伴系统 / 空闲页框链表 |
| **PageTable（页表）** | 每个序列一张，逻辑页号 → 物理块号 | 进程页表 |

再加一个**引用计数**实现块共享（多序列共享前缀），对应 OS 的共享内存页。

## 2. 操作系统分页机制回顾

### 2.1 虚拟内存

程序看到的是**连续的虚拟地址空间**，但物理内存可能不连续、甚至部分在磁盘上。OS 通过**页表**把虚拟地址翻译成物理地址，让程序以为自己在用一大块连续内存，实际物理页框可以分散在任意位置。

```
虚拟地址空间（程序看到的）     物理内存（实际的）
┌──────────┐  页表          ┌──────────┐
│ 页 0     │ ─────────────→ │ 页框 3   │
│ 页 1     │ ─────────────→ │ 页框 7   │
│ 页 2     │ ──× 缺页       │ 页框 1   │ ← 空闲
│ 页 3     │ ─────────────→ │ 页框 2   │
└──────────┘                └──────────┘
```

**关键洞察**：虚拟页号连续，物理页框可以不连续。这让 OS 能在物理内存有碎片时仍然满足程序的分配请求——只要还有空闲页框，就能分配，不用找连续区间。

### 2.2 页表

**页表**（Page Table）是虚拟页号 → 物理页框号的映射数组。每个进程一张。地址翻译：

```
虚拟地址 = [页号 | 页内偏移]
物理地址 = 页表[页号] 的页框号 + 页内偏移
```

```
页表:
  逻辑页 0 → 物理页框 3
  逻辑页 1 → 物理页框 7
  逻辑页 2 → 物理页框 2   (每个条目一个映射)
  逻辑页 3 → 未映射（缺页）
```

页表让"逻辑连续"和"物理分散"解耦。程序访问 page[2][offset] 时，OS 查页表得到物理页框 7，访问 frame[7][offset]。

### 2.3 缺页

访问未映射的页 → 触发**缺页中断** → OS 分配一个空闲页框 → 更新页表 → 恢复执行。这就是"按需分配"（demand paging）：只在实际访问时才分配物理内存，不预先分配整个地址空间。

KV Cache 的按需分配完全类似：序列生成新 token 时才申请新块，不预先分配最大长度。

### 2.4 固定大小页框

页框大小固定（通常 4KB），带来两个好处：

1. **无外部碎片**：任何空闲页框都能满足请求，不用找"足够大的连续区间"
2. **O(1) 分配**：从空闲链表取一个即可，不用搜索

代价是**内部碎片**：进程最后一页可能没填满，平均浪费半页。但页足够小（4KB vs GB 级内存），浪费可忽略。

KV Cache 的块同理：固定 block_size（vLLM 默认 16），无外部碎片，内部碎片最多 block_size - 1 个 slot。

## 3. 固定大小块（Block）

### 3.1 数据结构

每个块是显存的一小段，容纳 `block_size` 个 token 的 KV：

```c
typedef struct {
    int block_id;      // 物理块号（0 ~ n_blocks-1）
    int ref_count;     // 引用计数：多少个序列共享此块
    int used_slots;    // 已写入的 slot 数（0 ~ block_size）
    int next_free;     // 空闲链表的 next 指针（-1 = 链表末尾）
    bool in_use;       // 是否已分配
} Block;
```

- `block_id`：块的唯一编号，也是物理位置标识
- `ref_count`：引用计数，支持多序列共享前缀（见 §6）
- `used_slots`：块内已用 slot 数，用于算碎片化率
- `next_free`：空闲块用链表串联，`next_free` 指向下一个空闲块

### 3.2 块的大小选择

块大小是**利用率**和**管理开销**的权衡：

| 块大小 | 内部碎片 | 页表大小 | 分配次数 |
|---|---|---|---|
| 小（如 4） | 小（利用率高） | 大（每 token 1/4 页） | 多 |
| 大（如 128） | 大（每序列浪费多） | 小 | 少 |
| 16（vLLM 默认） | 适中 | 适中 | 适中 |

块越小，每序列最后一块浪费的 slot 越少，利用率越高。但页表项越多（序列长 1024 token，块大小 4 → 256 页表项 vs 块大小 16 → 64 项），管理开销越大。vLLM 实测 block_size=16 是 GPU kernel 友好性和利用率的平衡点。

## 4. 空闲块池（BlockPool）

### 4.1 数据结构

```c
typedef struct {
    Block *blocks;        // 所有块的数组（预分配 n_blocks 个）
    int n_blocks;         // 总块数
    int block_size;       // 每块 slot 数
    int free_list_head;   // 空闲链表头（-1 = 空）
    int n_free;           // 空闲块数
    int n_used;           // 已用块数
} BlockPool;
```

所有块预分配在一个数组里，空闲块用**侵入式链表**串联（`Block.next_free` 充当 next 指针，不额外分配链表节点）。

### 4.2 空闲链表

初始化时所有块都是空闲的，按 `block_id` 顺序串成链表：

```
free_list_head → block[0] → block[1] → ... → block[n-1] → -1
```

```c
for (int i = 0; i < n_blocks; i++) {
    blocks[i].next_free = (i + 1 < n_blocks) ? (i + 1) : -1;
    blocks[i].in_use = false;
}
free_list_head = 0;
```

### 4.3 分配块：O(1)

从链表头取一个块：

```c
int pool_alloc_block(BlockPool *pool) {
    if (pool->free_list_head == -1) return -1;     // 没有空闲块
    int id = pool->free_list_head;
    pool->free_list_head = pool->blocks[id].next_free;  // 头指针后移
    pool->blocks[id].in_use = true;
    pool->blocks[id].ref_count = 1;
    pool->blocks[id].used_slots = 0;
    pool->n_free--;
    pool->n_used++;
    return id;
}
```

**O(1)**——不搜索，直接取头。这是分页比连续分配快的根本原因：连续分配要找足够大的空闲区间（首次适应 O(n)），分页随便一个空闲块都行。

### 4.4 回收块：O(1)

归还块到链表头：

```c
void pool_release_block(BlockPool *pool, int id) {
    pool->blocks[id].in_use = false;
    pool->blocks[id].ref_count = 0;
    pool->blocks[id].used_slots = 0;
    pool->blocks[id].next_free = pool->free_list_head;  // 指向当前头
    pool->free_list_head = id;                          // 成为新头
    pool->n_free++;
    pool->n_used--;
}
```

**LIFO**（后进先出）：刚释放的块下次优先分配。这有良好的缓存局部性——刚用过的块可能还在 GPU L2 cache 里。

### 4.5 为什么不用位图

另一种方案是位图（bitmap），每位表示块是否空闲。分配时扫描位图找第一个 0：

- 位图省内存（n_blocks 位 vs 链表 n_blocks 个 next 指针）
- 但分配是 O(n/64)（要扫描字），链表是 O(1)
- 回收两者都 O(1)

链表用空间换时间，对 KV Cache（块数几千个）完全合理。

## 5. 页表（PageTable）：逻辑页号 → 物理块号

### 5.1 数据结构

每个序列一张页表，记录它的逻辑页映射到哪个物理块：

```c
typedef struct {
    int *pages;     // pages[i] = 物理块号，-1 = 未映射
    int n_pages;    // 已用页数
    int capacity;   // 数组容量（动态扩容）
} PageTable;
```

`pages` 是动态数组，`pages[i]` 存逻辑页 `i` 对应的物理块号。序列的第 `t` 个 token 在逻辑页 `t / block_size`、页内偏移 `t % block_size`，物理位置由 `pages[t / block_size]` 给出。

### 5.2 地址翻译

给定序列的 token 索引 `t`，找物理块：

```c
int kv_block_of_token(const KVCacheManager *mgr, int seq_id, int token_idx) {
    int slot = kv_find_seq(mgr, seq_id);
    const Sequence *seq = &mgr->seqs[slot];
    if (token_idx < 0 || token_idx >= seq->length) return -1;
    int page = token_idx / mgr->block_size;       // 逻辑页号
    return seq->page_table.pages[page];           // 物理块号
}
```

这就是 OS 地址翻译的翻版：`虚拟地址 → [页号 | 偏移] → 查页表 → 物理页框`。

### 5.3 动态扩容

序列不断生成新 token，页表要能增长。用动态数组，容量不够时翻倍：

```c
static int pt_ensure_capacity(PageTable *pt, int needed) {
    if (needed <= pt->capacity) return 0;
    int new_cap = pt->capacity == 0 ? 4 : pt->capacity;
    while (new_cap < needed) new_cap *= 2;
    int *new_pages = realloc(pt->pages, new_cap * sizeof(int));
    for (int i = pt->capacity; i < new_cap; i++)
        new_pages[i] = -1;                        // 新页未映射
    pt->pages = new_pages;
    pt->capacity = new_cap;
    return 0;
}
```

均摊 O(1) 追加。序列长度不可预测，动态扩容是必须的。

### 5.4 为什么每个序列一张页表

OS 每个进程一张页表，因为每个进程有自己的虚拟地址空间。KV Cache 同理：每个序列看到的是连续的逻辑 token 位置（token 0, 1, 2, ...），但物理块可以分散在各处。页表让序列**无需关心物理布局**，只管逻辑位置。

多序列共享块时，不同序列的页表可以指向**同一个物理块**——这正是引用计数要解决的。

## 6. 引用计数：块共享

### 6.1 为什么要共享

多个序列常有**相同前缀**（same system prompt、same few-shot examples）。如果每个序列各存一份前缀的 KV，显存浪费巨大。理想做法是让多个序列的页表指向**同一组物理块**，前缀只存一份。

```
序列 A: "请翻译以下句子：Hello"     前缀 "请翻译以下句子：" 共享
序列 B: "请翻译以下句子：World"     前缀 "请翻译以下句子：" 共享

页表 A: [块0, 块1, 块2, 块3]  ← 块0,1 是共享前缀
页表 B: [块0, 块1, 块4, 块5]  ← 块0,1 是同一物理块
        ref_count[块0] = 2
        ref_count[块1] = 2
```

### 6.2 引用计数

每个块维护 `ref_count`：有多少个序列的页表指向它。

- 分配新块：`ref_count = 1`
- 共享（页表指向已有块）：`ref_count++`
- 释放序列：遍历其页表，每块 `ref_count--`；归零才真正归还块池

```c
int kv_free_seq(KVCacheManager *mgr, int seq_id) {
    Sequence *seq = &mgr->seqs[slot];
    for (int p = 0; p < seq->page_table.n_pages; p++) {
        int block_id = seq->page_table.pages[p];
        mgr->pool.blocks[block_id].ref_count--;
        if (mgr->pool.blocks[block_id].ref_count <= 0) {
            pool_release_block(&mgr->pool, block_id);   // 引用归零，回收
        }
    }
    ...
}
```

### 6.3 共享前缀

`kv_share_prefix` 把源序列的前 n_tokens 个块共享给目标序列：

```c
int kv_share_prefix(KVCacheManager *mgr, int src_seq_id, int dst_seq_id, int n_tokens) {
    int blocks_to_share = (n_tokens + block_size - 1) / block_size;
    for (int p = 0; p < blocks_to_share; p++) {
        int block_id = src->page_table.pages[p];
        mgr->pool.blocks[block_id].ref_count++;      // 引用 +1
        pt_append(&dst->page_table, block_id);        // 目标页表指向同一块
    }
    ...
}
```

不复制数据，只增加引用计数和页表项——**O(共享块数)**，不复制 KV 数据。这是 prefix caching 的核心。

### 6.4 写时复制（Copy-on-Write）

共享块后，如果某个序列要**修改**块里的 KV（比如继续生成 token，写到最后一块的空 slot），而块被共享（`ref_count > 1`），就需要**写时复制**：复制一份块给这个序列独占，原块引用减 1。

本 C 实现简化了 CoW：`kv_append_token` 假设最后一块不被共享（或调用者保证语义正确）。vLLM 的生产实现会检查 `ref_count`，必要时 CoW。详见 `ai_application.md`。

## 7. KVCacheManager：管理器

### 7.1 数据结构

```c
typedef struct {
    BlockPool pool;          // 全局块池
    Sequence *seqs;          // 序列数组
    int n_seqs;              // 序列数
    int seq_capacity;        // 序列容量
    int block_size;          // 块大小
} KVCacheManager;

typedef struct {
    int seq_id;              // 序列 ID（用户指定）
    int length;              // 已分配 token 数
    PageTable page_table;    // 该序列的页表
    bool active;             // 是否活跃
} Sequence;
```

管理器持有全局块池和所有序列。每个序列有自己的页表。`seq_id` 是用户指定的逻辑 ID（如请求 ID），内部用 `seqs[]` 数组槽位管理。

### 7.2 分配序列

```c
int kv_allocate_seq(KVCacheManager *mgr, int seq_id, int n_tokens) {
    int slot = kv_find_free_seq_slot(mgr);        // 找空闲序列槽
    Sequence *seq = &mgr->seqs[slot];
    seq->seq_id = seq_id;
    seq->active = true;
    kv_alloc_blocks_for_seq(mgr, seq, n_tokens);  // 按需分配块
    return 0;
}
```

`kv_alloc_blocks_for_seq` 算出需要的块数 `ceil(n_tokens / block_size)`，逐个从块池分配，加入页表，设置每块的 `used_slots`。

### 7.3 追加 token

序列生成新 token 时，可能需要新块：

```c
int kv_append_token(KVCacheManager *mgr, int seq_id, int n_tokens) {
    // 1. 先填满当前最后一块的剩余 slot
    // 2. 剩余的 token 申请新块
    while (n_tokens > 0) {
        int block_id = pool_alloc_block(&mgr->pool);
        pt_append(&seq->page_table, block_id);
        int fill = min(n_tokens, block_size);
        mgr->pool.blocks[block_id].used_slots = fill;
        n_tokens -= fill;
    }
}
```

先填满最后一块的空 slot（避免浪费），再按需申请新块。这是"按需分页"——只在需要时才分配物理块，不预分配。

### 7.4 统计

```c
typedef struct {
    int total_blocks;        // 总块数
    int allocated_blocks;    // 已分配块数
    int free_blocks;         // 空闲块数
    int total_slots;         // 总 slot 数
    int used_slots;          // 已用 slot 数（实际 token 数）
    int wasted_slots;        // 浪费 slot 数（已分配块内未用）
    double usage_ratio;      // 显存利用率 = used_slots / total_slots
    double fragmentation_ratio;  // 碎片化率 = wasted / (allocated * block_size)
    int n_active_seqs;       // 活跃序列数
    int total_tokens;        // 总 token 数
} CacheStats;
```

两个关键指标：

- **显存利用率** `usage_ratio` = 实际 token 数 / 总显存容量。越高越好。
- **碎片化率** `fragmentation_ratio` = 已分配块内浪费 slot / 已分配块总 slot。这是**内部碎片**，只发生在每序列最后一块。分页方案没有外部碎片。

## 8. C 实现逐行讲解

### 8.1 初始化（kv_init）

```c
int kv_init(KVCacheManager *mgr, int n_blocks, int block_size, int max_seqs) {
    mgr->pool.blocks = malloc(n_blocks * sizeof(Block));
    for (int i = 0; i < n_blocks; i++) {
        mgr->pool.blocks[i].next_free = (i + 1 < n_blocks) ? (i + 1) : -1;
        mgr->pool.blocks[i].in_use = false;       // 全部空闲
    }
    mgr->pool.free_list_head = 0;                 // 空闲链表从块 0 开始
    mgr->pool.n_free = n_blocks;
    ...
}
```

预分配所有块，串成空闲链表。O(n_blocks) 一次性初始化。

### 8.2 分配块（pool_alloc_block）

```c
int pool_alloc_block(BlockPool *pool) {
    int id = pool->free_list_head;                // 取链表头
    pool->free_list_head = pool->blocks[id].next_free;  // 头后移
    pool->blocks[id].in_use = true;
    pool->blocks[id].ref_count = 1;               // 新分配，引用 1
    pool->n_free--;
    pool->n_used++;
    return id;
}
```

O(1) 取头。对比连续分配的首次适应 O(n)，这是分页的核心优势。

### 8.3 页表追加（pt_append）

```c
static int pt_append(PageTable *pt, int block_id) {
    pt_ensure_capacity(pt, pt->n_pages + 1);      // 容量不够则扩容
    pt->pages[pt->n_pages] = block_id;            // 追加映射
    pt->n_pages++;
    return 0;
}
```

均摊 O(1)（动态数组翻倍扩容）。

### 8.4 释放序列（kv_free_seq）

```c
int kv_free_seq(KVCacheManager *mgr, int seq_id) {
    for (int p = 0; p < seq->page_table.n_pages; p++) {
        int block_id = seq->page_table.pages[p];
        mgr->pool.blocks[block_id].ref_count--;   // 引用减 1
        if (mgr->pool.blocks[block_id].ref_count <= 0) {
            pool_release_block(&mgr->pool, block_id);  // 归零才回收
        }
    }
    pt_clear(&seq->page_table);
    seq->active = false;
}
```

遍历页表，每块引用减 1。引用归零才归还块池——支持共享块的正确回收。

### 8.5 统计（kv_usage_stats）

```c
CacheStats kv_usage_stats(const KVCacheManager *mgr) {
    for (int i = 0; i < mgr->pool.n_blocks; i++) {
        if (mgr->pool.blocks[i].in_use) {
            used_slots += mgr->pool.blocks[i].used_slots;
            allocated_slots += mgr->pool.block_size;
        }
    }
    stats.usage_ratio = (double)used_slots / total_slots;
    stats.fragmentation_ratio = 1.0 - (double)used_slots / allocated_slots;
}
```

遍历所有块统计。O(n_blocks)。`usage_ratio` 是全局利用率，`fragmentation_ratio` 是已分配块内的内部碎片率。

## 9. 复杂度分析

| 操作 | 时间 | 空间 |
|---|---|---|
| `kv_init(n, b, s)` | O(n + s) | O(n + s) |
| `kv_allocate_seq(L)` | O(ceil(L/b)) | O(ceil(L/b)) 页表项 |
| `kv_append_token(k)` | O(ceil(k/b)) | O(ceil(k/b)) |
| `kv_free_seq()` | O(页表长度) | 释放页表 |
| `kv_share_prefix(L)` | O(ceil(L/b)) | O(ceil(L/b)) 页表项（不复制数据） |
| `kv_block_of_token(t)` | O(1) | O(1) |
| `kv_usage_stats()` | O(n_blocks) | O(1) |

关键点：

- **分配/释放是 O(块数)**，不是 O(序列长度)。块大小 16 时，1024 token 的序列只有 64 个块，操作 O(64)。
- **地址翻译 O(1)**：一次除法 + 一次数组访问。GPU kernel 里这是关键——PagedAttention 的 attention 计算要频繁翻译地址，O(1) 保证不拖慢计算。
- **共享前缀不复制数据**：只增加引用计数和页表项，O(页数) 而非 O(token 数)。前缀越长，省的复制越多。
- **无外部碎片**：任何空闲块都能分配，不用找连续区间。这是分页比连续分配的根本优势。
- **内部碎片 ≤ block_size - 1 per 序列**：每序列最后一块可能未满。block_size=16 时最多浪费 15 slot，序列越长浪费占比越小。

## 10. 与连续分配的对比

| 维度 | 连续分配 | 分页分配 |
|---|---|---|
| 分配 | 找连续区间 O(n) | 取空闲块 O(1) |
| 释放 | 合并相邻区间 O(n) | 归还块池 O(1) |
| 外部碎片 | 严重（空洞无法用） | 无 |
| 内部碎片 | 无（精确分配） | 每序列 ≤ block_size-1 |
| 地址翻译 | 直接（连续） | 查页表 O(1) |
| 共享前缀 | 难（要复制） | 易（页表指向同块 + 引用计数） |
| 扩容 | 难（要搬移） | 易（申请新块 + 页表追加） |

分页在分配/释放/共享/扩容上都优于连续分配，代价是地址翻译多一次查表。对 KV Cache 来说，这个代价远小于碎片化带来的浪费——vLLM 实测分页把显存利用率从 30-60% 提升到 95%+。

## 11. 总结

分页 KV Cache 的四个关键思想：

1. **固定块**：显存切成等大块，分配单位是块不是 slot。消除外部碎片。
2. **空闲块池**：空闲块用链表串联，O(1) 分配/回收。不搜索连续区间。
3. **页表**：每序列一张，逻辑页号 → 物理块号。让逻辑连续和物理分散解耦。
4. **引用计数**：块可被多序列共享（前缀复用），引用归零才回收。支持 prefix caching。

这套机制把操作系统的分页思想原封不动搬到 KV Cache：块 = 页框，页表 = 进程页表，空闲池 = 空闲页框链表，引用计数 = 共享内存页。结果是显存利用率从 30-60% 提升到 95%+，并发序列数翻倍——这就是 vLLM PagedAttention 的核心创新。