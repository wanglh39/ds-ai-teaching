# 分块矩阵乘法：数组连续性在 AI 里的压榨

> 本文档讲清「矩阵乘法为什么是 AI 的核心运算、朴素实现有什么缓存问题、分块怎么解决」。配套的原理文档见 `principle.md`。

## 1. 矩阵乘法是 AI 的心脏

### 1.1 神经网络的本质是矩阵乘法

一个全连接层：

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

其中 $W \in \mathbb{R}^{m \times n}$，$\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} \in \mathbb{R}^m$。核心就是矩阵-向量乘法。

一个 batch 的前向传播：

$$
Y = X W^T + B
$$

其中 $X \in \mathbb{R}^{B \times n}$，$W \in \mathbb{R}^{m \times n}$，$Y \in \mathbb{R}^{B \times m}$。这就是矩阵乘法。

Transformer 的注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V
$$

$Q K^T$ 是矩阵乘法，$\text{softmax}(\cdot) V$ 也是矩阵乘法。

**一个 LLM 推理 90%+ 的时间在矩阵乘法**。所以矩阵乘法快一点，整个 AI 就快一点。

### 1.2 矩阵乘法的定义

$$
C = A B, \quad C_{ij} = \sum_{k=0}^{p-1} A_{ik} B_{kj}
$$

其中 $A \in \mathbb{R}^{m \times p}$，$B \in \mathbb{R}^{p \times n}$，$C \in \mathbb{R}^{m \times n}$。

朴素三重循环：

```python
for i in range(m):
    for j in range(n):
        s = 0
        for k in range(p):
            s += A[i][k] * B[k][j]
        C[i][j] = s
```

总运算量 $O(mnp)$。对 $n \times n$ 方阵，$O(n^3)$。

### 1.3 朴素实现的问题

朴素三重循环在数学上正确，但在现代硬件上慢，原因是**缓存不友好**。

看最内层循环 `for k`：
- `A[i][k]`：`k` 递增，`A[i][0], A[i][1], ...` 连续访问，**缓存友好**
- `B[k][j]`：`k` 递增，但 `B` 是行优先存储，`B[k][j]` 和 `B[k+1][j]` 相隔一整行（`n * sizeof(float)` 字节），**缓存不友好**

当 `n` 大时，每次 `B[k][j]` 都 miss 缓存，要從主存取，慢 100 倍。

## 2. 缓存层次结构回顾

现代 CPU 的存储层级：

```
寄存器 ←→ L1 缓存 ←→ L2 缓存 ←→ L3 缓存 ←→ 主存
  <1ns      ~1ns       ~4ns        ~12ns       ~100ns
  几十个    32-64KB    256KB-1MB   几-几十MB    GB级
```

**关键事实**：
1. 数据从主存加载到缓存是按**缓存行**（通常 64 字节）批量加载的
2. 一旦数据在缓存里，后续访问极快
3. 缓存很小（L1 只有 32KB），大矩阵放不下

### 2.1 缓存行的影响

访问 `A[0]` 时，`A[0]` 到 `A[15]`（假设 `int` 4 字节，一行 64 字节）被一起加载到缓存。后续访问 `A[1]`、`A[2]`、...、`A[15]` 都命中缓存，几乎免费。

**这就是数组的杀手锏：连续存放让缓存行批量加载发挥最大效用。**

### 2.2 矩阵的存储顺序

矩阵在内存里是线性存放的，有两种顺序：

- **行优先**（C/numpy 默认）：`A[i][j]` 和 `A[i][j+1]` 相邻
- **列优先**（Fortran/MATLAB）：`A[i][j]` 和 `A[i+1][j]` 相邻

numpy 默认行优先。所以 `A[i][k]`（k 递增）连续，`B[k][j]`（k 递增，j 固定）不连续。

## 3. 朴素矩阵乘法的缓存分析

### 3.1 访问模式

对 `C[i][j] = sum_k A[i][k] * B[k][j]`：

| 矩阵 | 访问模式 | 缓存 |
|---|---|---|
| A | `A[i][0], A[i][1], ..., A[i][p-1]` | 连续，友好 |
| B | `B[0][j], B[1][j], ..., B[p-1][j]` | 跨行，不友好 |
| C | 写 `C[i][0], C[i][1], ..., C[i][n-1]` | 连续，友好 |

B 是瓶颈。每次访问 `B[k][j]` 要跨 `n * sizeof(float)` 字节，如果 `n` 大（如 1024），跨度 4KB，远超缓存行 64 字节，几乎每次 miss。

### 3.2 miss 次数估算

假设 L1 缓存 32KB，缓存行 64 字节，`float` 4 字节，矩阵 $n \times n$。

- A：每行 $4n$ 字节。如果 $4n < 32\text{KB}$（$n < 8192$），一行能放进 L1，内层循环 A 不 miss
- B：每次访问跨 $4n$ 字节。如果 $n = 1024$，跨 4KB，每次访问一个新缓存行。$p$ 次内层循环，$p$ 次 miss
- 总 miss：$m \times n \times p / (\text{缓存行能放的 float 数}) = O(n^3 / 16)$

这个 miss 次数和计算量同阶，意味着**几乎每次算一个乘法都要等主存**，效率极低。

## 4. 分块矩阵乘法的原理

### 4.1 核心思想

把矩阵分成小块，让**三个小块同时放进缓存**：

```
A = [ A11 A12 ]   B = [ B11 B12 ]   C = [ C11 C12 ]
    [ A21 A22 ]       [ B21 B22 ]       [ C21 C22 ]

C11 = A11 B11 + A12 B21
C12 = A11 B12 + A12 B22
...
```

每个小块的乘法 `Aij Bjk` 三个小块都在缓存里，几乎不 miss。

### 4.2 分块后的代码

```python
def matmul_tiled(A, B, block=32):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(0, n, block):
        for j in range(0, n, block):
            for k in range(0, n, block):
                C[i:i+block, j:j+block] += (
                    A[i:i+block, k:k+block] @ B[k:k+block, j:j+block]
                )
    return C
```

- 外层三重循环遍历所有小块
- 内层 `A[i:i+block, k:k+block] @ B[k:k+block, j:j+block]` 是小块乘法
- 小块用 numpy `@` 算，享受向量化

### 4.3 分块的缓存分析

设分块大小 $b$，三个小块共 $3 b^2 \times 4$ 字节。要让它们放进 L1（32KB）：

$$
3 b^2 \times 4 < 32768 \implies b < \sqrt{32768 / 12} \approx 52
$$

所以 $b = 32$ 是常见选择（$3 \times 32^2 \times 4 = 12288$ 字节，留有余量）。

分块后：
- 每个小块乘法 $O(b^3)$，几乎不 miss
- 共 $(n/b)^3$ 个小块
- 总 miss：每个小块加载时 miss $O(b^2)$，共 $(n/b)^3 \times b^2 = n^3 / b$ 次

对比朴素 $O(n^3 / 16)$，分块降到 $O(n^3 / b)$，miss 减少 $b/16$ 倍。$b = 32$ 时减少 2 倍，$b = 64$ 时减少 4 倍。

## 5. 分块大小的选择

### 5.1 理论最优

分块大小 $b$ 的选择是缓存容量和循环开销的权衡：

- $b$ 太小：小块放不满缓存，且 Python/C 循环开销占比大
- $b$ 太大：三个小块放不进缓存，又 miss
- 最优：$3 b^2 \times \text{sizeof(elem)} \approx \text{L1 容量}$

对 L1 = 32KB，float 4 字节：$b \approx 52$，取 32 或 64。

### 5.2 实测：分块大小扫描

本 demo 的实测结果（n=512）：

| 分块大小 | 耗时 (ms) |
|---|---|
| 8 | 1345 |
| 16 | 180 |
| 32 | 48 |
| 64 | 23 |
| 128 | 21 |
| 256 | 9 |

**解读**：
- `b=8`：太小，Python 循环开销巨大（$(512/8)^3 = 262144$ 个小块，每个都走 Python 循环）
- `b=32`：开始有效，但 Python 循环仍多
- `b=64`：接近最优，Python 循环开销和缓存效果平衡
- `b=256`：几乎不分块，但 numpy `@` 内部已经做了分块，所以快
- `b=512`（不分块）：就是 `A @ B`，numpy 内部 BLAS 最优

### 5.3 Python 层面的特殊性

在 Python 层面，分块矩阵乘法的缓存优势被 **Python 循环开销**掩盖了：

- 每个小块要走 Python 的 `for` 循环，开销 ~100ns/次
- 小块越多，Python 循环越多，开销越大
- 所以在 Python 里，**块越大越快**（直到等于矩阵大小，即不分块）

真正的缓存效果在 C/BLAS 层面才体现。numpy 的 `@` 内部就是 BLAS，已经做了最优分块。

**所以本 demo 的真正教学价值是**：
1. 展示从纯 Python → numpy 向量化的巨大提升（解释器开销）
2. 展示分块大小对性能的影响
3. 展示 BLAS 的最优实现（内部已分块 + SIMD + 多线程）

## 6. demo 结果解读

### 6.1 正确性验证

四种实现（纯 Python、逐行、分块、BLAS）结果一致，max error < 1e-14，验证正确性。

### 6.2 性能对比（n=512）

| 实现 | 耗时 (ms) | 加速比 |
|---|---|---|
| row_wise | 28 | 1.00x |
| tiled_b32 | 48 | 0.59x |
| tiled_b64 | 19 | 1.50x |
| blas(@) | 3.4 | 8.24x |

**解读**：
- `row_wise`：逐行 `A[i] @ B`，每行用 numpy dot，但 B 整个放不进缓存
- `tiled_b32`：块太小，Python 循环开销导致比 row_wise 还慢
- `tiled_b64`：块够大，比 row_wise 快 1.5x
- `blas(@)`：numpy 内部 BLAS，SIMD + 多线程 + 最优分块，快 8x

### 6.3 纯 Python 的慢

纯 Python n=80：36ms。n=80 的 $80^3 = 512000$ 次乘法，每次约 70ns，这是 Python 解释器的开销（字节码解释 + 动态类型检查 + 引用计数）。

对比 numpy BLAS n=512：3.4ms。$512^3 = 1.3 \times 10^8$ 次乘法，每次约 0.026ns，快 2700 倍。这就是「连续内存 + SIMD + C 实现」的力量。

## 7. 生产级实现：BLAS

### 7.1 BLAS 是什么

BLAS（Basic Linear Algebra Subprograms）是矩阵运算的标准库，分三级：

- Level 1：向量-向量运算（点积、axpy）
- Level 2：矩阵-向量运算（矩阵乘向量）
- Level 3：矩阵-矩阵运算（矩阵乘法）

矩阵乘法是 Level 3，最复杂也最优化。

### 7.2 BLAS 的优化手段

1. **分块**：如上所述，让小块放进缓存
2. **SIMD**：一条指令同时算多个数据（AVX2 一条指令算 8 个 float）
3. **多线程**：OpenMP 并行化外层循环
4. **循环展开**：减少循环开销
5. **寄存器分配**：把热数据放寄存器
6. **数据预取**：提前加载数据到缓存

### 7.3 常见 BLAS 实现

- **OpenBLAS**：开源，常用
- **MKL**：Intel 提供，Intel CPU 上最快
- **cuBLAS**：NVIDIA GPU 上的 BLAS
- **BLIS**：OpenBLAS 的后继

numpy 默认用 OpenBLAS。PyTorch 用 MKL（CPU）或 cuBLAS（GPU）。

### 7.4 为什么不自己写

自己写矩阵乘法很难超过 BLAS，因为：
1. SIMD 要写汇编或 intrinsics
2. 缓存大小要针对具体 CPU 调优
3. 多线程要处理 false sharing、负载均衡

所以实践中**直接调 BLAS**。本 demo 的分块实现是为了教学，不是生产用。

## 8. 张量存储与连续内存

### 8.1 张量是数组的推广

标量是 0 维数组，向量是 1 维，矩阵是 2 维，张量是 n 维。但**底层都是一块连续内存**。

```python
import numpy as np
T = np.zeros((2, 3, 4), dtype=np.float32)
# 底层是 2*3*4*4 = 96 字节的连续内存
```

### 8.2 strides：多维到一维的映射

numpy 用 `strides` 把多维下标映射到一维地址：

$$
\text{addr}(i, j, k) = \text{base} + i \times \text{stride}[0] + j \times \text{stride}[1] + k \times \text{stride}[2]
$$

对 `(2, 3, 4)` 的 float32 行优先数组，`strides = (48, 16, 4)`：
- 跨第一维 48 字节（3*4*4）
- 跨第二维 16 字节（4*4）
- 跨第三维 4 字节（4）

### 8.3 连续性对 GPU 的影响

GPU 要求内存连续才能高效并行：
- GPU 从全局内存加载数据到共享内存是按块加载的
- 不连续的访问会导致 uncoalesced memory access，性能下降 10 倍以上
- 所以 PyTorch 的 tensor `.contiguous()` 调用很常见

```python
x = torch.randn(3, 4)
y = x.T  # 转置，strides 变了，不连续
z = y.contiguous()  # 强制拷贝成连续
```

## 9. GPU 上的分块：tiling 在 CUDA 里的体现

GPU 矩阵乘法的核心也是分块，但分块到 GPU 的共享内存（shared memory）：

```cuda
// 简化的 CUDA 矩阵乘法分块
__shared__ float A_tile[BLOCK][BLOCK];
__shared__ float B_tile[BLOCK][BLOCK];

for (int k = 0; k < n; k += BLOCK) {
    // 把 A 和 B 的小块加载到共享内存
    A_tile[ty][tx] = A[i * n + k + tx];
    B_tile[ty][tx] = B[(k + ty) * n + j];
    __syncthreads();

    // 在共享内存里算小块乘法
    for (int kk = 0; kk < BLOCK; kk++) {
        acc += A_tile[ty][kk] * B_tile[kk][tx];
    }
    __syncthreads();
}
```

- 共享内存类似 L1 缓存，但程序员显式控制
- 把小块加载到共享内存后，后续访问极快
- 这就是「分块」在 GPU 上的体现

## 10. 与其他数据结构的关联

- **数组**（本章）：矩阵乘法的底层
- **堆**（第 06 章）：Top-K 采样要从 logits 里选最大的 K 个
- **哈希表**（第 05 章）：embedding 查找要 O(1) 定位词向量
- **页表**（第 11 章）：PagedAttention 把 KV Cache 分块管理，本质是数组的分页

## 11. 小结

分块矩阵乘法的核心：
1. **矩阵乘法是 AI 的心脏**，90%+ 时间在这
2. **朴素实现缓存不友好**，B 矩阵按列访问导致 miss
3. **分块让三个小块同时进缓存**，miss 大幅减少
4. **BLAS 进一步优化**：SIMD + 多线程 + 寄存器分配
5. **GPU 上同样分块**，但分到共享内存

数组的连续存放是这一切的基础——没有连续，就没有缓存友好，就没有 SIMD，就没有 GPU 并行。这就是为什么数组是 AI 里最核心的数据结构。

下一篇：第 02 章 链表，讲 LRU 缓存在 KV Cache 淘汰里的应用。