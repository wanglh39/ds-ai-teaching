# 第 20 章 · 模型压缩 — AI 应用

> AI 场景：大模型部署到资源受限设备的压缩技术。
> 数据结构：稀疏存储（CSR）+ 查找表（LUT）+ 树（蒸馏）+ 哈希表（索引）协作。

---

## 1. 大模型部署的挑战

### 1.1 问题：模型太大，设备太小

现代大模型参数量爆炸增长：

```
GPT-2 (2019)    : 1.5B 参数  → 6 GB（float32）
GPT-3 (2020)    : 175B 参数 → 700 GB（float32）
LLaMA-2 (2023)  : 70B 参数  → 280 GB（float32）
LLaMA-3 (2024)  : 405B 参数 → 1620 GB（float32）
```

但部署设备的资源极其有限：

| 设备 | 内存 | 算力 | 典型场景 |
|------|------|------|---------|
| 手机 | 4~12 GB | 1~5 TFLOPS | 移动助手 |
| 边缘盒子 | 1~4 GB | 0.5~2 TFLOPS | 智能家居 |
| 树莓派 | 0.5~8 GB | 0.05 TFLOPS | IoT |
| 智能手表 | 0.5~1 GB | 0.01 TFLOPS | 可穿戴 |

**矛盾**：模型 700 GB，手机只有 8 GB。即使能装下，推理延迟也会到分钟级，不可用。

### 1.2 模型压缩的四大技术

| 技术 | 压缩比 | 精度损失 | 数据结构 | 适用场景 |
|------|--------|---------|---------|---------|
| 剪枝 | 5~10x | < 1% | 稀疏存储 | 通用 |
| 量化 | 4~16x | 1~3% | 查找表 | 通用 |
| 蒸馏 | 10~100x | 2~5% | 树 | 有 teacher |
| 哈希索引 | 1~2x | 0% | 哈希表 | 稀疏查找 |

组合使用可达 **32x~100x 压缩**，让 70B 模型在手机上运行。

### 1.3 为什么需要多种数据结构协作

单一技术难以同时满足「高压缩比 + 低精度损失 + 快推理」：

- 只剪枝：稀疏率高时精度损失大，且稀疏矩阵硬件支持有限
- 只量化：INT4 以下精度损失剧增，且需要 LUT 恢复
- 只蒸馏：student 容量太小难以达到 teacher 精度
- 只哈希：不压缩，只加速查找

**组合**：剪枝去冗余 + 量化降精度 + 蒸馏补精度 + 哈希加速访问 → 四种数据结构各司其职，协同达到最优。

---

## 2. 剪枝原理 + 稀疏存储

### 2.1 幅度剪枝

**核心观察**：神经网络权重幅值分布近似正态，大部分权重接近 0，对输出贡献微弱。

```python
def magnitude_prune(W, sparsity):
    threshold = np.quantile(np.abs(W), sparsity)
    W[np.abs(W) < threshold] = 0.0   # 小幅值置零
    return W
```

**为什么有效**：小幅值权重的贡献 `w × x` 很小，置零后输出变化小，精度损失可控。

**Lottery Ticket Hypothesis**（2018）：剪枝后的网络包含一个「中奖彩票」——从初始权重继承的子网络，训练后能达到原网络精度。这说明剪枝不是「丢掉」，而是「找出本质子网络」。

### 2.2 结构化剪枝

非结构化剪枝的稀疏模式不规则，硬件难以加速。结构化剪枝按「通道/头/层」整块剪：

| 粒度 | 单位 | 硬件支持 | 典型方法 |
|------|------|---------|---------|
| 细粒度 | 单个权重 | 需稀疏库 | Magnitude Pruning |
| 通道级 | 整个通道 | 直接跳过 | Channel Pruning |
| 头级 | 整个注意力头 | 直接跳过 | Head Pruning |
| 层级 | 整个 Transformer 层 | 直接跳过 | Layer Dropout |

**通道剪枝**：

```python
def channel_prune(W, sparsity):
    # 按通道 L2 范数排序
    norms = np.sqrt(np.sum(W**2, axis=1))
    threshold = np.quantile(norms, sparsity)
    W[norms < threshold, :] = 0   # 整行置零
    return W
```

结构化剪枝后矩阵出现整零行，硬件可以直接跳过这些通道，无需稀疏库。

### 2.3 稀疏存储格式选择

剪枝后选择存储格式的决策树：

```
稀疏率 < 50%?
  ├─ 是 → 稠密存储（稀疏存储反而更大）
  └─ 否 → 需要行切片?
           ├─ 是 → CSR（行压缩）
           └─ 否 → 需要列切片?
                    ├─ 是 → CSC（列压缩）
                    └─ 否 → 动态修改?
                             ├─ 是 → 哈希表
                             └─ 否 → CSR（默认）
```

神经网络推理主要是矩阵 × 向量（行优先），**CSR 是最常用格式**。

### 2.4 demo 结果解读

demo 模拟 64×64 权重矩阵，剪枝到稀疏率 70%：

```
稠密 matvec: 1.20 ms（遍历 4096 元素）
CSR  matvec: 0.39 ms（只遍历 1229 非零元素）→ 加速 3.10x

内存：稠密 16384 bytes → CSR 10092 bytes（压缩 1.62x）
```

**注意**：CSR 内存压缩比（1.62x）低于稀疏率倒数（3.33x），因为 CSR 要额外存列索引和行指针。稀疏率越高，额外开销占比越小，压缩比越接近稀疏率倒数。

---

## 3. 量化原理 + 查找表

### 3.1 对称量化 vs 非对称量化

**对称量化**（demo 使用）：权重范围关于 0 对称

```
scale = max(|W|) / 127
q     = round(W / scale) + 128      # 映射到 0~255
W'    = (q - 128) * scale
```

**非对称量化**：权重范围不对称

```
scale = (max(W) - min(W)) / 255
zero_point = round(-min(W) / scale)
q     = round(W / scale) + zero_point
W'    = (q - zero_point) * scale
```

非对称量化精度更高（充分利用 256 个量化级），但计算稍复杂。

### 3.2 查找表的本质

量化的反量化 `W' = (q - zero_point) * scale` 看起来是乘法，但 `q` 只有 256 个可能值。**预计算**所有结果存成 LUT：

```python
LUT = np.array([(q - zero_point) * scale for q in range(256)])
W'  = LUT[q]   # 一次数组索引，比乘法快
```

**LUT 的真正威力**在于非均匀量化：

```python
# K-means 量化：权重聚成 256 类，LUT 存聚类中心
centers = KMeans(n_clusters=256).fit(W).cluster_centers_
LUT = centers   # 非均匀中心，精度比线性量化高
```

GPTQ、AWQ 等大模型量化算法本质上都是构造更优的 LUT。

### 3.3 GPTQ：基于二阶信息的量化

GPTQ（Generalized Post-Training Quantization）用 Hessian 矩阵的二阶信息指导量化：

```
1. 计算权重 W 的 Hessian H = X^T X（X 是输入）
2. 逐列量化，每列量化时补偿前面列的误差
3. 用 Hessian 加权最小化量化误差
```

GPTQ 能把 LLaMA-65B 量化到 INT4，精度损失仅 1~2%，推理加速 4x。本质是构造一个**自适应 LUT**，让量化误差在重要方向上最小。

### 3.4 AWQ：激活感知的量化

AWQ（Activation-aware Weight Quantization）观察激活值分布，对「重要权重」用更细的量化级：

```
1. 校准数据跑一遍，记录每层激活值
2. 激活值大的通道 → 权重重要 → 用更细量化
3. 激活值小的通道 → 权重不重要 → 用更粗量化
```

AWQ 不量化激活，只量化权重，但根据激活分布调整权重的 LUT。比 GPTQ 更简单，精度相当。

### 3.5 demo 结果解读

demo 模拟 64×64 权重矩阵，INT8 对称量化：

```
float32 matvec: 1.01 ms
INT8   matvec: 1.12 ms（含 LUT 反量化）→ 略慢

内存：float32 16384 bytes → INT8+LUT 5120 bytes（压缩 3.20x）
量化误差：平均 0.0077，最大 0.0155
```

**注意**：INT8 matvec 比 float32 略慢，因为 demo 用 Python 模拟，LUT 反量化引入额外开销。**真实推理引擎**（如 llama.cpp）用 INT8 SIMD 指令（如 AVX2、NEON）直接做整数矩阵乘法，比 float32 快 2~4x。量化的速度收益来自硬件整数指令，不是算法本身。

---

## 4. 知识蒸馏原理 + 树结构

### 4.1 蒸馏的数学原理

student 的损失函数 = 硬标签损失 + 软标签损失：

```
L = α × L_hard(student, y_true) + (1-α) × T² × L_soft(student, teacher_soft)
```

- `L_hard`：student 预测 vs 真实硬标签的交叉熵
- `L_soft`：student 预测 vs teacher 软标签的 KL 散度
- `T²`：温度平方补偿（高温 softmax 梯度小，乘 T² 补偿）
- `α`：硬/软标签权重，典型 α=0.3

### 4.2 软标签的「暗知识」

teacher 的软标签包含硬标签没有的信息：

```
输入: 一张猫的图片
硬标签: [1, 0, 0, 0, 0]                    # 只是猫
软标签: [0.29, 0.22, 0.20, 0.14, 0.15]     # 猫，但像狗多于像汽车
```

软标签的「暗知识」（dark knowledge）：
- **类间关系**：猫像狗多于像汽车（类 1 > 类 4）
- **类内难度**：这个样本比较难（概率分散，不是 0.99 vs 0.01）
- **决策边界**：teacher 在边界附近的软标签更平滑，student 学到更鲁棒的边界

### 4.3 温度参数的影响

温度 T 控制软标签的熵：

| T | 熵 (bits) | 软标签形态 | 暗知识 |
|---|----------|-----------|--------|
| 1 | 1.42 | 接近 one-hot | 少 |
| 2 | 1.95 | 中等平滑 | 中 |
| 4 | 2.21 | 平滑 | 多 |
| 8 | 2.27 | 接近均匀 | 最多 |
| 16 | 2.32 | 几乎均匀 | 过多（噪声） |

demo 测得 T=4 时软标签熵 2.21 bits，是 T=1（1.42 bits）的 1.5 倍。**T=4~8 是常用范围**，暗知识丰富但不至于变成噪声。

### 4.4 决策树作为 student

demo 用决策树作为 student 模型，模拟蒸馏流程：

```
1. 生成 5 个高斯团（500 样本，8 特征）
2. 注入 25% 标签噪声
3. teacher：深树（depth=6）+ 全特征 → 准确率 0.950（去噪成功）
4. student 独立：浅树（depth=2）+ 3 特征 → 准确率 0.672（受噪声影响）
5. teacher 输出软标签（T=4）
6. student 蒸馏：浅树（depth=2）+ 4 特征 + teacher 预测标签 → 准确率 0.814
```

**蒸馏提升 +0.142**，来自两个因素：
1. teacher 预测标签去噪了（0.950 vs 原始 0.75）
2. student 蒸馏后多用 1 个特征（模拟蒸馏传递了 teacher 的特征表示知识）

### 4.5 真实蒸馏案例

| Teacher | Student | 压缩比 | 精度变化 | 应用 |
|---------|---------|--------|---------|------|
| BERT-Large | DistilBERT | 1.9x | -1% | 文本理解 |
| LLaMA-70B | TinyLlama-1.1B | 64x | -5% | 移动助手 |
| ResNet-152 | ResNet-18 | 8.5x | -2% | 图像分类 |
| GPT-3 175B | GPT-3 6.7B | 26x | -3% | 通用对话 |

DistilBERT 是经典案例：student 用 6 层（teacher 12 层），参数减少 40%，推理快 60%，精度保持 95%+。

---

## 5. 哈希表在量化中的作用

### 5.1 量化权重的索引需求

量化后权重是 int8 数组，推理时需要查找特定位置的权重：

```python
# 稠密数组直接索引（O(1)）
W = np.uint8 array
value = W[i, j]

# 稀疏 + 量化 → 哈希表索引
hash_table = {(i, j): quantized_value for non-zero positions}
value = hash_table[(i, j)]   # O(1) 哈希查找
```

哈希表在**稀疏 + 量化**场景的优势：
- 只存非零位置 → 内存 = `nnz × 9` bytes（键 8 + 值 1）
- 动态修改友好 → 增删非零元素只需 `dict[key] = value`
- 任意键类型 → 支持 `(layer, row, col)` 多维索引

### 5.2 demo 结果解读

demo 模拟 32×32 量化矩阵，100 次查询：

```
哈希索引 O(1): 0.038 ms
线性查找 O(n): 0.043 ms → 加速 1.12x
```

加速比不大是因为矩阵小（32×32），哈希函数开销与线性查找差距小。**大矩阵**（如 4096×4096）时哈希表优势显著，因为线性查找 O(n) 随 n 增长，哈希表始终 O(1)。

### 5.3 哈希表 vs 数组的取舍

| 场景 | 推荐 | 原因 |
|------|------|------|
| 稠密矩阵 | 数组 | 直接索引，缓存友好 |
| 稀疏 + 静态 | CSR 数组 | 紧凑 + 行切片高效 |
| 稀疏 + 动态 | 哈希表 | 增删 O(1) |
| 多维稀疏 | 哈希表 | 键灵活 |

---

## 6. demo 结果解读

### 6.1 性能对比汇总

```
(a) 剪枝：稠密 vs CSR matvec（64×64, 稀疏率 70%）
    稠密 1.20 ms → CSR 0.39 ms（加速 3.10x）
    内存 16384 → 10092 bytes（压缩 1.62x）

(b) 量化：float32 vs INT8 matvec（64×64）
    float32 1.01 ms → INT8 1.12 ms（Python 模拟略慢）
    内存 16384 → 5120 bytes（压缩 3.20x）
    量化误差 0.0077

(c) 哈希索引：O(1) vs O(n)（32×32, 100 查询）
    哈希 0.038 ms → 线性 0.043 ms（加速 1.12x）

(d) 蒸馏：student 独立 vs 蒸馏（500 样本, 5 类, 25% 噪声）
    teacher 0.950 → student 独立 0.672 → student 蒸馏 0.814
    蒸馏提升 +0.142
```

### 6.2 关键发现

1. **CSR 加速与稀疏率成正比**：稀疏率 70% → 加速 3.1x，接近理论值 3.3x
2. **量化内存收益显著**：INT8 压缩 3.2x，是模型压缩的主力
3. **蒸馏能补精度**：student 准确率从 0.672 提升到 0.814，接近 teacher 的 0.950
4. **哈希索引适合稀疏**：小矩阵优势不明显，大矩阵 + 稀疏时 O(1) 优势显著

### 6.3 组合压缩的理论收益

```
剪枝 70% + INT4 量化 + 蒸馏：
  内存 = 原始 × 0.3（剪枝）× 0.25（INT4）= 7.5% → 压缩 13x
  精度 = teacher 精度 - 蒸馏损失 - 量化损失 ≈ teacher - 3%
  速度 = 稀疏加速 3x × INT4 硬件加速 4x = 12x
```

这是 llama.cpp、GPTQ、AWQ 等工具能达到的典型压缩比。

---

## 7. 生产级实现

### 7.1 llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) 是 C++ 实现的 LLaMA 推理引擎，核心是量化 + 查找表：

- **INT4/INT8 量化**：权重存 int4/int8，LUT 反量化
- **SIMD 指令**：用 AVX2/NEON 指令做整数矩阵乘法，比 float32 快 4x
- **内存映射**：模型文件直接 mmap，不加载到内存
- **CPU 推理**：不需要 GPU，能在笔记本/手机上跑 70B 模型

```bash
# 量化 LLaMA 模型到 INT4
./quantize ./models/llama-7b.bin ./models/llama-7b-q4.bin 2

# 推理
./main -m ./models/llama-7b-q4.bin -p "你好"
```

### 7.2 GPTQ

[GPTQ](https://github.com/IST-DASLab/gptq) 是基于二阶信息的后训练量化：

- **Hessian 加权**：用输入数据的 Hessian 矩阵指导量化误差分配
- **逐列量化 + 误差补偿**：每列量化后补偿前面列的误差
- **INT4 量化**：LLaMA-65B 量化到 INT4，精度损失 1~2%

```python
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("TheBloke/LLaMA2-7B-GPTQ")
```

### 7.3 AWQ

[AWQ](https://github.com/mit-han-lab/llm-awq) 是激活感知的量化：

- **激活校准**：用校准数据跑一遍，记录激活分布
- **重要通道保护**：激活大的通道用更细量化级
- **等价于自适应 LUT**：不同通道用不同 LUT

```python
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("TheBloke/LLaMA2-7B-AWQ")
```

### 7.4 BitsAndBytes

[BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 是 HuggingFace 默认量化库：

- **NF4 量化**：Normal Float 4-bit，针对正态分布权重优化的 LUT
- **双量化**：权重 + 量化常数都量化，进一步省内存
- **QLoRA**：NF4 量化 + LoRA 微调，4bit 推理 + 低秩微调

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_4bit=True,   # NF4 量化
    device_map="auto",
)
```

### 7.5 稀疏推理：SparseGPT

[SparseGPT](https://github.com/IST-DASLab/sparsegpt) 结合剪枝 + 量化：

- **结构化剪枝**：按通道剪枝，稀疏率 50%+
- **GPTQ 量化**：剪枝后 INT4 量化
- **组合压缩**：50% 稀疏 × INT4 = 压缩 8x，精度损失 < 3%

```python
# SparseGPT + GPTQ 组合
model = sparse_gpt_prune(model, sparsity=0.5)   # 剪枝 50%
model = gptq_quantize(model, bits=4)             # INT4 量化
# 总压缩 8x，精度损失 ~3%
```

### 7.6 蒸馏：DistilBERT

[DistilBERT](https://huggingface.co/distilbert-base-uncased) 是 BERT 蒸馏的经典案例：

- **Teacher**：BERT-Base（12 层，110M 参数）
- **Student**：DistilBERT（6 层，66M 参数）
- **蒸馏损失**：MLM 损失 + 蒸馏 KL 损失 + 余弦相似度损失
- **结果**：参数减少 40%，推理快 60%，精度保持 95%+

```python
from transformers import DistilBertForMaskedLM
student = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
# 6 层 student，达到 12 层 teacher 95% 的精度
```

---

## 8. 小结

模型压缩是「大模型 → 小设备」的关键技术，四种数据结构各司其职：

1. **稀疏存储（CSR）**：剪枝后权重的紧凑表示，省内存 + 加速稀疏矩阵乘法
2. **查找表（LUT）**：量化的本质是 LUT 映射，INT8/INT4 + LUT 反量化
3. **树**：知识蒸馏的 student 模型，teacher 软标签 → student，去噪 + 知识传递
4. **哈希表**：量化权重的 O(1) 索引，稀疏 + 动态修改友好

生产级工具（llama.cpp、GPTQ、AWQ、BitsAndBytes）都是这些数据结构的工程化实现：
- llama.cpp = INT4 量化 + LUT + SIMD
- GPTQ = Hessian 加权 LUT 构造
- AWQ = 激活感知 LUT
- BitsAndBytes = NF4 LUT + 双量化
- DistilBERT = teacher 树 → student 树

组合使用可达 32x~100x 压缩，让 70B 大模型在手机上运行。这是「数据结构 × AI」的典型协作场景。