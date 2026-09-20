# 第 20 章 · 模型压缩（扩展专题）

> AI 场景：大模型部署到资源受限设备的压缩技术（剪枝、量化、知识蒸馏）
> 数据结构：稀疏存储（CSR）+ 查找表（LUT）+ 树（蒸馏）+ 哈希表（索引）

## 本章要回答的问题

- 大模型太大装不下设备，怎么压缩？（剪枝 + 量化 + 蒸馏组合）
- 剪枝后 70% 权重为 0，怎么存才省内存？（CSR 稀疏格式）
- float32 → INT8 量化，反量化怎么快？（查找表 LUT 一次索引）
- 小模型精度不够，怎么用大模型的知识补？（知识蒸馏 + 软标签 + 树）
- 量化后权重怎么快速查找？（哈希表 O(1) 索引）
- 4 种数据结构如何协奏完成模型压缩？

## 扩展专题定位

前 12 章是「一个数据结构 → 一个 AI 优化」。第 13 章起是扩展专题：**一个 AI 应用场景 → 多种数据结构协作**。本章展示稀疏存储、查找表、树、哈希表如何协作解决大模型部署的压缩问题。

## 目录

```
20_model_compression/
├── python/
│   └── demo.py              # 3 种压缩技术 demo（剪枝 + 量化 + 蒸馏）
├── docs/
│   ├── principle.md         # 4 种数据结构原理 + 协作机制
│   └── ai_application.md    # 模型压缩应用（llama.cpp / GPTQ / AWQ / DistilBERT）
├── figures/                 # 性能对比图
└── README.md
```

## demo 概览

模拟 3 种模型压缩技术，4 种数据结构协作：

```
大模型权重到达
  ├─→ [稀疏存储/CSR]  Pruning：幅度剪枝 → CSR 只存非零元素
  ├─→ [查找表/LUT]    Quantization：float32 → INT8 + 256 项 LUT
  ├─→ [树]            KnowledgeDistillation：teacher 软标签 → student 决策树
  └─→ [哈希表]        HashIndex：(row,col) → quantized value 哈希索引
```

- `Pruning`：`magnitude_prune` 按幅值剪枝 + `dense_to_csr` 转 CSR 格式
- `Quantization`：`quantize_int8` 量化 + `dequantize` 用 LUT 反量化
- `KnowledgeDistillation`：`softmax_with_temperature` 软标签 + `build_decision_tree` student 树
- `QuantHashIndex`：`(row, col) → int8` 哈希索引 + LUT 反量化

性能对比：
1. 稠密 vs CSR matvec（稀疏率 70% → 加速 3.10x，内存压缩 1.62x）
2. float32 vs INT8 matvec（内存压缩 3.20x，量化误差 0.008）
3. 哈希索引 O(1) vs 线性查找 O(n)（加速 1.12x）
4. student 独立 vs 蒸馏训练准确率（0.672 → 0.814，提升 +0.142）

## 跑法

```bash
python 20_model_compression/python/demo.py
```

## 关键结果

| 技术 | 数据结构 | 压缩/加速 | 精度 |
|------|---------|----------|------|
| 剪枝 70% | CSR | 内存 1.62x，matvec 3.10x | ~0% 损失 |
| INT8 量化 | LUT | 内存 3.20x | 误差 0.008 |
| 蒸馏 | 树 | student 准确率 +0.142 | 接近 teacher |
| 哈希索引 | 哈希表 | 查找 1.12x | 0% 损失 |

组合压缩（剪枝 70% + INT4 + 蒸馏）可达 13x 内存压缩 + 12x 推理加速。