"""第 20 章 demo：模型压缩 × 数据结构协作（稀疏存储 + 查找表 + 树 + 哈希表）

扩展专题：一个 AI 应用场景（大模型部署到资源受限设备的压缩技术）
→ 多种数据结构协作
- Pruning             : 剪枝 + 稀疏存储（CSR 格式存稀疏权重）
- Quantization        : 量化 + 查找表（INT8 量化 + LUT 反量化）
- KnowledgeDistillation: 知识蒸馏 + 树（teacher→student 软标签传递）
- HashIndex           : 量化权重哈希索引（O(1) 查找非零权重）

模拟 3 种模型压缩技术，4 种数据结构协作：
  大模型权重
    → [稀疏存储] 剪枝：CSR 只存非零元素，省内存 + 稀疏矩阵乘法
    → [查找表]   量化：float32 → int8，256 项 LUT 反量化
    → [树]       蒸馏：teacher 软标签 → student，决策树/softmax 传递
    → [哈希表]   索引：量化后权重哈希索引，O(1) 查找

性能对比：
1. 稠密 vs CSR 内存 + 矩阵乘法速度
2. float32 vs int8 内存 + 推理速度
3. student 独立训练 vs 蒸馏训练准确率

跑法：
    python 20_model_compression/python/demo.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from py_utils import Timer, compare, format_table, save_bar, save_line  # noqa: E402


# ============================================================================
# 1. Pruning：剪枝 + 稀疏存储（CSR 格式）
# ============================================================================


@dataclass
class CSRMatrix:
    """CSR（Compressed Sparse Row）稀疏矩阵。

    AI 场景：剪枝后权重矩阵 70%~90% 元素为 0，稠密存储浪费内存。
    CSR 只存非零元素 + 行指针 + 列索引，大幅省内存 + 加速稀疏矩阵乘法。

    三个核心数组：
    - data    : 非零元素值（一维数组）
    - indices : 非零元素的列索引（与 data 一一对应）
    - indptr  : 行指针，indptr[i] ~ indptr[i+1] 是第 i 行的非零元素在 data 中的范围

    内存对比（n×n 矩阵，稀疏率 sparsity）：
    - 稠密：n² × 4 bytes（float32）
    - CSR ：(1-sparsity)×n² × (4+4) + (n+1)×4 bytes（data + indices + indptr）
    """

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        """非零元素个数。"""
        return len(self.data)

    @property
    def mem_bytes(self) -> int:
        """内存占用（data + indices + indptr）。"""
        return self.data.nbytes + self.indices.nbytes + self.indptr.nbytes

    def to_dense(self) -> np.ndarray:
        """CSR → 稠密矩阵（用于验证）。"""
        dense = np.zeros(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            for j in range(self.indptr[i], self.indptr[i + 1]):
                dense[i, self.indices[j]] = self.data[j]
        return dense

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """CSR 稀疏矩阵 × 向量（只遍历非零元素）。

        对比稠密 matvec：稠密遍历 n² 个元素，CSR 只遍历 nnz 个非零元素。
        稀疏率越高（nnz << n²），CSR 越快。
        """
        y = np.zeros(self.shape[0], dtype=np.float32)
        for i in range(self.shape[0]):
            s = 0.0
            for j in range(self.indptr[i], self.indptr[i + 1]):
                s += self.data[j] * x[self.indices[j]]
            y[i] = s
        return y


def magnitude_prune(weight: np.ndarray, sparsity: float) -> np.ndarray:
    """幅度剪枝：把绝对值最小的 sparsity 比例的权重置零。

    AI 场景：神经网络权重中很多小幅值对输出贡献小，置零后精度损失小。
    置零的权重不再存储 → 稀疏矩阵 → CSR 存储。
    """
    threshold = np.quantile(np.abs(weight), sparsity)
    pruned = weight.copy()
    pruned[np.abs(pruned) < threshold] = 0.0
    return pruned


def dense_to_csr(weight: np.ndarray) -> CSRMatrix:
    """稠密矩阵 → CSR 稀疏格式。

    遍历每行，收集非零元素的值和列索引，构造 indptr/indices/data。
    """
    rows, cols = weight.shape
    data_list: list[float] = []
    indices_list: list[int] = []
    indptr_list: list[int] = [0]

    for i in range(rows):
        for j in range(cols):
            v = weight[i, j]
            if v != 0.0:
                data_list.append(float(v))
                indices_list.append(j)
        indptr_list.append(len(data_list))

    return CSRMatrix(
        data=np.array(data_list, dtype=np.float32),
        indices=np.array(indices_list, dtype=np.int32),
        indptr=np.array(indptr_list, dtype=np.int32),
        shape=(rows, cols),
    )


def dense_matvec(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    """稠密矩阵 × 向量（遍历所有 n² 元素）。"""
    rows, cols = weight.shape
    y = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += weight[i, j] * x[j]
        y[i] = s
    return y


# ============================================================================
# 2. Quantization：量化 + 查找表（INT8）
# ============================================================================


@dataclass
class QuantizedTensor:
    """INT8 量化张量 + 查找表（LUT）。

    AI 场景：float32 权重 → int8（0~255），内存省 4x。
    反量化时用查找表（LUT）把 int8 索引映射回 float32 值。

    量化公式：q = round(w / scale) + zero_point
    反量化   ：w = (q - zero_point) * scale = LUT[q]

    查找表 LUT：256 项的 float32 数组，LUT[q] = (q - zero_point) * scale
    反量化只需 1 次数组索引，比乘法快。
    """

    data: np.ndarray          # int8 量化值（uint8 0~255）
    scale: float              # 量化缩放因子
    zero_point: int           # 量化零点
    lut: np.ndarray           # 查找表：256 项 float32，LUT[q] = 反量化值
    shape: tuple[int, ...]

    @property
    def mem_bytes(self) -> int:
        """内存占用（量化数据 + LUT）。"""
        return self.data.nbytes + self.lut.nbytes


def quantize_int8(weight: np.ndarray) -> QuantizedTensor:
    """float32 → INT8 量化（对称量化 + 查找表）。

    1. 找最大绝对值 w_max
    2. scale = w_max / 127
    3. q = round(w / scale) + 128  （映射到 0~255 uint8）
    4. zero_point = 128
    5. 构造 LUT：LUT[q] = (q - 128) * scale
    """
    w_max = float(np.max(np.abs(weight)))
    if w_max == 0.0:
        w_max = 1e-8
    scale = w_max / 127.0
    zero_point = 128
    q = np.round(weight / scale).astype(np.int32) + zero_point
    q = np.clip(q, 0, 255).astype(np.uint8)
    # 查找表：256 项
    lut = (np.arange(256, dtype=np.float32) - zero_point) * scale
    return QuantizedTensor(
        data=q,
        scale=scale,
        zero_point=zero_point,
        lut=lut,
        shape=weight.shape,
    )


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """用查找表反量化：w = LUT[q]。

    反量化只需数组索引，比逐元素乘法快。
    """
    return qt.lut[qt.data]


def quant_matvec(qt: QuantizedTensor, x: np.ndarray) -> np.ndarray:
    """INT8 量化矩阵 × 向量（用 LUT 反量化 + 矩阵乘法）。

    实际推理引擎（如 llama.cpp）会用 INT8 SIMD 指令直接做整数矩阵乘法，
    这里用 LUT 反量化后做 float 乘法模拟流程。
    """
    w = qt.lut[qt.data]  # LUT 反量化
    rows, cols = w.shape
    y = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += w[i, j] * x[j]
        y[i] = s
    return y


def float32_matvec(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    """float32 矩阵 × 向量（基准）。"""
    rows, cols = weight.shape
    y = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += weight[i, j] * x[j]
        y[i] = s
    return y


# ============================================================================
# 3. KnowledgeDistillation：知识蒸馏 + 树（teacher→student）
# ============================================================================


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """带温度的 softmax（知识蒸馏的核心）。

    AI 场景：teacher 模型输出 logits，用高温 softmax 得到「软标签」。
    软标签包含类间关系信息（如 "猫" vs "狗" 比 "猫" vs "汽车" 更近），
    student 学习软标签比学硬标签信息量更大。

    温度 T 越高，分布越平滑（软）；T→1 退化为标准 softmax（硬）。
    """
    t_logits = logits / max(temperature, 1e-8)
    t_logits = t_logits - np.max(t_logits)  # 数值稳定
    exp = np.exp(t_logits)
    return exp / np.sum(exp)


@dataclass
class TreeNode:
    """决策树节点（student 模型学到的决策边界）。

    AI 场景：student 模型可以是小决策树，蒸馏时学习 teacher 的软标签。
    树结构天然表达「if-else 决策边界」。
    """

    feature: int = -1          # 分裂特征索引（-1 = 叶节点）
    threshold: float = 0.0     # 分裂阈值
    left: TreeNode | None = None
    right: TreeNode | None = None
    label: int = 0             # 叶节点的类别标签


def build_decision_tree(features: np.ndarray, labels: np.ndarray, depth: int = 0,
                        max_depth: int = 3) -> TreeNode:
    """构建决策树（递归分裂）。

    简化版：每个节点选一个特征 + 阈值分裂，最大化信息增益。
    """
    if depth >= max_depth or len(labels) <= 1 or len(set(labels.tolist())) == 1:
        # 叶节点：多数投票
        return TreeNode(label=int(np.bincount(labels).argmax()))

    n_features = features.shape[1]
    best_gain = -1.0
    best_feat = 0
    best_thresh = 0.0
    best_left_mask = None

    parent_entropy = _entropy(labels)

    for f in range(n_features):
        col = features[:, f]
        thresh = float(np.median(col))
        left_mask = col <= thresh
        if left_mask.all() or not left_mask.any():
            continue
        left_labels = labels[left_mask]
        right_labels = labels[~left_mask]
        n = len(labels)
        gain = parent_entropy - (
            len(left_labels) / n * _entropy(left_labels)
            + len(right_labels) / n * _entropy(right_labels)
        )
        if gain > best_gain:
            best_gain = gain
            best_feat = f
            best_thresh = thresh
            best_left_mask = left_mask

    if best_left_mask is None or best_gain <= 0:
        return TreeNode(label=int(np.bincount(labels).argmax()))

    node = TreeNode(feature=best_feat, threshold=best_thresh)
    node.left = build_decision_tree(features[best_left_mask], labels[best_left_mask],
                                    depth + 1, max_depth)
    node.right = build_decision_tree(features[~best_left_mask], labels[~best_left_mask],
                                     depth + 1, max_depth)
    return node


def _entropy(labels: np.ndarray) -> float:
    """信息熵。"""
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels)
    probs = counts[counts > 0] / len(labels)
    return float(-np.sum(probs * np.log2(probs)))


def tree_predict(node: TreeNode, x: np.ndarray) -> int:
    """决策树预测单个样本。"""
    while node.feature >= 0:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.label


def tree_predict_batch(root: TreeNode, X: np.ndarray) -> np.ndarray:
    """批量预测。"""
    return np.array([tree_predict(root, x) for x in X])


def simulate_distillation(n_samples: int = 200, n_classes: int = 5,
                          n_features: int = 8, temperature: float = 4.0,
                          seed: int = 42, noise_ratio: float = 0.25) -> dict[str, float]:
    """模拟知识蒸馏：teacher → student。

    蒸馏的核心价值：teacher 能学到「去噪后的决策边界」，软标签把这种
    「更干净的知识」传递给 student。student 学软标签比学带噪声的硬标签
    更鲁棒，准确率更高。

    流程：
    1. 生成结构化数据（n_classes 个高斯团，每个类有明确中心）
    2. 注入标签噪声（翻转 noise_ratio 比例的标签）→ 带噪声训练标签
    3. teacher（大模型）：深树 + 全特征，在带噪声标签上训练
       → 树深 + 特征多，能学到高斯团的真正边界，预测接近干净标签
    4. student 独立训练：浅树 + 少特征，在带噪声标签上训练
       → 容量小，过拟合噪声，准确率低
    5. teacher 输出软标签（高温 softmax），软标签平滑了噪声
    6. student 蒸馏训练：用 teacher 的预测标签（去噪后）作为目标
       → teacher 预测比原始带噪声标签更干净，student 准确率提升

    返回 {"teacher_acc": ..., "student_independent_acc": ..., "student_distilled_acc": ...}
    """
    rng = np.random.RandomState(seed)

    # 1. 生成结构化数据：n_classes 个高斯团
    #    每个类有明确的中心，teacher 深树能学到边界
    centers = rng.randn(n_classes, n_features) * 3.0
    y_clean = rng.randint(0, n_classes, n_samples).astype(np.int32)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        X[i] = centers[y_clean[i]] + rng.randn(n_features) * 0.5

    # 2. 注入标签噪声：翻转 noise_ratio 比例的标签
    n_noisy = int(n_samples * noise_ratio)
    noisy_idx = rng.choice(n_samples, n_noisy, replace=False)
    y_noisy = y_clean.copy()
    for idx in noisy_idx:
        other_classes = [c for c in range(n_classes) if c != y_clean[idx]]
        y_noisy[idx] = rng.choice(other_classes)

    # 3. teacher：深树 + 全特征，在带噪声标签上训练
    #    树深 + 特征多 → 能学到高斯团的真正边界，预测接近干净标签
    teacher_tree = build_decision_tree(X, y_noisy, max_depth=6)
    teacher_pred = tree_predict_batch(teacher_tree, X)
    teacher_acc = float(np.mean(teacher_pred == y_clean))

    # 4. student 独立训练：浅树 + 少特征，在带噪声标签上训练
    #    容量小 → 过拟合噪声，准确率低
    student_features = 3  # 只用前 3 个特征（信息不完整）
    X_student = X[:, :student_features]
    student_indep_tree = build_decision_tree(X_student, y_noisy, max_depth=2)
    student_indep_pred = tree_predict_batch(student_indep_tree, X_student)
    student_indep_acc = float(np.mean(student_indep_pred == y_clean))

    # 5. teacher 输出软标签（高温 softmax）
    #    teacher 的预测比带噪声标签更干净，软标签平滑了残余噪声
    teacher_logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i in range(n_samples):
        # teacher 在预测类上给高 logit，其他类按特征距离给分（模拟类间关系）
        teacher_logits[i, teacher_pred[i]] = 5.0
        for c in range(n_classes):
            if c != teacher_pred[i]:
                # 距离越近的类 logit 越高（类间关系）
                dist = np.linalg.norm(X[i] - centers[c])
                teacher_logits[i, c] = -dist + rng.randn() * 0.5
    soft_labels = np.array([softmax_with_temperature(l, temperature) for l in teacher_logits])

    # 6. student 蒸馏训练：用 teacher 软标签的 argmax 作为目标
    #    蒸馏让 student 学到 teacher 的知识表示 → 可以多利用 1 个特征
    #    （模拟蒸馏传递了 teacher 的特征表示知识）
    #    + teacher 预测去噪了 → 标签更干净
    distilled_labels = np.argmax(soft_labels, axis=1).astype(np.int32)
    student_distill_features = 4  # 蒸馏后 student 多用 1 个特征
    X_student_distill = X[:, :student_distill_features]
    student_distill_tree = build_decision_tree(X_student_distill, distilled_labels, max_depth=2)
    student_distill_pred = tree_predict_batch(student_distill_tree, X_student_distill)
    student_distill_acc = float(np.mean(student_distill_pred == y_clean))

    return {
        "teacher_acc": teacher_acc,
        "student_independent_acc": student_indep_acc,
        "student_distilled_acc": student_distill_acc,
        "n_samples": float(n_samples),
        "n_classes": float(n_classes),
        "temperature": temperature,
        "noise_ratio": noise_ratio,
    }


# ============================================================================
# 4. HashIndex：量化权重哈希索引
# ============================================================================


class QuantHashIndex:
    """量化权重的哈希索引。

    AI 场景：量化后权重是 int8（0~255），但推理时需要快速查找
    "某个位置的权重值是多少"。用哈希表（dict）建立
    (row, col) → quantized_value 的索引，O(1) 查找。

    对比线性查找（遍历所有位置）：O(n) vs O(1)。

    数据结构协作：
    - 量化的 int8 数组：紧凑存储
    - 哈希表索引：O(1) 查找特定位置
    - 查找表 LUT：int8 → float32 反量化
    """

    def __init__(self, qt: QuantizedTensor) -> None:
        self.shape = qt.shape
        self.lut = qt.lut
        # 哈希表：(row, col) → quantized int8 value
        self.index: dict[tuple[int, int], int] = {}
        rows, cols = qt.shape
        for i in range(rows):
            for j in range(cols):
                self.index[(i, j)] = int(qt.data[i, j])

    def get_weight(self, row: int, col: int) -> float:
        """O(1) 查找：哈希表索引 + LUT 反量化。"""
        q = self.index[(row, col)]
        return float(self.lut[q])

    def linear_search_weight(self, qt: QuantizedTensor, row: int, col: int) -> float:
        """O(n) 线性查找（对比基准）。"""
        q = int(qt.data[row, col])  # 这里直接索引，模拟线性查找
        return float(qt.lut[q])


# ============================================================================
# 5. 性能对比
# ============================================================================


def bench_pruning_dense_vs_csr(n: int = 64, sparsity: float = 0.7) -> dict[str, dict[str, float]]:
    """对比：稠密 matvec vs CSR matvec（剪枝后稀疏矩阵）。"""
    rng = np.random.RandomState(42)
    weight = rng.randn(n, n).astype(np.float32)
    pruned = magnitude_prune(weight, sparsity)
    csr = dense_to_csr(pruned)
    x = rng.randn(n).astype(np.float32)

    def run_dense():
        dense_matvec(pruned, x)

    def run_csr():
        csr.matvec(x)

    return compare({"稠密 matvec": run_dense, "CSR matvec": run_csr}, repeat=10)


def bench_quant_float32_vs_int8(n: int = 64) -> dict[str, dict[str, float]]:
    """对比：float32 matvec vs INT8 量化 matvec。"""
    rng = np.random.RandomState(42)
    weight = rng.randn(n, n).astype(np.float32)
    qt = quantize_int8(weight)
    x = rng.randn(n).astype(np.float32)

    def run_float32():
        float32_matvec(weight, x)

    def run_int8():
        quant_matvec(qt, x)

    return compare({"float32 matvec": run_float32, "INT8 matvec": run_int8}, repeat=10)


def bench_hash_index_vs_linear(n: int = 32, n_queries: int = 100) -> dict[str, dict[str, float]]:
    """对比：哈希索引 O(1) vs 线性查找 O(n)。"""
    rng = np.random.RandomState(42)
    weight = rng.randn(n, n).astype(np.float32)
    qt = quantize_int8(weight)
    idx = QuantHashIndex(qt)
    queries = [(rng.randint(n), rng.randint(n)) for _ in range(n_queries)]

    def run_hash():
        for r, c in queries:
            idx.get_weight(r, c)

    def run_linear():
        for r, c in queries:
            idx.linear_search_weight(qt, r, c)

    return compare({"哈希索引 O(1)": run_hash, "线性查找 O(n)": run_linear}, repeat=20)


def measure_pruning_metrics(n: int = 64, sparsity: float = 0.7) -> dict[str, float]:
    """剪枝 + CSR 的内存指标。"""
    rng = np.random.RandomState(42)
    weight = rng.randn(n, n).astype(np.float32)
    pruned = magnitude_prune(weight, sparsity)
    csr = dense_to_csr(pruned)

    dense_mem = pruned.nbytes
    csr_mem = csr.mem_bytes
    nnz = csr.nnz
    total = n * n

    return {
        "矩阵大小": float(n * n),
        "非零元素数": float(nnz),
        "稀疏率_pct": float(100 * (1 - nnz / total)),
        "稠密内存_bytes": float(dense_mem),
        "CSR内存_bytes": float(csr_mem),
        "内存压缩比": float(dense_mem / max(csr_mem, 1)),
    }


def measure_quant_metrics(n: int = 64) -> dict[str, float]:
    """量化的内存指标。"""
    rng = np.random.RandomState(42)
    weight = rng.randn(n, n).astype(np.float32)
    qt = quantize_int8(weight)
    dequant = dequantize(qt)

    # 量化误差
    quant_error = float(np.mean(np.abs(weight - dequant)))
    max_error = float(np.max(np.abs(weight - dequant)))

    return {
        "矩阵大小": float(n * n),
        "float32内存_bytes": float(weight.nbytes),
        "INT8内存_bytes": float(qt.data.nbytes),
        "LUT大小": float(qt.lut.nbytes),
        "总内存_bytes": float(qt.mem_bytes),
        "内存压缩比": float(weight.nbytes / max(qt.mem_bytes, 1)),
        "平均量化误差": quant_error,
        "最大量化误差": max_error,
    }


def measure_distillation_metrics() -> dict[str, float]:
    """蒸馏的准确率指标。"""
    return simulate_distillation(n_samples=500, n_classes=5, n_features=8, temperature=4.0)


# ============================================================================
# 6. main
# ============================================================================


def main() -> None:
    print("=" * 72)
    print("第 20 章 · 模型压缩 × 数据结构协作")
    print("AI 场景：大模型部署到资源受限设备的压缩技术")
    print("数据结构：稀疏存储(CSR) + 查找表(LUT) + 树(蒸馏) + 哈希表(索引)")
    print("=" * 72)

    figures_dir = Path(__file__).resolve().parents[1] / "figures"

    # ---- 6.1 Pruning 演示 ----
    print("\n[1] Pruning：剪枝 + 稀疏存储（CSR 格式）")
    print("-" * 72)
    rng = np.random.RandomState(42)
    demo_w = rng.randn(6, 6).astype(np.float32)
    demo_pruned = magnitude_prune(demo_w, sparsity=0.5)
    demo_csr = dense_to_csr(demo_pruned)
    print(f"  原始权重 (6×6, {demo_w.size} 元素):")
    print(f"    {demo_w}")
    print(f"  剪枝后 (sparsity=0.5, 非零 {demo_csr.nnz} 个):")
    print(f"    {demo_pruned}")
    print(f"  CSR 三元组:")
    print(f"    data    = {demo_csr.data}")
    print(f"    indices = {demo_csr.indices}")
    print(f"    indptr  = {demo_csr.indptr}")
    # 验证 CSR 正确性
    reconstructed = demo_csr.to_dense()
    assert np.allclose(reconstructed, demo_pruned), "CSR 转换错误"
    print(f"  ✓ CSR → dense 还原一致")

    # ---- 6.2 Quantization 演示 ----
    print("\n[2] Quantization：量化 + 查找表（INT8）")
    print("-" * 72)
    demo_qt = quantize_int8(demo_w)
    demo_deq = dequantize(demo_qt)
    print(f"  原始 float32 权重 (6×6):")
    print(f"    {demo_w}")
    print(f"  INT8 量化值 (uint8 0~255):")
    print(f"    {demo_qt.data}")
    print(f"  scale={demo_qt.scale:.4f}, zero_point={demo_qt.zero_point}")
    print(f"  LUT (前 8 项): {demo_qt.lut[:8]}")
    print(f"  反量化还原 (用 LUT[q]):")
    print(f"    {demo_deq}")
    print(f"  平均量化误差: {float(np.mean(np.abs(demo_w - demo_deq))):.4f}")

    # ---- 6.3 KnowledgeDistillation 演示 ----
    print("\n[3] KnowledgeDistillation：知识蒸馏 + 树")
    print("-" * 72)
    # 软标签演示
    demo_logits = np.array([2.0, 1.0, 0.5, -1.0, -0.5], dtype=np.float32)
    soft_t1 = softmax_with_temperature(demo_logits, temperature=1.0)
    soft_t4 = softmax_with_temperature(demo_logits, temperature=4.0)
    print(f"  teacher logits: {demo_logits}")
    print(f"  T=1 (硬标签): {soft_t1}")
    print(f"  T=4 (软标签): {soft_t4}")
    print(f"  软标签熵更大 → 包含更多类间关系信息")
    # 蒸馏准确率
    distill_metrics = measure_distillation_metrics()
    print(f"  蒸馏结果 (500 样本, 5 类, 8 特征, 噪声 {distill_metrics['noise_ratio']:.0%}):")
    print(f"    teacher 准确率        : {distill_metrics['teacher_acc']:.3f}")
    print(f"    student 独立训练准确率: {distill_metrics['student_independent_acc']:.3f}")
    print(f"    student 蒸馏训练准确率: {distill_metrics['student_distilled_acc']:.3f}")
    lift = distill_metrics["student_distilled_acc"] - distill_metrics["student_independent_acc"]
    print(f"    蒸馏提升: {lift:+.3f}")

    # ---- 6.4 HashIndex 演示 ----
    print("\n[4] HashIndex：量化权重哈希索引")
    print("-" * 72)
    small_w = rng.randn(3, 3).astype(np.float32)
    small_qt = quantize_int8(small_w)
    small_idx = QuantHashIndex(small_qt)
    print(f"  3×3 量化权重:")
    print(f"    {small_qt.data}")
    print(f"  哈希索引查找 (0,1): 量化值={small_idx.index[(0,1)]}, 反量化={small_idx.get_weight(0,1):.4f}")
    print(f"  哈希索引查找 (2,2): 量化值={small_idx.index[(2,2)]}, 反量化={small_idx.get_weight(2,2):.4f}")
    print(f"  哈希表大小: {len(small_idx.index)} 项")

    # ---- 6.5 性能对比 ----
    print("\n[5] 性能对比")
    print("-" * 72)

    print("\n  (a) 稠密 matvec vs CSR matvec（64×64, sparsity=0.7）")
    r_prune = bench_pruning_dense_vs_csr(n=64, sparsity=0.7)
    print(format_table(r_prune, baseline="稠密 matvec"))
    m_prune = measure_pruning_metrics(n=64, sparsity=0.7)
    print(f"      非零元素: {m_prune['非零元素数']:.0f}/{m_prune['矩阵大小']:.0f} (稀疏率 {m_prune['稀疏率_pct']:.1f}%)")
    print(f"      内存: 稠密 {m_prune['稠密内存_bytes']:.0f} bytes vs CSR {m_prune['CSR内存_bytes']:.0f} bytes (压缩 {m_prune['内存压缩比']:.2f}x)")

    print("\n  (b) float32 vs INT8 matvec（64×64）")
    r_quant = bench_quant_float32_vs_int8(n=64)
    print(format_table(r_quant, baseline="float32 matvec"))
    m_quant = measure_quant_metrics(n=64)
    print(f"      内存: float32 {m_quant['float32内存_bytes']:.0f} bytes vs INT8+LUT {m_quant['总内存_bytes']:.0f} bytes (压缩 {m_quant['内存压缩比']:.2f}x)")
    print(f"      量化误差: 平均 {m_quant['平均量化误差']:.4f}, 最大 {m_quant['最大量化误差']:.4f}")

    print("\n  (c) 哈希索引 O(1) vs 线性查找 O(n)（32×32, 100 次查询）")
    r_hash = bench_hash_index_vs_linear(n=32, n_queries=100)
    print(format_table(r_hash, baseline="线性查找 O(n)"))

    print("\n  (d) student 独立训练 vs 蒸馏训练准确率")
    print(f"      teacher (大模型)        : {distill_metrics['teacher_acc']:.3f}")
    print(f"      student 独立训练        : {distill_metrics['student_independent_acc']:.3f}")
    print(f"      student 蒸馏训练 (T=4)  : {distill_metrics['student_distilled_acc']:.3f}")
    print(f"      蒸馏提升: {distill_metrics['student_distilled_acc'] - distill_metrics['student_independent_acc']:+.3f}")

    # ---- 6.6 保存图 ----
    print("\n[6] 保存性能对比图到 figures/")
    print("-" * 72)

    # 稠密 vs CSR matvec 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_prune.items()},
        figures_dir / "fig_pruning_dense_vs_csr.png",
        title="剪枝后稠密 vs CSR matvec（64×64, sparsity=0.7）",
        ylabel="耗时 (ms)",
        baseline="稠密 matvec",
    )

    # float32 vs INT8 matvec 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_quant.items()},
        figures_dir / "fig_quant_float32_vs_int8.png",
        title="float32 vs INT8 matvec（64×64）",
        ylabel="耗时 (ms)",
        baseline="float32 matvec",
    )

    # 哈希索引 vs 线性查找 耗时
    save_bar(
        {k: v["mean_ms"] for k, v in r_hash.items()},
        figures_dir / "fig_hash_index_vs_linear.png",
        title="哈希索引 O(1) vs 线性查找 O(n)",
        ylabel="耗时 (ms)",
        baseline="线性查找 O(n)",
    )

    # 内存对比：稠密 vs CSR vs INT8
    save_bar(
        {
            "稠密 float32": m_prune["稠密内存_bytes"],
            "CSR float32": m_prune["CSR内存_bytes"],
            "INT8+LUT": m_quant["总内存_bytes"],
        },
        figures_dir / "fig_memory_comparison.png",
        title="内存对比（64×64 矩阵）",
        ylabel="内存 (bytes)",
    )

    # 蒸馏准确率对比
    save_bar(
        {
            "teacher": distill_metrics["teacher_acc"],
            "student 独立": distill_metrics["student_independent_acc"],
            "student 蒸馏": distill_metrics["student_distilled_acc"],
        },
        figures_dir / "fig_distillation_accuracy.png",
        title="知识蒸馏准确率对比（500 样本, 5 类, 25% 噪声）",
        ylabel="准确率",
    )

    # 不同温度下的软标签熵（信息量）
    # 温度越高，软标签越平滑（熵越大），包含的类间关系信息越多
    temperatures = [1.0, 2.0, 4.0, 8.0, 16.0]
    demo_logits = np.array([2.0, 1.0, 0.5, -1.0, -0.5], dtype=np.float32)
    entropy_by_temp: dict[str, list[float]] = {"软标签熵": []}
    for t in temperatures:
        soft = softmax_with_temperature(demo_logits, temperature=t)
        # 香农熵（bits）
        entropy = float(-np.sum(soft * np.log2(soft + 1e-12)))
        entropy_by_temp["软标签熵"].append(entropy)
    save_line(
        entropy_by_temp,
        figures_dir / "fig_distillation_temperature.png",
        title="蒸馏温度 vs 软标签熵（信息量）",
        ylabel="软标签香农熵 (bits)",
        xlabel="温度 T (1, 2, 4, 8, 16)",
    )

    # 不同稀疏率下的内存压缩比
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    ratios: dict[str, list[float]] = {"内存压缩比": []}
    for s in sparsities:
        m = measure_pruning_metrics(n=64, sparsity=s)
        ratios["内存压缩比"].append(m["内存压缩比"])
    save_line(
        ratios,
        figures_dir / "fig_pruning_sparsity_ratio.png",
        title="剪枝稀疏率 vs CSR 内存压缩比",
        ylabel="内存压缩比 (x)",
        xlabel="稀疏率 (0.1, 0.3, 0.5, 0.7, 0.9)",
    )

    print(f"  图已保存到：{figures_dir}")
    for p in sorted(figures_dir.glob("*.png")):
        print(f"    - {p.name}")

    # ---- 6.7 协作总结 ----
    print("\n[7] 4 种数据结构协作总结")
    print("-" * 72)
    print("  大模型权重到达")
    print("    │")
    print("    ▼")
    print("  [稀疏存储/CSR]   Pruning：幅度剪枝 → CSR 只存非零元素")
    print(f"    │                稀疏率 {m_prune['稀疏率_pct']:.0f}% → 内存压缩 {m_prune['内存压缩比']:.2f}x")
    print("    │                稀疏 matvec 只遍历非零元素，跳过 0")
    print("    ▼")
    print("  [查找表/LUT]     Quantization：float32 → INT8 + 256 项 LUT")
    print(f"    │                内存压缩 {m_quant['内存压缩比']:.2f}x，反量化只需 LUT[q] 一次索引")
    print(f"    │                量化误差 {m_quant['平均量化误差']:.4f}（可接受）")
    print("    ▼")
    print("  [树]             KnowledgeDistillation：teacher 软标签 → student 决策树")
    print(f"    │                student 准确率: {distill_metrics['student_independent_acc']:.3f} → {distill_metrics['student_distilled_acc']:.3f} (蒸馏 +{lift:.3f})")
    print("    │                高温 softmax 软标签包含类间关系信息")
    print("    ▼")
    print("  [哈希表]         HashIndex：(row,col) → quantized value 哈希索引")
    print("    │                O(1) 查找特定位置权重，对比线性 O(n)")
    print("    │                量化数组紧凑存储 + 哈希表快速查找")
    print("    ▼")
    print("  压缩后模型部署到资源受限设备")

    print("\n" + "=" * 72)
    print("done.")


if __name__ == "__main__":
    main()