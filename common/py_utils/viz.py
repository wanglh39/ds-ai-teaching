"""可视化工具：保存静态图（PNG）+ 简单动画（GIF）。

设计原则：
- 用 matplotlib（Agg 后端，无需显示环境）
- 所有函数都把图保存到文件，不弹窗
- 文件名 snake_case，UTF-8
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体：Windows 用 SimHei/微软雅黑，Linux 用 WenQuanYi，macOS 用 PingFang
for _font in ["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_bar(
    data: dict[str, float],
    out_path: str | Path,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    baseline: str | None = None,
) -> Path:
    """柱状图对比多个实现的指标（如耗时）。

    Args:
        data: {"朴素": 12.3, "优化": 3.4}
        baseline: 基准名，会在柱顶标注加速比
    """
    p = _ensure_dir(out_path)
    names = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    bars = ax.bar(names, values, color=["#d62728", "#2ca02c"][: len(names)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if baseline and baseline in data and data[baseline] > 0:
        for bar, name in zip(bars, names):
            speedup = data[baseline] / data[name] if data[name] > 0 else float("inf")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
            )
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def save_line(
    series: dict[str, list[float]],
    out_path: str | Path,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
) -> Path:
    """折线图，多组数据对比。

    Args:
        series: {"朴素": [y1, y2, ...], "优化": [y1, y2, ...]}
    """
    p = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for name, ys in series.items():
        ax.plot(ys, label=name, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def save_scatter(
    points: list[tuple[float, float]],
    out_path: str | Path,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    extra: list[tuple[float, float, str]] | None = None,
) -> Path:
    """散点图，常用于数据分布 + 标注点（如 KNN 的查询点）。

    Args:
        points: [(x, y), ...]
        extra: [(x, y, label), ...] 额外标注点
    """
    p = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=10, alpha=0.5)
    if extra:
        for x, y, label in extra:
            ax.scatter([x], [y], s=60, c="red", marker="x")
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p