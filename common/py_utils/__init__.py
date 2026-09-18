"""公用工具：基准测试 + 可视化。

所有章节的 Python demo 都通过相对导入使用：
    from ...common.py_utils import benchmark, viz
"""

from .benchmark import Timer, compare, format_table
from .viz import save_bar, save_line, save_scatter

__all__ = [
    "Timer",
    "compare",
    "format_table",
    "save_bar",
    "save_line",
    "save_scatter",
]