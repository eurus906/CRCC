"""提供五折训练与测试共享的比例、折号和路径规范化工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def normalize_ratio(value: str | int) -> str:
    """将整数或比例文本规范化为 ``ratio_XX``，并返回标准目录名。"""

    text = str(value).strip().lower()
    if text.startswith("ratio_"):
        text = text[6:]
    if text.endswith("%"):
        text = text[:-1]
    try:
        ratio = int(text)
    except ValueError as error:
        raise ValueError(f"Invalid ratio: {value}") from error
    if ratio <= 0 or ratio >= 100:
        raise ValueError("ratio must be a percentage between 1 and 99")
    return f"ratio_{ratio:02d}"


def validate_folds(folds: Iterable[int]) -> List[int]:
    """校验折号非空、不重复且位于 1 至 5，并按输入顺序返回列表。"""

    values = list(folds)
    if not values:
        raise ValueError("At least one fold is required")
    if len(values) != len(set(values)):
        raise ValueError("Fold numbers must not repeat")
    invalid = [fold for fold in values if fold < 1 or fold > 5]
    if invalid:
        raise ValueError(f"Fold numbers must be in 1..5, got {invalid}")
    return values


def split_directory(splits_root: str | Path, fold: int, ratio: str | int) -> Path:
    """根据划分根目录、折号和标注比例返回对应的折目录路径。"""

    return Path(splits_root) / f"fold_{fold}" / normalize_ratio(ratio)
