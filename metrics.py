"""计算二值医学图像分割的综合指标。

各指标按病例计算以支持宏平均，不把整个测试集展平为由背景主导的单个混淆
矩阵。距离指标使用传入的像素间距；默认 ``spacing=(1, 1)`` 时单位为像素。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree


BOUNDARY_METRICS = {"hausdorff", "hd95", "assd", "surface_dice"}

METRIC_NAMES = (
    "dice",
    "iou",
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "npv",
    "balanced_accuracy",
    "mcc",
    "fpr",
    "fnr",
    "mae",
    "volumetric_similarity",
    "rvd_percent",
    "ravd_percent",
    "hausdorff",
    "hd95",
    "assd",
    "surface_dice",
    "roc_auc",
    "average_precision",
    "brier",
    "nll",
    "ece",
    "class_balanced_mae",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def binary_ranking_metrics(
    target: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    """使用一次稳定像素排序返回精确 ROC-AUC 与阶梯式 AP。

    输入为单病例二值标签和前景概率，输出为 ``(roc_auc, average_precision)``。
    对概率执行一次降序稳定排序并合并同分阈值，可同时保持两项指标的定义，
    避免对原始分辨率概率图分别排序两次。本函数只计算排序指标，不执行模型
    推理、阈值分割或跨病例汇总。
    """

    target_flat = target.astype(np.uint8).ravel()
    probability_flat = probability.astype(np.float64).ravel()
    positives = int(target_flat.sum())
    negatives = target_flat.size - positives
    if probability_flat.size == 0:
        return float("nan"), float("nan")

    order = np.argsort(-probability_flat, kind="mergesort")
    sorted_target = target_flat[order]
    sorted_probability = probability_flat[order]
    cumulative_tp = np.cumsum(sorted_target, dtype=np.float64)
    threshold_ends = np.r_[
        np.where(np.diff(sorted_probability) != 0)[0], sorted_probability.size - 1
    ]
    tp = cumulative_tp[threshold_ends]
    evaluated = threshold_ends.astype(np.float64) + 1.0
    fp = evaluated - tp

    if positives == 0 or negatives == 0:
        roc_auc = float("nan")
    else:
        true_positive_rate = tp / positives
        false_positive_rate = fp / negatives
        previous_tpr = np.r_[0.0, true_positive_rate[:-1]]
        previous_fpr = np.r_[0.0, false_positive_rate[:-1]]
        roc_auc = float(
            np.sum(
                (false_positive_rate - previous_fpr)
                * (true_positive_rate + previous_tpr)
                * 0.5
            )
        )

    if positives == 0:
        average_precision = float("nan")
    else:
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / positives
        previous_recall = np.r_[0.0, recall[:-1]]
        average_precision = float(np.sum((recall - previous_recall) * precision))
    return roc_auc, average_precision


def binary_roc_auc(target: np.ndarray, probability: np.ndarray) -> float:
    """输入二值标签和前景概率，返回单病例 ROC-AUC。"""

    return binary_ranking_metrics(target, probability)[0]


def binary_average_precision(target: np.ndarray, probability: np.ndarray) -> float:
    """输入二值标签和前景概率，按同分阈值分组后返回阶梯式平均精确率。"""

    return binary_ranking_metrics(target, probability)[1]


def foreground_ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    """输入二值标签和前景概率，返回等宽分箱的前景期望校准误差。"""

    if bins < 2:
        raise ValueError("ece bins must be at least 2")
    target = target.astype(np.float64).ravel()
    probability = np.clip(probability.astype(np.float64).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.searchsorted(edges, probability, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    counts = np.bincount(bin_indices, minlength=bins).astype(np.float64)
    probability_sums = np.bincount(
        bin_indices, weights=probability, minlength=bins
    )
    target_sums = np.bincount(bin_indices, weights=target, minlength=bins)
    valid = counts > 0
    mean_probability = probability_sums[valid] / counts[valid]
    mean_target = target_sums[valid] / counts[valid]
    return float(
        np.sum((counts[valid] / probability.size) * np.abs(mean_probability - mean_target))
    )


def class_balanced_mae(target: np.ndarray, probability: np.ndarray) -> float:
    """分别计算存在类别的绝对概率误差，并返回类别均衡平均值。"""

    target_bool = target.astype(bool)
    values = []
    for class_value in (False, True):
        selected = target_bool == class_value
        if not selected.any():
            continue
        values.append(float(np.mean(np.abs(probability[selected] - target_bool[selected]))))
    return float(np.mean(values)) if values else float("nan")


def _surfaces(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def _contour_elements(
    mask: np.ndarray, spacing: Tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """返回二维轮廓边的中点坐标及其物理长度。

    水平边长度为 ``spacing_x``，垂直边长度为 ``spacing_y``。以轮廓长度而非
    边界像素计数，可避免归一化 Surface Dice 中的各向异性和对角线偏差。
    """

    mask = mask.astype(bool)
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    top = center & ~padded[:-2, 1:-1]
    bottom = center & ~padded[2:, 1:-1]
    left = center & ~padded[1:-1, :-2]
    right = center & ~padded[1:-1, 2:]
    coordinates = []
    lengths = []

    rows, cols = np.nonzero(top)
    if rows.size:
        coordinates.append(np.column_stack((rows * spacing[0], (cols + 0.5) * spacing[1])))
        lengths.append(np.full(rows.size, spacing[1], dtype=np.float64))
    rows, cols = np.nonzero(bottom)
    if rows.size:
        coordinates.append(
            np.column_stack(((rows + 1.0) * spacing[0], (cols + 0.5) * spacing[1]))
        )
        lengths.append(np.full(rows.size, spacing[1], dtype=np.float64))
    rows, cols = np.nonzero(left)
    if rows.size:
        coordinates.append(np.column_stack(((rows + 0.5) * spacing[0], cols * spacing[1])))
        lengths.append(np.full(rows.size, spacing[0], dtype=np.float64))
    rows, cols = np.nonzero(right)
    if rows.size:
        coordinates.append(
            np.column_stack(((rows + 0.5) * spacing[0], (cols + 1.0) * spacing[1]))
        )
        lengths.append(np.full(rows.size, spacing[0], dtype=np.float64))

    if not coordinates:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(coordinates, axis=0), np.concatenate(lengths, axis=0)


def _boundary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float],
    surface_tolerance: float,
) -> Dict[str, float]:
    """按给定间距和容差计算单个二维病例的四项边界指标。"""

    prediction_has_fg = bool(prediction.any())
    target_has_fg = bool(target.any())
    height, width = target.shape
    diagonal = math.hypot((height - 1) * spacing[0], (width - 1) * spacing[1])

    if not target_has_fg and not prediction_has_fg:
        return {"hausdorff": 0.0, "hd95": 0.0, "assd": 0.0, "surface_dice": 1.0}
    if not target_has_fg or not prediction_has_fg:
        return {
            "hausdorff": diagonal,
            "hd95": diagonal,
            "assd": diagonal,
            "surface_dice": 0.0,
        }

    # 距离图只需覆盖预测与标签前景的联合包围盒；所有查询表面及其可能的最近
    # 目标表面都位于该区域内，因此裁剪与全图 EDT 数值等价，同时可避免高分辨率
    # ISIC 背景产生数 GB 的临时数组。
    union = prediction | target
    union_rows, union_columns = np.nonzero(union)
    row_start = int(union_rows.min())
    row_stop = int(union_rows.max()) + 1
    column_start = int(union_columns.min())
    column_stop = int(union_columns.max()) + 1
    prediction = prediction[row_start:row_stop, column_start:column_stop]
    target = target[row_start:row_stop, column_start:column_stop]

    prediction_surface = _surfaces(prediction)
    target_surface = _surfaces(target)
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    pred_to_target = distance_to_target[prediction_surface]
    target_to_pred = distance_to_prediction[target_surface]

    hausdorff = max(float(pred_to_target.max()), float(target_to_pred.max()))
    # HD95 取两个有向距离第 95 百分位数的较大值；部分库会对拼接后的距离集合
    # 统一取百分位数，因此在此明确本项目采用的定义。
    hd95 = max(
        float(np.percentile(pred_to_target, 95)),
        float(np.percentile(target_to_pred, 95)),
    )
    assd = float(
        (pred_to_target.sum() + target_to_pred.sum())
        / (pred_to_target.size + target_to_pred.size)
    )
    prediction_contour, prediction_lengths = _contour_elements(prediction, spacing)
    target_contour, target_lengths = _contour_elements(target, spacing)
    contour_pred_to_target = cKDTree(target_contour).query(prediction_contour, k=1)[0]
    contour_target_to_pred = cKDTree(prediction_contour).query(target_contour, k=1)[0]
    surface_dice = float(
        (
            prediction_lengths[contour_pred_to_target <= surface_tolerance].sum()
            + target_lengths[contour_target_to_pred <= surface_tolerance].sum()
        )
        / (prediction_lengths.sum() + target_lengths.sum())
    )
    return {
        "hausdorff": hausdorff,
        "hd95": hd95,
        "assd": assd,
        "surface_dice": surface_dice,
    }


def compute_binary_segmentation_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    spacing: Tuple[float, float] = (1.0, 1.0),
    surface_tolerance: float = 2.0,
    ece_bins: int = 15,
) -> Dict[str, float]:
    """计算单个二维二值分割病例的重叠、误差、边界与概率指标。

    ``probability`` 为前景概率图，``target`` 为标签图；``threshold`` 决定二值
    预测，``spacing`` 和 ``surface_tolerance`` 控制物理边界距离，``ece_bins``
    控制校准分箱。返回以 ``METRIC_NAMES`` 中名称为键的指标字典。本函数边界止于
    单病例数组计算，不负责网络推理、图像尺寸恢复、跨病例统计或五折汇总。
    """

    probability = np.asarray(probability, dtype=np.float64).squeeze()
    target = np.asarray(target).squeeze() >= 0.5
    if probability.ndim != 2 or target.ndim != 2:
        raise ValueError("probability and target must each describe one 2D mask")
    if probability.shape != target.shape:
        raise ValueError(f"shape mismatch: probability={probability.shape}, target={target.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains NaN or Inf")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    if spacing[0] <= 0 or spacing[1] <= 0:
        raise ValueError("spacing values must be positive")
    if surface_tolerance < 0:
        raise ValueError("surface_tolerance must be non-negative")

    probability = np.clip(probability, 0.0, 1.0)
    prediction = probability >= threshold
    tp = float(np.count_nonzero(prediction & target))
    tn = float(np.count_nonzero(~prediction & ~target))
    fp = float(np.count_nonzero(prediction & ~target))
    fn = float(np.count_nonzero(~prediction & target))
    target_fg = tp + fn
    prediction_fg = tp + fp
    target_bg = tn + fp
    prediction_bg = tn + fn

    dice_denominator = 2.0 * tp + fp + fn
    union = tp + fp + fn
    dice = 1.0 if dice_denominator == 0 else 2.0 * tp / dice_denominator
    iou = 1.0 if union == 0 else tp / union
    sensitivity = _safe_ratio(tp, target_fg)
    specificity = _safe_ratio(tn, target_bg)
    precision = _safe_ratio(tp, prediction_fg)
    npv = _safe_ratio(tn, prediction_bg)
    balanced_accuracy = (
        0.5 * (sensitivity + specificity)
        if math.isfinite(sensitivity) and math.isfinite(specificity)
        else float("nan")
    )
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else float("nan")
    volume_denominator = 2.0 * tp + fp + fn
    volumetric_similarity = (
        1.0 if volume_denominator == 0 else 1.0 - abs(fn - fp) / volume_denominator
    )
    if target_fg > 0:
        rvd_percent = 100.0 * (prediction_fg - target_fg) / target_fg
        ravd_percent = abs(rvd_percent)
    elif prediction_fg == 0:
        rvd_percent = 0.0
        ravd_percent = 0.0
    else:
        rvd_percent = float("nan")
        ravd_percent = float("nan")

    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    target_float = target.astype(np.float64)
    roc_auc, average_precision = binary_ranking_metrics(target, probability)
    result: Dict[str, float] = {
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float((tp + tn) / target.size),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "npv": float(npv),
        "balanced_accuracy": balanced_accuracy,
        "mcc": float(mcc),
        "fpr": float(_safe_ratio(fp, target_bg)),
        "fnr": float(_safe_ratio(fn, target_fg)),
        "mae": float(np.mean(np.abs(probability - target_float))),
        "volumetric_similarity": float(volumetric_similarity),
        "rvd_percent": float(rvd_percent),
        "ravd_percent": float(ravd_percent),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "brier": float(np.mean((probability - target_float) ** 2)),
        "nll": float(
            -np.mean(target_float * np.log(clipped) + (1.0 - target_float) * np.log(1.0 - clipped))
        ),
        "ece": foreground_ece(target, probability, bins=ece_bins),
        "class_balanced_mae": class_balanced_mae(target, probability),
    }
    result.update(_boundary_metrics(prediction, target, spacing, surface_tolerance))
    return result


def pooled_auc_ap_from_histograms(
    positive_histogram: np.ndarray, negative_histogram: np.ndarray
) -> tuple[float, float]:
    """由升序概率直方图近似返回汇总像素级 AUC 与 AP。

    输入是等长的正类和负类计数向量，输出为 ``(auc, average_precision)``；
    本函数只聚合已统计的直方图，不读取病例、执行模型推理或生成逐病例指标。
    """

    positive = np.asarray(positive_histogram, dtype=np.float64)
    negative = np.asarray(negative_histogram, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("positive and negative histograms must be equal-length vectors")
    positives = float(positive.sum())
    negatives = float(negative.sum())
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")

    negative_below = np.cumsum(negative) - negative
    auc = float(np.sum(positive * (negative_below + 0.5 * negative)) / (positives * negatives))

    positive_desc = positive[::-1]
    negative_desc = negative[::-1]
    cumulative_tp = np.cumsum(positive_desc)
    cumulative_fp = np.cumsum(negative_desc)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall_increment = positive_desc / positives
    average_precision = float(np.sum(recall_increment * precision))
    return auc, average_precision
