"""执行五折离线测试，并输出逐病例指标及 CSV 汇总结果。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from compare.common.inference import build_inference_model
from compare.registry import normalize_compare_method
from data import IMAGENET_MEAN, IMAGENET_STD, LabeledSegmentationDataset, seed_worker
from fold_utils import normalize_ratio, split_directory, validate_folds
from metrics import (
    BOUNDARY_METRICS,
    METRIC_NAMES,
    compute_binary_segmentation_metrics,
    pooled_auc_ap_from_histograms,
)
from train import resolve_device


EVALUATION_IMPLEMENTATION_VERSION = 5


GLOBAL_METRIC_NAMES = (
    "global_dice",
    "global_iou",
    "global_accuracy",
    "global_precision",
    "global_sensitivity",
    "global_specificity",
    "global_mcc",
    "positive_case_dice",
    "positive_case_iou",
    "pooled_roc_auc",
    "pooled_average_precision",
    "negative_case_fp_rate",
    "negative_case_mean_fp_pixels",
)


def parse_args() -> argparse.Namespace:
    """解析并校验离线测试参数，返回供后续评测使用的命名空间。"""

    parser = argparse.ArgumentParser(
        description="Test all fold checkpoints and save comprehensive medical segmentation metrics"
    )
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing fold_1, ..., fold_5 checkpoint folders",
    )
    parser.add_argument("--ratio", required=True, help="1, 2, 5, 10, ratio_05, ...")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=None,
        help="If set, every fold model evaluates this same independent test set",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Use splits-root/external_test.tsv for all fold models",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--metric-workers",
        type=int,
        default=None,
        help=(
            "CPU processes for per-case medical metrics; default is 2 in original "
            "space and 0 in resized space. Use 0 to disable metric multiprocessing"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--surface-tolerance", type=float, default=2.0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--probability-histogram-bins", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument(
        "--metric-space",
        choices=("original", "resized"),
        default="original",
        help="Restore probabilities to original mask size before metrics, or use network size",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=2,
        metavar=("Y", "X"),
        default=None,
        help="Constant pixel spacing for distance metrics; default 1 1 reports pixels",
    )
    parser.add_argument("--distance-unit", choices=("pixel", "mm"), default="pixel")
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="测试时保存预测掩膜、轮廓叠加图和四联对比图",
    )
    parser.add_argument(
        "--visualization-max-side",
        type=int,
        default=512,
        help="限制叠加图与四联图中单个子图的最大边长；原始预测掩膜不缩放",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        args.ratio = normalize_ratio(args.ratio)
        args.folds = validate_folds(args.folds)
    except ValueError as error:
        parser.error(str(error))
    if args.external and args.external_manifest is not None:
        parser.error("Use either --external or --external-manifest, not both")
    if args.external:
        args.external_manifest = args.splits_root / "external_test.tsv"
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0, 1)")
    if args.surface_tolerance < 0:
        parser.error("--surface-tolerance must be non-negative")
    if args.ece_bins < 2:
        parser.error("--ece-bins must be at least 2")
    if args.probability_histogram_bins < 128:
        parser.error("--probability-histogram-bins must be at least 128")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.metric_workers is None:
        args.metric_workers = 2 if args.metric_space == "original" else 0
    if args.metric_workers < 0:
        parser.error("--metric-workers must be non-negative")
    spacing_was_provided = args.spacing is not None
    if args.spacing is None:
        args.spacing = (1.0, 1.0)
    if min(args.spacing) <= 0:
        parser.error("--spacing values must be positive")
    if args.distance_unit == "mm" and not spacing_was_provided:
        parser.error("mm was requested but --spacing Y X was not explicitly supplied")
    if args.distance_unit == "mm" and args.metric_space == "resized":
        parser.error("physical mm distances require --metric-space original")
    if args.visualization_max_side < 64:
        parser.error("--visualization-max-side must be at least 64")
    if (
        args.distance_unit == "pixel"
        and spacing_was_provided
        and tuple(args.spacing) != (1.0, 1.0)
    ):
        parser.error("non-unit --spacing requires --distance-unit mm")
    return args


def _read_original_mask(path: str) -> np.ndarray:
    with Image.open(path) as mask:
        return np.asarray(mask.convert("L"), dtype=np.uint8) >= 128


def _checkpoint_configuration(checkpoint: dict) -> tuple[str, int, int, str]:
    training_args = checkpoint.get("args", {})
    backbone = str(training_args.get("backbone", "resnet50"))
    image_size = int(training_args.get("image_size", 256))
    train_seed = int(training_args.get("seed", -1))
    split_dir = str(training_args.get("split_dir", ""))
    return backbone, image_size, train_seed, split_dir


def _checkpoint_method_configuration(checkpoint: dict) -> tuple[str, int, str]:
    """从检查点返回规范方法签名，用于阻止不同方法配置混入同一五折汇总。"""

    training_args = checkpoint.get("args", {})
    if not isinstance(training_args, dict):
        training_args = {}
    model_spec = checkpoint.get("model_spec", {})
    if not isinstance(model_spec, dict):
        model_spec = {}
    method = normalize_compare_method(
        str(
            model_spec.get(
                "compare_method", training_args.get("compare_method", "crcc")
            )
        )
    )
    version = int(
        model_spec.get(
            "version", checkpoint.get("method_implementation_version", -1)
        )
    )
    if method == "crcc":
        defaults: dict[str, object] = {
            "lambda_crcc": 0.05,
            "crcc_momentum": 0.99,
            "crcc_min_separation": 0.1,
            "crcc_calibration_mode": "adaptive",
            "crcc_confidence_weighting": True,
            "warmup_epochs": 3,
            "rampup_epochs": 15,
            "crcc_use_sac": True,
            "crcc_use_crm": True,
            "crcc_use_pra": True,
        }
        parameters = {
            name: training_args.get(name, default)
            for name, default in defaults.items()
        }
    else:
        parameters = training_args.get(
            "method_parameters", model_spec.get("core_parameters", {})
        )
        if not isinstance(parameters, dict):
            raise ValueError("checkpoint method parameters are malformed")
    parameters_json = json.dumps(
        parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return method, version, parameters_json


def _validate_inputs(args: argparse.Namespace) -> List[dict[str, Path | int]]:
    """验证各折清单与检查点来源，并返回逐折评测任务列表。"""

    jobs: List[dict[str, Path | int]] = []
    if args.external_manifest is not None:
        expected_external = args.splits_root / "external_test.tsv"
        if not expected_external.is_file():
            raise FileNotFoundError(
                f"This split root has no registered external_test.tsv: {args.splits_root}"
            )
        if not args.external_manifest.is_file():
            raise FileNotFoundError(f"External manifest not found: {args.external_manifest}")
        if args.external_manifest.resolve() != expected_external.resolve():
            raise ValueError(
                "External provenance mismatch. For this project, --external-manifest must be exactly: "
                f"{expected_external}"
            )
    for fold in args.folds:
        split_dir = split_directory(args.splits_root, fold, args.ratio)
        training_files = (
            split_dir / "train_labeled.tsv",
            split_dir / "train_unlabeled.tsv",
            split_dir / "val.tsv",
            split_dir / "test.tsv",
        )
        missing = [str(path) for path in training_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete fold split:\n" + "\n".join(missing))
        checkpoint = args.run_root / f"fold_{fold}" / args.checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Fold {fold} checkpoint not found: {checkpoint}")
        manifest = args.external_manifest or split_dir / "test.tsv"
        jobs.append({"fold": fold, "split_dir": split_dir, "checkpoint": checkpoint, "manifest": manifest})
    return jobs


def _metric_probability_and_target(
    probability: torch.Tensor,
    resized_target: torch.Tensor,
    mask_path: str,
    metric_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    """按指标空间恢复概率与标签尺寸，并返回单病例 NumPy 数组。"""

    if metric_space == "original":
        original_target = _read_original_mask(mask_path)
        restored = F.interpolate(
            probability[None, None].float(),
            size=original_target.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return restored.cpu().numpy(), original_target
    return probability.cpu().numpy(), (resized_target.cpu().numpy() >= 0.5)


def _case_visualization_name(image_path: str, used_names: set[str]) -> str:
    """生成折内唯一且跨方法稳定的病例文件名。"""

    stem = Path(image_path).stem
    candidate = stem
    if candidate.casefold() in used_names:
        digest = hashlib.sha1(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()[:8]
        candidate = f"{stem}_{digest}"
    used_names.add(candidate.casefold())
    return candidate


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """提取二值区域的两像素轮廓，便于在缩放图像上观察边界误差。"""

    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = binary & ~eroded
    padded_boundary = np.pad(boundary, 1, mode="constant", constant_values=False)
    return (
        padded_boundary[1:-1, 1:-1]
        | padded_boundary[:-2, 1:-1]
        | padded_boundary[2:, 1:-1]
        | padded_boundary[1:-1, :-2]
        | padded_boundary[1:-1, 2:]
    )


def _display_image(image: torch.Tensor) -> Image.Image:
    """把 ImageNet 归一化张量还原为可视化 RGB 图像。"""

    tensor = image.detach().cpu().float()
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    array = (
        tensor.mul(std)
        .add(mean)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _resize_visualization_tiles(
    image: Image.Image,
    target: Image.Image,
    prediction: Image.Image,
    max_side: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """按相同比例缩放图像、GT 与预测，禁止掩膜使用平滑插值。"""

    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == image.size:
        return image, target, prediction
    return (
        image.resize(size, Image.Resampling.LANCZOS),
        target.resize(size, Image.Resampling.NEAREST),
        prediction.resize(size, Image.Resampling.NEAREST),
    )


def _save_case_visualizations(
    image: torch.Tensor,
    resized_target: torch.Tensor,
    probability: torch.Tensor,
    mask_path: str,
    metric_space: str,
    threshold: float,
    fold_root: Path,
    case_name: str,
    max_side: int,
) -> dict[str, str]:
    """保存一例原始指标空间预测和论文筛图所需的两类可视化结果。

    原始预测掩膜与医学指标使用完全相同的尺寸恢复和阈值；叠加图与四联图使用
    DataLoader 已加载的网络输入，不重复解码超大原图。函数返回三个输出文件的
    绝对路径，供逐病例 CSV 追溯。
    """

    probability_cpu = probability.detach().cpu().float()
    target_cpu = resized_target.detach().cpu().float()
    metric_probability, _ = _metric_probability_and_target(
        probability_cpu,
        target_cpu,
        mask_path,
        metric_space,
    )
    metric_prediction = metric_probability >= threshold

    prediction_directory = fold_root / "prediction_masks"
    overlay_directory = fold_root / "overlays"
    panel_directory = fold_root / "comparison_panels"
    for directory in (prediction_directory, overlay_directory, panel_directory):
        directory.mkdir(parents=True, exist_ok=True)

    prediction_path = prediction_directory / f"{case_name}.png"
    overlay_path = overlay_directory / f"{case_name}.jpg"
    panel_path = panel_directory / f"{case_name}.jpg"
    Image.fromarray(metric_prediction.astype(np.uint8) * 255, mode="L").save(
        prediction_path,
        compress_level=1,
    )

    display_image = _display_image(image)
    display_target_array = target_cpu.numpy() >= 0.5
    display_prediction_array = probability_cpu.numpy() >= threshold
    display_target = Image.fromarray(display_target_array.astype(np.uint8) * 255, mode="L")
    display_prediction = Image.fromarray(
        display_prediction_array.astype(np.uint8) * 255,
        mode="L",
    )
    display_image, display_target, display_prediction = _resize_visualization_tiles(
        display_image,
        display_target,
        display_prediction,
        max_side,
    )

    target_array = np.asarray(display_target, dtype=np.uint8) >= 128
    prediction_array = np.asarray(display_prediction, dtype=np.uint8) >= 128
    target_boundary = _mask_boundary(target_array)
    prediction_boundary = _mask_boundary(prediction_array)
    overlap_boundary = target_boundary & prediction_boundary
    overlay_array = np.asarray(display_image, dtype=np.uint8).copy()
    overlay_array[target_boundary & ~prediction_boundary] = (0, 255, 0)
    overlay_array[prediction_boundary & ~target_boundary] = (255, 0, 0)
    overlay_array[overlap_boundary] = (255, 255, 0)
    overlay = Image.fromarray(overlay_array, mode="RGB")
    overlay.save(overlay_path, quality=92)

    target_rgb = Image.merge("RGB", (display_target, display_target, display_target))
    prediction_rgb = Image.merge(
        "RGB",
        (display_prediction, display_prediction, display_prediction),
    )
    tiles = (display_image, target_rgb, prediction_rgb, overlay)
    titles = ("Image", "Ground Truth", "Prediction", "Overlay")
    label_height = 28
    tile_width, tile_height = display_image.size
    panel = Image.new("RGB", (tile_width * len(tiles), tile_height + label_height), "white")
    draw = ImageDraw.Draw(panel)
    for index, (tile, title) in enumerate(zip(tiles, titles)):
        left = index * tile_width
        panel.paste(tile, (left, label_height))
        draw.text((left + 8, 7), title, fill="black")
    panel.save(panel_path, quality=92)

    return {
        "prediction_mask_path": str(prediction_path.resolve()),
        "overlay_path": str(overlay_path.resolve()),
        "comparison_panel_path": str(panel_path.resolve()),
    }


def _initialize_metric_worker() -> None:
    """将每个派生指标进程限制为单线程，使内存和 CPU 占用可预测。"""

    torch.set_num_threads(1)


def _compute_case_payload(
    task: tuple[
        np.ndarray,
        np.ndarray,
        str,
        str,
        float,
        tuple[float, float],
        float,
        int,
        int,
    ]
) -> dict[str, object]:
    """在 CPU 上恢复一例预测并计算全部病例级结果。

    输入为概率、缩放标签、掩码路径和指标协议组成的任务元组；输出包括指标、
    混淆计数、前景统计及概率直方图。本函数不加载模型、不执行网络前向，也不
    负责跨病例或五折汇总。
    """

    (
        probability_array,
        resized_target_array,
        mask_path,
        metric_space,
        threshold,
        spacing,
        surface_tolerance,
        ece_bins,
        histogram_bins,
    ) = task
    probability, target = _metric_probability_and_target(
        torch.from_numpy(probability_array),
        torch.from_numpy(resized_target_array),
        mask_path,
        metric_space,
    )
    metrics = compute_binary_segmentation_metrics(
        probability,
        target,
        threshold=threshold,
        spacing=spacing,
        surface_tolerance=surface_tolerance,
        ece_bins=ece_bins,
    )
    prediction = probability >= threshold
    tp = int(np.count_nonzero(prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    positive_histogram = np.histogram(
        probability[target], bins=histogram_bins, range=(0.0, 1.0)
    )[0]
    negative_histogram = np.histogram(
        probability[~target], bins=histogram_bins, range=(0.0, 1.0)
    )[0]
    return {
        "metrics": metrics,
        "height": int(target.shape[0]),
        "width": int(target.shape[1]),
        "gt_has_foreground": int(bool(target.any())),
        "pred_has_foreground": int(bool(prediction.any())),
        "gt_foreground_pixels": int(target.sum()),
        "pred_foreground_pixels": int(prediction.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "positive_histogram": positive_histogram,
        "negative_histogram": negative_histogram,
    }


@torch.inference_mode()
def evaluate_fold(
    job: dict[str, Path | int],
    args: argparse.Namespace,
    device: torch.device,
    dataset_name: str,
    eval_scope: str,
    metric_executor: Executor | None = None,
    visualization_root: Path | None = None,
) -> tuple[List[dict[str, object]], dict[str, object]]:
    """对一个折的已保存检查点执行完整离线推理与病例级评测。

    ``job`` 提供折号、训练划分、检查点和测试清单，``args`` 提供评测协议，
    ``device`` 指定推理设备；可选 ``metric_executor`` 将逐病例 CPU 指标分发到
    进程池；可选 ``visualization_root`` 保存逐病例预测及对比图。返回值为逐病例
    记录列表和该折的全局汇总字典。本函数只读取冻结检查点并评测，不参与训练、
    模型选择、梯度计算、优化器或 EMA 更新。
    """

    fold = int(job["fold"])
    checkpoint_path = Path(job["checkpoint"])
    manifest = Path(job["manifest"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone, image_size, train_seed, recorded_split = _checkpoint_configuration(checkpoint)
    compare_method, method_version, method_parameters_json = (
        _checkpoint_method_configuration(checkpoint)
    )
    expected_split = Path(job["split_dir"])
    if recorded_split:
        try:
            if Path(recorded_split).resolve() != expected_split.resolve():
                raise ValueError(
                    f"Fold {fold} checkpoint was trained on {recorded_split}, expected {expected_split}"
                )
        except OSError:
            pass

    model = build_inference_model(checkpoint).to(device).eval()
    dataset = LabeledSegmentationDataset(manifest, image_size=image_size, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    amp_enabled = args.amp and device.type == "cuda"
    rows: List[dict[str, object]] = []
    positive_histogram = np.zeros(args.probability_histogram_bins, dtype=np.int64)
    negative_histogram = np.zeros(args.probability_histogram_bins, dtype=np.int64)
    global_tp = global_tn = global_fp = global_fn = 0
    negative_case_count = 0
    negative_case_fp_count = 0
    negative_case_fp_pixels: List[int] = []
    used_visualization_names: set[str] = set()
    fold_visualization_root = (
        visualization_root / f"fold_{fold}" if visualization_root is not None else None
    )
    for batch in tqdm(loader, desc=f"Testing fold {fold}", leave=False, ncols=80):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)["logits"]
        probabilities = torch.sigmoid(logits.float()).cpu()
        targets = batch["mask"]
        metadata: list[tuple[str, str, dict[str, str]]] = []
        tasks = []
        for index in range(images.shape[0]):
            image_path = str(batch["image_path"][index])
            mask_path = str(batch["mask_path"][index])
            visualization_paths = {
                "prediction_mask_path": "",
                "overlay_path": "",
                "comparison_panel_path": "",
            }
            if fold_visualization_root is not None:
                case_name = _case_visualization_name(
                    image_path,
                    used_visualization_names,
                )
                visualization_paths = _save_case_visualizations(
                    image=batch["image"][index],
                    resized_target=targets[index, 0],
                    probability=probabilities[index, 0],
                    mask_path=mask_path,
                    metric_space=args.metric_space,
                    threshold=float(args.threshold),
                    fold_root=fold_visualization_root,
                    case_name=case_name,
                    max_side=int(args.visualization_max_side),
                )
            metadata.append((image_path, mask_path, visualization_paths))
            tasks.append(
                (
                    np.array(probabilities[index, 0].numpy(), dtype=np.float32, copy=True),
                    np.array(targets[index, 0].numpy(), dtype=np.float32, copy=True),
                    mask_path,
                    args.metric_space,
                    float(args.threshold),
                    tuple(args.spacing),
                    float(args.surface_tolerance),
                    int(args.ece_bins),
                    int(args.probability_histogram_bins),
                )
            )
        if metric_executor is None:
            case_results = map(_compute_case_payload, tasks)
        else:
            case_results = metric_executor.map(
                _compute_case_payload, tasks, chunksize=1
            )
        for (image_path, mask_path, visualization_paths), case_result in zip(
            metadata,
            case_results,
        ):
            metrics = dict(case_result["metrics"])
            tp = int(case_result["tp"])
            tn = int(case_result["tn"])
            fp = int(case_result["fp"])
            fn = int(case_result["fn"])
            global_tp += tp
            global_tn += tn
            global_fp += fp
            global_fn += fn
            positive_histogram += np.asarray(case_result["positive_histogram"])
            negative_histogram += np.asarray(case_result["negative_histogram"])
            if not bool(case_result["gt_has_foreground"]):
                negative_case_count += 1
                negative_case_fp_count += int(fp > 0)
                negative_case_fp_pixels.append(fp)
            row: dict[str, object] = {
                "dataset": dataset_name,
                "ratio": args.ratio,
                "train_seed": train_seed,
                "compare_method": compare_method,
                "method_implementation_version": method_version,
                "method_parameters_json": method_parameters_json,
                "fold": fold,
                "eval_scope": eval_scope,
                "case_id": Path(image_path).stem,
                "image_path": image_path,
                "mask_path": mask_path,
                **visualization_paths,
                "manifest": str(manifest),
                "checkpoint": str(checkpoint_path),
                "best_epoch": int(checkpoint.get("epoch", -1)) + 1,
                "backbone": backbone,
                "image_size": image_size,
                "metric_space": args.metric_space,
                "distance_unit": args.distance_unit,
                "spacing_y": args.spacing[0],
                "spacing_x": args.spacing[1],
                "surface_tolerance": args.surface_tolerance,
                "threshold": args.threshold,
                "height": int(case_result["height"]),
                "width": int(case_result["width"]),
                "gt_has_foreground": int(case_result["gt_has_foreground"]),
                "pred_has_foreground": int(case_result["pred_has_foreground"]),
                "gt_foreground_pixels": int(case_result["gt_foreground_pixels"]),
                "pred_foreground_pixels": int(case_result["pred_foreground_pixels"]),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
            row.update(metrics)
            rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pooled_auc, pooled_ap = pooled_auc_ap_from_histograms(
        positive_histogram, negative_histogram
    )
    global_dice_denominator = 2 * global_tp + global_fp + global_fn
    global_union = global_tp + global_fp + global_fn
    mcc_denominator = math.sqrt(
        (global_tp + global_fp)
        * (global_tp + global_fn)
        * (global_tn + global_fp)
        * (global_tn + global_fn)
    )
    positive_rows = [row for row in rows if bool(row["gt_has_foreground"])]
    aggregate: dict[str, object] = {
        "fold": fold,
        "train_seed": train_seed,
        "compare_method": compare_method,
        "method_implementation_version": method_version,
        "method_parameters_json": method_parameters_json,
        "backbone": backbone,
        "image_size": image_size,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest),
        "global_dice": (2 * global_tp / global_dice_denominator) if global_dice_denominator else 1.0,
        "global_iou": (global_tp / global_union) if global_union else 1.0,
        "global_accuracy": (global_tp + global_tn)
        / max(global_tp + global_tn + global_fp + global_fn, 1),
        "global_precision": global_tp / (global_tp + global_fp)
        if global_tp + global_fp
        else float("nan"),
        "global_sensitivity": global_tp / (global_tp + global_fn)
        if global_tp + global_fn
        else float("nan"),
        "global_specificity": global_tn / (global_tn + global_fp)
        if global_tn + global_fp
        else float("nan"),
        "global_mcc": (global_tp * global_tn - global_fp * global_fn) / mcc_denominator
        if mcc_denominator
        else float("nan"),
        "positive_case_dice": float(np.mean([float(row["dice"]) for row in positive_rows]))
        if positive_rows
        else float("nan"),
        "positive_case_iou": float(np.mean([float(row["iou"]) for row in positive_rows]))
        if positive_rows
        else float("nan"),
        "pooled_roc_auc": pooled_auc,
        "pooled_average_precision": pooled_ap,
        "negative_case_fp_rate": negative_case_fp_count / negative_case_count
        if negative_case_count
        else float("nan"),
        "negative_case_mean_fp_pixels": float(np.mean(negative_case_fp_pixels))
        if negative_case_fp_pixels
        else float("nan"),
        "gt_positive_cases": len(positive_rows),
        "gt_negative_cases": negative_case_count,
    }
    return rows, aggregate


def _finite_values(
    rows: Sequence[dict[str, object]], metric: str
) -> np.ndarray:
    values = []
    for row in rows:
        if metric in BOUNDARY_METRICS and not bool(row["gt_has_foreground"]):
            continue
        value = float(row[metric])
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _statistics(
    values: np.ndarray,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2026,
    compute_ci: bool = True,
) -> dict[str, float | int | str]:
    """汇总有限标量并按需计算病例自助法置信区间，返回统计字段字典。"""

    count = int(values.size)
    if count == 0:
        return {
            "n_valid": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "ci_method": "not_available",
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    if count > 1 and compute_ci:
        generator = np.random.default_rng(bootstrap_seed)
        indices = generator.integers(0, count, size=(bootstrap_samples, count))
        bootstrap_means = values[indices].mean(axis=1)
        ci95_low, ci95_high = np.percentile(bootstrap_means, (2.5, 97.5)).tolist()
        ci_method = "case_bootstrap_percentile"
    else:
        # 单病例区间会错误地声称不确定性为零；折均值或模型均值也只报告均值和
        # 标准差，因为它们不是彼此独立的患者观测。
        ci95_low = ci95_high = float("nan")
        ci_method = "not_reported"
    return {
        "n_valid": count,
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "ci_method": ci_method,
    }


def build_summary_rows(
    rows: Sequence[dict[str, object]],
    folds: Sequence[int],
    eval_scope: str,
    fold_aggregates: dict[int, dict[str, object]] | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 2026,
) -> tuple[List[dict[str, object]], List[dict[str, object]]]:
    """把逐病例与逐折全局结果整理为长表和折级宽表。

    输入为病例记录、目标折、评测范围、可选折级汇总及自助法配置；输出为
    ``(summary_rows, fold_rows)``。本函数只执行统计聚合，不读取检查点、不运行
    网络，也不写文件；外层测试与外部测试的合并规则由 ``eval_scope`` 决定。
    """

    fold_aggregates = fold_aggregates or {}
    long_rows: List[dict[str, object]] = []
    fold_wide_rows: List[dict[str, object]] = []
    dataset_name = str(rows[0]["dataset"])
    ratio = str(rows[0]["ratio"])
    train_seed = int(rows[0]["train_seed"])
    fold_metric_means: dict[str, List[float]] = {metric: [] for metric in METRIC_NAMES}
    fold_global_values: dict[str, List[float]] = {
        metric: [] for metric in GLOBAL_METRIC_NAMES
    }

    for fold in folds:
        fold_cases = [row for row in rows if int(row["fold"]) == fold]
        wide: dict[str, object] = {
            "dataset": dataset_name,
            "ratio": ratio,
            "train_seed": train_seed,
            "eval_scope": eval_scope,
            "aggregation": f"fold_{fold}_case_mean",
            "fold": fold,
            "n_rows": len(fold_cases),
            "n_unique_images": len({str(row["image_path"]) for row in fold_cases}),
        }
        common = {
            key: wide[key]
            for key in (
                "dataset",
                "ratio",
                "train_seed",
                "eval_scope",
                "aggregation",
                "fold",
                "n_rows",
                "n_unique_images",
            )
        }
        for metric_index, metric in enumerate(METRIC_NAMES):
            metric_stats = _statistics(
                _finite_values(fold_cases, metric),
                bootstrap_samples,
                bootstrap_seed + fold * 1000 + metric_index,
            )
            wide[metric] = metric_stats["mean"]
            fold_metric_means[metric].append(float(metric_stats["mean"]))
            long_rows.append(
                {
                    **common,
                    "metric": metric,
                    **metric_stats,
                }
            )

        positive_cases = [row for row in fold_cases if bool(row["gt_has_foreground"])]
        for offset, metric in enumerate(("dice", "iou")):
            positive_stats = _statistics(
                _finite_values(positive_cases, metric),
                bootstrap_samples,
                bootstrap_seed + fold * 1000 + 100 + offset,
            )
            long_rows.append(
                {
                    **common,
                    "aggregation": f"fold_{fold}_gt_positive_cases",
                    "n_rows": len(positive_cases),
                    "n_unique_images": len(
                        {str(row["image_path"]) for row in positive_cases}
                    ),
                    "metric": metric,
                    **positive_stats,
                }
            )

        aggregate = fold_aggregates.get(fold, {})
        for metric in GLOBAL_METRIC_NAMES:
            value = float(aggregate.get(metric, float("nan")))
            wide[metric] = value
            fold_global_values[metric].append(value)
            metric_stats = _statistics(
                np.asarray([value], dtype=np.float64)[np.isfinite([value])],
                bootstrap_samples,
                bootstrap_seed,
                compute_ci=False,
            )
            long_rows.append(
                {
                    **common,
                    "aggregation": f"fold_{fold}_global",
                    "metric": metric,
                    **metric_stats,
                }
            )
        fold_wide_rows.append(wide)

    unique_images = len({str(row["image_path"]) for row in rows})
    if eval_scope == "outer_fold":
        aggregation_groups = [("outer_pooled_cases", rows)]
    else:
        aggregation_groups = []
    for aggregation, selected_rows in aggregation_groups:
        for metric_index, metric in enumerate(METRIC_NAMES):
            long_rows.append(
                {
                    "dataset": dataset_name,
                    "ratio": ratio,
                    "train_seed": train_seed,
                    "eval_scope": eval_scope,
                    "aggregation": aggregation,
                    "fold": "all",
                    "n_rows": len(selected_rows),
                    "n_unique_images": unique_images,
                    "metric": metric,
                    **_statistics(
                        _finite_values(selected_rows, metric),
                        bootstrap_samples,
                        bootstrap_seed + 10000 + metric_index,
                    ),
                }
            )
        positive_cases = [row for row in selected_rows if bool(row["gt_has_foreground"])]
        for offset, metric in enumerate(("dice", "iou")):
            long_rows.append(
                {
                    "dataset": dataset_name,
                    "ratio": ratio,
                    "train_seed": train_seed,
                    "eval_scope": eval_scope,
                    "aggregation": "outer_gt_positive_pooled_cases",
                    "fold": "all",
                    "n_rows": len(positive_cases),
                    "n_unique_images": len(
                        {str(row["image_path"]) for row in positive_cases}
                    ),
                    "metric": metric,
                    **_statistics(
                        _finite_values(positive_cases, metric),
                        bootstrap_samples,
                        bootstrap_seed + 11000 + offset,
                    ),
                }
            )

    model_aggregation = "outer_fold_means" if eval_scope == "outer_fold" else "external_model_means"
    mean_wide: dict[str, object] = {
        "dataset": dataset_name,
        "ratio": ratio,
        "train_seed": train_seed,
        "eval_scope": eval_scope,
        "aggregation": model_aggregation,
        "fold": "mean",
        "n_rows": len(rows),
        "n_unique_images": unique_images,
    }
    std_wide = dict(mean_wide)
    std_wide["fold"] = "std"
    for metric in METRIC_NAMES:
        values = np.asarray(fold_metric_means[metric], dtype=np.float64)
        values = values[np.isfinite(values)]
        metric_stats = _statistics(
            values, bootstrap_samples, bootstrap_seed, compute_ci=False
        )
        mean_wide[metric] = metric_stats["mean"]
        std_wide[metric] = metric_stats["std"]
        long_rows.append(
            {
                **{key: mean_wide[key] for key in ("dataset", "ratio", "train_seed", "eval_scope", "aggregation", "fold", "n_rows", "n_unique_images")},
                "metric": metric,
                **metric_stats,
            }
        )
    for metric in GLOBAL_METRIC_NAMES:
        values = np.asarray(fold_global_values[metric], dtype=np.float64)
        values = values[np.isfinite(values)]
        metric_stats = _statistics(
            values, bootstrap_samples, bootstrap_seed, compute_ci=False
        )
        mean_wide[metric] = metric_stats["mean"]
        std_wide[metric] = metric_stats["std"]
        long_rows.append(
            {
                **{
                    key: mean_wide[key]
                    for key in (
                        "dataset",
                        "ratio",
                        "train_seed",
                        "eval_scope",
                        "aggregation",
                        "fold",
                        "n_rows",
                        "n_unique_images",
                    )
                },
                "metric": metric,
                **metric_stats,
            }
        )
    fold_wide_rows.extend([mean_wide, std_wide])
    return long_rows, fold_wide_rows


def build_fivefold_mean_std_rows(
    fold_rows: Sequence[dict[str, object]],
    eval_scope: str,
) -> List[dict[str, object]]:
    """生成面向论文报告的显式五折或五模型均值与标准差表。

    输入为折级宽表及评测范围，输出为每项指标一行的五折记录。标准差始终取五个
    折级标量的样本标准差 ``ddof=1``，绝不是折内病例标准差；不完整或非有限的
    五折结果保持显式缺失，不会被包装成完整五折结果。本函数不进行推理或写盘。
    """

    actual_folds = {
        int(row["fold"]): row
        for row in fold_rows
        if isinstance(row.get("fold"), (int, np.integer))
    }
    if not actual_folds:
        return []
    first = actual_folds[min(actual_folds)]
    std_basis = "across_folds" if eval_scope == "outer_fold" else "across_models"
    rows: List[dict[str, object]] = []
    for metric in (*METRIC_NAMES, *GLOBAL_METRIC_NAMES):
        values_by_fold: dict[int, float] = {}
        for fold in range(1, 6):
            fold_row = actual_folds.get(fold)
            value = float(fold_row.get(metric, float("nan"))) if fold_row else float("nan")
            values_by_fold[fold] = value

        finite_values = np.asarray(
            [value for value in values_by_fold.values() if math.isfinite(value)],
            dtype=np.float64,
        )
        complete = finite_values.size == 5
        mean = float(finite_values.mean()) if complete else float("nan")
        std = float(finite_values.std(ddof=1)) if complete else float("nan")
        rows.append(
            {
                "dataset": first["dataset"],
                "ratio": first["ratio"],
                "train_seed": first["train_seed"],
                "eval_scope": eval_scope,
                "metric": metric,
                **{f"fold_{fold}": values_by_fold[fold] for fold in range(1, 6)},
                "mean_5fold": mean,
                "std_5fold": std,
                "mean_pm_std": f"{mean:.4f} ± {std:.4f}" if complete else "",
                "n_folds_valid": int(finite_values.size),
                "std_basis": std_basis,
                "std_ddof": 1,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    """按指定字段顺序把记录写入带 UTF-8 签名的 CSV 文件。"""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """执行五折离线评测、配置一致性检查和结果文件汇总。

    输入来自命令行指定的划分根目录、各折检查点及评测协议；输出逐病例 CSV、
    汇总 CSV、折级 CSV、五折均值标准差 CSV、协议 JSON，以及可选的逐病例预测
    掩膜与对比图。本函数只评估训练完成的冻结检查点，不启动训练、不选择最佳
    epoch，也不改变 Student、Teacher 或任何优化状态。
    """

    args = parse_args()
    jobs = _validate_inputs(args)
    eval_scope = "external" if args.external_manifest is not None else "outer_fold"
    output_name = f"{eval_scope.split('_')[0]}_{args.metric_space}"
    output_dir = args.output_dir or args.run_root / "metrics" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_root = output_dir / "visualizations" if args.save_visualizations else None
    output_files = {
        "per_case": output_dir / "per_case_metrics.csv",
        "summary": output_dir / "summary_metrics.csv",
        "fold": output_dir / "fold_mean_metrics.csv",
        "fivefold": output_dir / "five_fold_mean_std.csv",
        "protocol": output_dir / "evaluation_protocol.json",
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing test outputs; add --overwrite:\n" + "\n".join(existing)
        )

    device = resolve_device(args.device)
    dataset_name = args.splits_root.name
    all_rows: List[dict[str, object]] = []
    fold_aggregates: dict[int, dict[str, object]] = {}
    metric_executor: ProcessPoolExecutor | None = None
    if args.metric_workers > 0:
        metric_executor = ProcessPoolExecutor(
            max_workers=args.metric_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_initialize_metric_worker,
        )
    try:
        for job in jobs:
            fold_rows, fold_aggregate = evaluate_fold(
                job,
                args,
                device,
                dataset_name,
                eval_scope,
                metric_executor=metric_executor,
                visualization_root=visualization_root,
            )
            all_rows.extend(fold_rows)
            fold_aggregates[int(job["fold"])] = fold_aggregate
    finally:
        if metric_executor is not None:
            metric_executor.shutdown(wait=True, cancel_futures=True)
    if not all_rows:
        raise RuntimeError("No test cases were evaluated")
    configurations = {
        (
            int(aggregate["train_seed"]),
            str(aggregate["backbone"]),
            int(aggregate["image_size"]),
            str(aggregate["compare_method"]),
            int(aggregate["method_implementation_version"]),
            str(aggregate["method_parameters_json"]),
        )
        for aggregate in fold_aggregates.values()
    }
    if len(configurations) != 1:
        raise ValueError(
            "Fold checkpoint configuration mismatch (seed, backbone, image_size, "
            "comparison method, method version, or method parameters): "
            f"{sorted(configurations)}"
        )
    configuration = next(iter(configurations))
    method_metadata = {
        "compare_method": configuration[3],
        "method_implementation_version": configuration[4],
        "method_parameters_json": configuration[5],
    }
    summary_rows, fold_rows = build_summary_rows(
        all_rows,
        args.folds,
        eval_scope,
        fold_aggregates=fold_aggregates,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    fivefold_rows = build_fivefold_mean_std_rows(fold_rows, eval_scope)
    for output_rows in (summary_rows, fold_rows, fivefold_rows):
        for row in output_rows:
            row.update(method_metadata)

    per_case_fields = [
        key for key in all_rows[0].keys() if key not in METRIC_NAMES
    ] + list(METRIC_NAMES)
    summary_fields = [
        "dataset",
        "ratio",
        "train_seed",
        "compare_method",
        "method_implementation_version",
        "method_parameters_json",
        "eval_scope",
        "aggregation",
        "fold",
        "n_rows",
        "n_unique_images",
        "metric",
        "n_valid",
        "mean",
        "std",
        "median",
        "minimum",
        "maximum",
        "ci95_low",
        "ci95_high",
        "ci_method",
    ]
    fold_fields = [
        "dataset",
        "ratio",
        "train_seed",
        "compare_method",
        "method_implementation_version",
        "method_parameters_json",
        "eval_scope",
        "aggregation",
        "fold",
        "n_rows",
        "n_unique_images",
    ] + list(METRIC_NAMES) + list(GLOBAL_METRIC_NAMES)
    fivefold_fields = [
        "dataset",
        "ratio",
        "train_seed",
        "compare_method",
        "method_implementation_version",
        "method_parameters_json",
        "eval_scope",
        "metric",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "fold_5",
        "mean_5fold",
        "std_5fold",
        "mean_pm_std",
        "n_folds_valid",
        "std_basis",
        "std_ddof",
    ]
    write_csv(output_files["per_case"], all_rows, per_case_fields)
    write_csv(output_files["summary"], summary_rows, summary_fields)
    write_csv(output_files["fold"], fold_rows, fold_fields)
    write_csv(output_files["fivefold"], fivefold_rows, fivefold_fields)
    protocol = {
        "evaluation_implementation_version": EVALUATION_IMPLEMENTATION_VERSION,
        "dataset": dataset_name,
        "ratio": args.ratio,
        "folds": args.folds,
        "compare_method": method_metadata["compare_method"],
        "method_implementation_version": method_metadata[
            "method_implementation_version"
        ],
        "method_parameters": json.loads(
            str(method_metadata["method_parameters_json"])
        ),
        "eval_scope": eval_scope,
        "metric_space": args.metric_space,
        "threshold": args.threshold,
        "spacing": list(args.spacing),
        "distance_unit": args.distance_unit,
        "surface_tolerance": args.surface_tolerance,
        "ece_bins": args.ece_bins,
        "probability_histogram_bins": args.probability_histogram_bins,
        "metric_workers": args.metric_workers,
        "save_visualizations": args.save_visualizations,
        "visualization_root": (
            str(visualization_root.resolve()) if visualization_root is not None else None
        ),
        "visualization_max_side": args.visualization_max_side,
        "visualization_outputs": (
            "original/resized metric-space binary mask plus network-space overlay and four-panel image"
            if args.save_visualizations
            else None
        ),
        "metric_parallelism": (
            "spawned per-case CPU process pool"
            if args.metric_workers > 0
            else "sequential per-case CPU metrics"
        ),
        "ranking_metric_implementation": (
            "exact per-case ROC-AUC and step-wise AP from one shared stable pixel sort"
        ),
        "boundary_metric_implementation": (
            "exact joint-foreground bounding-box crop before EDT and contour distance"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "image/case; use a patient-ID extension for multi-slice patients",
        "boundary_summary_rule": "GT-positive cases only; empty predictions receive image-diagonal penalty",
        "hd95_definition": "max of the two directed 95th percentiles",
        "assd_definition": "surface-point-count-weighted mean of both directed distances",
        "surface_dice_definition": "2D contour-length-weighted normalized surface Dice",
        "auc_ap_protocol": (
            "per-case macro plus fold-level pooled "
            f"{args.probability_histogram_bins}-bin probability histograms"
        ),
        "external_summary_rule": "same cases across models; report model mean+SD without a model-level CI",
        "fivefold_sd_rule": (
            "sample SD (ddof=1) across five fold means for outer_fold, "
            "or across five model means for external; incomplete metrics are not formatted as five-fold results"
        ),
        "run_root": str(args.run_root),
        "external_manifest": str(args.external_manifest) if args.external_manifest else None,
        "manifests": [str(job["manifest"]) for job in jobs],
        "checkpoints": [str(job["checkpoint"]) for job in jobs],
        "n_rows": len(all_rows),
        "n_unique_images": len({str(row["image_path"]) for row in all_rows}),
    }
    with output_files["protocol"].open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
    print(f"Per-case CSV: {output_files['per_case']}")
    print(f"Summary CSV:  {output_files['summary']}")
    print(f"Fold CSV:     {output_files['fold']}")
    print(f"Mean+SD CSV:  {output_files['fivefold']}")
    if visualization_root is not None:
        print(f"Visualizations: {visualization_root}")


if __name__ == "__main__":
    main()
