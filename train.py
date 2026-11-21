"""使用 CRCC 半监督流程训练 ResNet-U-Net。

SSL 模式维护结构相同的 Student 与 EMA Teacher。有标注图像和无标注强视图拼成
一个批次进入 Student，随后拆分为有标注 logits 与强视图 logits；无标注弱视图
单独进入不接收梯度的 Teacher。有标注 logits 由 BCE+Dice 监督损失约束，强视图
logits 由 CRCC 生成的 Teacher 校准软目标约束。一次成功的 Student 优化之后，
再执行 Student→Teacher 的 EMA 更新。

示例：
python train.py --split-dir splits_isic/ISIC2018/fold_1/ratio_05 \
    --backbone resnet50 --pretrained --epochs 100 --output runs/isic_r05_f1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterator, Sequence, Sized

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from compare.registry import (
    AVAILABLE_COMPARE_METHODS,
    normalize_compare_method,
    parse_method_parameters,
)
from data import (
    LabeledSegmentationDataset,
    UnlabeledPairDataset,
    find_split_files,
    seed_worker,
)
from method import (
    METHOD_IMPLEMENTATION_VERSION,
    CleanResponseCalibratedConsistency,
    ProgressiveReliabilityActivation,
    supervised_segmentation_loss,
)
from model import ResNetUNet


RAM_PRELOAD_WORKERS = min(20, os.cpu_count() or 1)
TRAINING_PIPELINE_VERSION = 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_argument_parser(exit_on_error: bool = True) -> argparse.ArgumentParser:
    """构建训练命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description=(
            "Semi-supervised binary medical image segmentation with CRCC or a "
            "project-adapted comparison method"
        ),
        exit_on_error=exit_on_error,
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=None,
        help="Optional external image<TAB>mask TSV used instead of split-dir/test.tsv",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/experiment"))
    parser.add_argument(
        "--compare-method",
        choices=AVAILABLE_COMPARE_METHODS,
        default="crcc",
        help="Training method; crcc preserves the original project path",
    )
    parser.add_argument(
        "--compare-weight",
        type=float,
        default=1.0,
        help="Multiplier applied to the selected comparison objective after ramp-up",
    )
    parser.add_argument(
        "--method-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Repeatable comparison-method parameter. VALUE is decoded as JSON when "
            "possible, for example --method-param grid_size=4"
        ),
    )
    parser.add_argument("--backbone", choices=("resnet18", "resnet34", "resnet50"), default="resnet50")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet encoder weights",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--training-mode",
        choices=("ssl", "supervised"),
        default="ssl",
        help=(
            "ssl uses the Student/EMA-Teacher method; supervised trains only the "
            "Student on labeled images and never forwards unlabeled images"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Validation/test batch size used by train.py",
    )
    parser.add_argument(
        "--labeled-batch-size",
        type=int,
        default=32,
        help="Labeled training-stream batch size",
    )
    parser.add_argument(
        "--unlabeled-batch-size",
        type=int,
        default=32,
        help="Unlabeled training-stream batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; shared-RAM/GPU-augmentation mode is fastest with 0 on Windows",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by every DataLoader worker",
    )
    parser.add_argument(
        "--ram-preload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode resized training/validation tensors once into shared RAM; never writes disk cache",
    )
    parser.add_argument(
        "--ram-preload-workers",
        type=int,
        default=RAM_PRELOAD_WORKERS,
        help="CPU threads used only for the one-time shared-RAM preload",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=20,
        help="Optimizer steps per epoch; 0 uses the longer training loader",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--rampup-epochs", type=int, default=15)
    parser.add_argument(
        "--lambda-crcc",
        type=float,
        default=0.05,
        help="Weight of clean-response calibrated soft consistency",
    )
    parser.add_argument("--crcc-momentum", type=float, default=0.99)
    parser.add_argument("--crcc-min-separation", type=float, default=0.1)
    parser.add_argument(
        "--crcc-calibration-mode",
        choices=("none", "fixed", "adaptive"),
        default="adaptive",
        help=(
            "CRCC target ablation: none uses the raw Teacher probability, fixed "
            "uses full clean-response calibration, and adaptive is the proposed "
            "separation-aware blend"
        ),
    )
    parser.add_argument(
        "--crcc-confidence-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Weight CRCC soft consistency by Teacher pixel confidence",
    )
    parser.add_argument(
        "--crcc-use-sac",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable separation-aware calibration; disabling it yields the raw-soft baseline",
    )
    parser.add_argument(
        "--crcc-use-crm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable temporal clean-response memory; SAC otherwise uses current-batch responses",
    )
    parser.add_argument(
        "--crcc-use-pra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reliability gating together with warm-up and ramp-up scheduling",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Comparison-method micro-batches per optimizer step; use this for "
            "dual-network methods when the full physical batch does not fit"
        ),
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument(
        "--validate-teacher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate the EMA Teacher at every validation event in SSL mode",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Validation events without Student Dice improvement; 0 disables early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum Student validation Dice increase counted as an improvement",
    )
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=20,
        help="First 1-based epoch eligible for patience counting; 0 starts immediately",
    )
    parser.add_argument("--compute-hd95", action="store_true")
    parser.add_argument("--test-after-training", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to CUDA channels-last layout after benchmarking this GPU/model",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 CUDA kernels on supported GPUs",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer reproducible kernels; disable only for a controlled speed benchmark",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a compact training progress bar",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def finalize_training_args(args: argparse.Namespace) -> argparse.Namespace:
    """补全兼容默认值，并校验参数之间的联合约束。"""

    if args.labeled_batch_size is None:
        args.labeled_batch_size = args.batch_size
    if args.unlabeled_batch_size is None:
        args.unlabeled_batch_size = args.batch_size

    args.compare_method = normalize_compare_method(args.compare_method)
    args.method_parameters = parse_method_parameters(args.method_param)
    if args.compare_weight < 0.0:
        raise ValueError("--compare-weight must be non-negative")
    if args.compare_method == "crcc" and args.method_parameters:
        raise ValueError("--method-param is only valid with a non-CRCC --compare-method")
    if args.training_mode == "supervised" and args.compare_method != "crcc":
        raise ValueError("comparison methods require --training-mode ssl")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.compare_method == "crcc" and args.gradient_accumulation_steps != 1:
        raise ValueError(
            "gradient accumulation is implemented only by the comparison engine"
        )
    if args.compare_method == "crcc" and args.training_mode == "ssl":
        if args.crcc_use_crm and not args.crcc_use_sac:
            raise ValueError("--crcc-use-crm requires --crcc-use-sac")
        if args.crcc_use_pra and not args.crcc_use_sac:
            raise ValueError("--crcc-use-pra requires --crcc-use-sac")

    if args.lambda_crcc < 0.0:
        raise ValueError("--lambda-crcc must be non-negative")
    for name in (
        "ema_decay",
        "crcc_momentum",
    ):
        value = getattr(args, name)
        if not (0.0 <= value < 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1)")
    if not (0.0 < args.crcc_min_separation < 1.0):
        raise ValueError("--crcc-min-separation must be in (0, 1)")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.labeled_batch_size < 1
        or args.unlabeled_batch_size < 1
    ):
        raise ValueError("--epochs and all batch sizes must be positive")
    if args.steps_per_epoch < 0:
        raise ValueError("--steps-per-epoch must be non-negative")
    if args.warmup_epochs < 0 or args.rampup_epochs < 0:
        raise ValueError("--warmup-epochs and --rampup-epochs must be non-negative")
    if args.validation_interval < 1:
        raise ValueError("--validation-interval must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    if args.early_stopping_start_epoch < 0:
        raise ValueError("--early-stopping-start-epoch must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.prefetch_factor < 1 or args.ram_preload_workers < 1:
        raise ValueError("--prefetch-factor and --ram-preload-workers must be positive")
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        return finalize_training_args(args)
    except ValueError as error:
        parser.error(str(error))
        raise AssertionError("argparse.error should terminate") from error


def set_global_seed(seed: int, deterministic: bool = False, tf32: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


class RepeatedRandomSampler(Sampler[int]):
    """在不重启 DataLoader 的情况下，连续产生新的随机排列。

    低标注比例下，有标注数据流可能远短于无标注数据流。为每条数据流提供覆盖
    整个 epoch 的采样器，可以保持工作进程及其预取队列存活，避免在一个 epoch
    内反复重建迭代器。
    """

    def __init__(self, data_source: Sized, num_samples: int, seed: int) -> None:
        if len(data_source) < 1:
            raise ValueError("RepeatedRandomSampler requires a non-empty dataset")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.data_source = data_source
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        dataset_size = len(self.data_source)
        remaining = self.num_samples
        while remaining > 0:
            permutation = torch.randperm(dataset_size, generator=generator).tolist()
            take = min(remaining, dataset_size)
            yield from permutation[:take]
            remaining -= take

    def __len__(self) -> int:
        return self.num_samples


def build_loaders(args: argparse.Namespace, device: torch.device) -> Dict[str, DataLoader]:
    """构建有标注、无标注、验证和测试数据流。

    SSL 模式同时创建有标注与无标注训练流；纯监督模式不读取无标注训练数据。
    在 CUDA 共享内存模式下，训练增强延迟到批次进入 GPU 后执行。
    """

    split_files = find_split_files(
        args.split_dir,
        require_unlabeled=args.training_mode == "ssl",
    )
    if args.ram_preload:
        print(
            f"Preparing resized data in shared RAM with {args.ram_preload_workers} threads "
            "(no disk cache files are created)"
        )
    datasets = {
        "labeled": LabeledSegmentationDataset(
            split_files["labeled"],
            image_size=args.image_size,
            training=True,
            preload_to_ram=args.ram_preload,
            preload_workers=args.ram_preload_workers,
            defer_training_augmentation=args.ram_preload and device.type == "cuda",
        ),
        "val": LabeledSegmentationDataset(
            split_files["val"],
            image_size=args.image_size,
            training=False,
            preload_to_ram=args.ram_preload,
            preload_workers=args.ram_preload_workers,
        ),
        "test": LabeledSegmentationDataset(
            args.test_manifest or split_files["test"],
            image_size=args.image_size,
            training=False,
        ),
    }
    if args.training_mode == "ssl":
        datasets["unlabeled"] = UnlabeledPairDataset(
            split_files["unlabeled"],
            image_size=args.image_size,
            preload_to_ram=args.ram_preload,
            preload_workers=args.ram_preload_workers,
            defer_augmentation=args.ram_preload and device.type == "cuda",
        )

    if args.training_mode == "ssl":
        steps = args.steps_per_epoch or max(
            math.ceil(len(datasets["labeled"]) / args.labeled_batch_size),

            math.ceil(len(datasets["unlabeled"]) / args.unlabeled_batch_size),
        )
    else:
        steps = args.steps_per_epoch or math.ceil(
            len(datasets["labeled"]) / args.labeled_batch_size
        )
    accumulation = (
        args.gradient_accumulation_steps
        if getattr(args, "compare_method", "crcc") != "crcc"
        else 1
    )
    labeled_sampler = RepeatedRandomSampler(
        datasets["labeled"],
        steps * accumulation * args.labeled_batch_size,
        seed=args.seed,
    )
    common = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    loaders = {
        "labeled": DataLoader(
            datasets["labeled"],
            sampler=labeled_sampler,
            batch_size=args.labeled_batch_size,
            drop_last=True,
            generator=torch.Generator().manual_seed(args.seed + 1000),
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **common,
        ),
    }
    if args.training_mode == "ssl":
        unlabeled_sampler = RepeatedRandomSampler(
            datasets["unlabeled"],
            steps * accumulation * args.unlabeled_batch_size,
            seed=args.seed + 1,
        )
        loaders["unlabeled"] = DataLoader(
            datasets["unlabeled"],
            sampler=unlabeled_sampler,
            batch_size=args.unlabeled_batch_size,
            drop_last=True,
            generator=torch.Generator().manual_seed(args.seed + 1001),
            **common,
        )
    return loaders


def create_ema_teacher(student: nn.Module) -> nn.Module:
    """复制 Student 得到同结构 Teacher，并关闭其梯度与训练态行为。"""

    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema_teacher(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """在 Student 优化成功后，以 EMA 更新 Teacher 参数和浮点缓冲区。

    Teacher 的更新方向为 Student→Teacher。浮点状态按
    ``teacher = decay×teacher + (1-decay)×student`` 更新，非浮点缓冲区直接从
    Student 复制；该过程不建立计算图。
    """

    student_parameters = dict(student.named_parameters())
    teacher_parameter_list = []
    student_parameter_list = []
    for name, teacher_parameter in teacher.named_parameters():
        teacher_parameter_list.append(teacher_parameter)
        student_parameter_list.append(student_parameters[name].detach())
    if (
        teacher_parameter_list
        and teacher_parameter_list[0].is_cuda
        and hasattr(torch, "_foreach_lerp_")
    ):
        torch._foreach_lerp_(teacher_parameter_list, student_parameter_list, 1.0 - decay)
    else:
        for teacher_parameter, student_parameter in zip(
            teacher_parameter_list, student_parameter_list
        ):
            teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)

    student_buffers = dict(student.named_buffers())
    floating_teacher_buffers = []
    floating_student_buffers = []
    for name, teacher_buffer in teacher.named_buffers():
        student_buffer = student_buffers[name].detach()
        if torch.is_floating_point(teacher_buffer):
            floating_teacher_buffers.append(teacher_buffer)
            floating_student_buffers.append(student_buffer)
        else:
            teacher_buffer.copy_(student_buffer)
    if (
        floating_teacher_buffers
        and floating_teacher_buffers[0].is_cuda
        and hasattr(torch, "_foreach_lerp_")
    ):
        torch._foreach_lerp_(
            floating_teacher_buffers,
            floating_student_buffers,
            1.0 - decay,
        )
    else:
        for teacher_buffer, student_buffer in zip(
            floating_teacher_buffers, floating_student_buffers
        ):
            teacher_buffer.mul_(decay).add_(student_buffer, alpha=1.0 - decay)


def consistency_weight(epoch: int, step: int, steps_per_epoch: int, args: argparse.Namespace) -> float:
    """根据 epoch 内连续进度调用 PRA，得到当前 CRCC 调度权重。"""

    if (
        getattr(args, "compare_method", "crcc") == "crcc"
        and not getattr(args, "crcc_use_pra", True)
    ):
        return 1.0
    progress = epoch + step / max(steps_per_epoch, 1)
    return ProgressiveReliabilityActivation.training_weight(
        progress,
        args.warmup_epochs,
        args.rampup_epochs,
    )


def update_early_stopping_state(
    validation_dice: float,
    reference_dice: float,
    bad_validations: int,
    min_delta: float,
) -> tuple[float, int]:
    """更新早停耐心状态，不改变另行记录的真实最佳 Dice。"""

    if validation_dice > reference_dice + min_delta:
        return validation_dice, 0
    return reference_dice, bad_validations + 1


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _move_image(
    image: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> torch.Tensor:
    if channels_last and device.type == "cuda":
        return image.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return image.to(device, non_blocking=True)


def _normalize_batch(image: torch.Tensor) -> torch.Tensor:
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (image - mean) / std


def _batch_geometry_parameters(batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    horizontal = torch.rand(batch_size, device=device) < 0.5
    vertical = torch.rand(batch_size, device=device) < 0.5
    rotations = torch.randint(0, 4, (batch_size,), device=device)
    return horizontal, vertical, rotations


def _apply_batch_geometry(
    image: torch.Tensor,
    horizontal: torch.Tensor,
    vertical: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    output = image.clone()
    if horizontal.any():
        output[horizontal] = torch.flip(output[horizontal], dims=(-1,))
    if vertical.any():
        output[vertical] = torch.flip(output[vertical], dims=(-2,))
    for turns in (1, 2, 3):
        selected = rotations == turns
        if selected.any():
            output[selected] = torch.rot90(output[selected], turns, dims=(-2, -1))
    return output


def _gpu_labeled_batch(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 GPU 上同步增强有标注图像与掩码，保持二者逐像素对齐。"""

    image = batch["image_uint8"].to(device, non_blocking=True).float().div_(255.0)
    mask = batch["mask_uint8"].to(device, non_blocking=True).float()
    horizontal, vertical, rotations = _batch_geometry_parameters(image.shape[0], device)
    image = _apply_batch_geometry(image, horizontal, vertical, rotations)
    mask = _apply_batch_geometry(mask, horizontal, vertical, rotations)
    return _normalize_batch(image), mask


def _gpu_unlabeled_pair(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 GPU 上构造空间对齐的无标注弱视图和强视图。

    两个视图先共享同一组翻转与旋转；弱视图随后只归一化，强视图额外接受颜色、
    饱和度、模糊和噪声增强，因此 Teacher 与 Student 的预测仍具有像素级对应关系。
    """

    image = batch["image_uint8"].to(device, non_blocking=True).float().div_(255.0)
    horizontal, vertical, rotations = _batch_geometry_parameters(image.shape[0], device)
    image = _apply_batch_geometry(image, horizontal, vertical, rotations)
    weak = _normalize_batch(image)
    strong = image.clone()
    batch_size = strong.shape[0]

    color_mask = torch.rand(batch_size, device=device) < 0.8
    brightness = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    contrast = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.75, 1.25)
    adjusted = strong * brightness
    adjusted_mean = adjusted.mean(dim=(-2, -1), keepdim=True)
    adjusted = (adjusted - adjusted_mean) * contrast + adjusted_mean
    strong = torch.where(color_mask[:, None, None, None], adjusted, strong)

    saturation_mask = torch.rand(batch_size, device=device) < 0.5
    saturation = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.8, 1.2)
    gray = (
        strong[:, 0:1] * 0.2989
        + strong[:, 1:2] * 0.5870
        + strong[:, 2:3] * 0.1140
    )
    saturated = gray + (strong - gray) * saturation
    strong = torch.where(saturation_mask[:, None, None, None], saturated, strong)

    blur_mask = torch.rand(batch_size, device=device) < 0.25
    blurred = F.avg_pool2d(strong, kernel_size=3, stride=1, padding=1)
    strong = torch.where(blur_mask[:, None, None, None], blurred, strong)

    noise_mask = torch.rand(batch_size, device=device) < 0.5
    noise_std = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0.0, 0.04)
    noisy = strong + torch.randn_like(strong) * noise_std
    strong = torch.where(noise_mask[:, None, None, None], noisy, strong).clamp_(0.0, 1.0)
    return weak, _normalize_batch(strong)


def _gpu_reserved_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_reserved(device) / (1024**3))


def _training_progress(steps: int, epoch: int, args: argparse.Namespace):
    if not args.progress:
        return range(steps)
    return tqdm(
        range(steps),
        desc=f"Epoch {epoch + 1}/{args.epochs}",
        ncols=80,
        leave=False,
        mininterval=0.2,
    )


def train_one_epoch(
    epoch: int,
    student: nn.Module,
    teacher: nn.Module | None,
    crcc_module: CleanResponseCalibratedConsistency | None,
    loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    """训练一个 epoch，并返回损失、响应和优化步数等平均指标。

    SSL 模式中的单步顺序为：

    1. 有标注图像和无标注强视图拼接后共同前向 Student，再按原批次大小拆分
       ``labeled_logits`` 与 ``strong_logits``；
    2. 空间对齐的无标注弱视图在无梯度路径中前向 EMA Teacher，得到软概率；
    3. ``labeled_logits`` 与真实标签计算 ``L_sup``，同时送入 CRM 更新干净响应；
    4. CRM→SAC 生成 Teacher 校准软目标，并以 ``L_CRCC`` 约束
       ``strong_logits``；
    5. 总损失 ``L_sup + w(t)×lambda_crcc×L_CRCC`` 只反向更新 Student；
    6. Student 优化成功后，最后执行 Student→Teacher 的 EMA 更新。

    warm-up 只令 ``w(t)=0``，并不会跳过无标注前向、CRM/SAC/PRA、CRCC 计算或
    Teacher EMA 更新；因此 warm-up 期间 CRM 仍积累响应，Teacher 仍跟随由监督
    损失更新的 Student。
    """

    student.train()
    ssl_enabled = args.training_mode == "ssl"
    if ssl_enabled:
        if teacher is None or crcc_module is None:
            raise RuntimeError("SSL mode requires the EMA Teacher and CRCC module")
        teacher.eval()
    steps = args.steps_per_epoch or (
        max(len(loaders["labeled"]), len(loaders["unlabeled"]))
        if ssl_enabled
        else len(loaders["labeled"])
    )
    streams = ("labeled", "unlabeled") if ssl_enabled else ("labeled",)
    for stream in streams:
        sampler = loaders[stream].sampler
        if isinstance(sampler, RepeatedRandomSampler):
            sampler.set_epoch(epoch)
    labeled_iterator = iter(loaders["labeled"])
    unlabeled_iterator = iter(loaders["unlabeled"]) if ssl_enabled else None
    amp_enabled = args.amp and device.type == "cuda"
    running = {
        "loss": 0.0,
        "supervised": 0.0,
        "crcc": 0.0,
        "crcc_weighted": 0.0,
        "valid_crcc_images": 0.0,
        "clean_background_response": 0.0,
        "clean_foreground_response": 0.0,
        "background_response": 0.0,
        "foreground_response": 0.0,
        "response_separation": 0.0,
        "calibration_strength": 0.0,
        "teacher_pseudo_foreground_fraction": 0.0,
        "calibrated_target_foreground_fraction": 0.0,
        "student_foreground_fraction": 0.0,
        "crcc_confidence": 0.0,
        "crcc_foreground_loss": 0.0,
        "crcc_background_loss": 0.0,
        "weight": 0.0,
    }
    unlabeled_images_seen = 0
    optimizer_steps = 0

    progress = _training_progress(steps, epoch, args)
    for step in progress:
        labeled_batch = next(labeled_iterator)
        if "image_uint8" in labeled_batch:
            labeled_image, labels = _gpu_labeled_batch(labeled_batch, device)
        else:
            labeled_image = _move_image(labeled_batch["image"], device, args.channels_last)
            labels = labeled_batch["mask"].to(device, non_blocking=True)
        if ssl_enabled:
            assert unlabeled_iterator is not None
            unlabeled_batch = next(unlabeled_iterator)
            if "image_uint8" in unlabeled_batch:
                weak_image, strong_image = _gpu_unlabeled_pair(unlabeled_batch, device)
            else:
                weak_image = _move_image(unlabeled_batch["weak"], device, args.channels_last)
                strong_image = _move_image(unlabeled_batch["strong"], device, args.channels_last)
            unlabeled_images_seen += int(strong_image.shape[0])

        optimizer.zero_grad(set_to_none=True)
        weight = consistency_weight(epoch, step, steps, args) if ssl_enabled else 0.0
        # 即使 warm-up 令 weight 为零，以下 SSL 前向与 CRM 状态更新仍完整执行。
        with _autocast(device, amp_enabled):
            if ssl_enabled:
                # 有标注图像与无标注强视图共用一次 Student 前向，再按批次边界拆分。
                student_batch = torch.cat([labeled_image, strong_image], dim=0)
                student_output = student(student_batch)
                labeled_count = labeled_image.shape[0]
                labeled_logits = student_output["logits"][:labeled_count]
                strong_logits = student_output["logits"][labeled_count:]

                # 空间对齐的无标注弱视图只进入无梯度 EMA Teacher，产生软概率目标源。
                with torch.no_grad():
                    teacher_output = teacher(weak_image)
                    teacher_probability = torch.sigmoid(teacher_output["logits"].float())

                # L_sup 约束有标注预测；CRCC 约束强视图 Student 与校准 Teacher 软目标。
                supervised_loss = supervised_segmentation_loss(labeled_logits, labels)
                crcc_loss, crcc_info = crcc_module(
                    labeled_logits=labeled_logits,
                    labels=labels,
                    student_logits=strong_logits,
                    teacher_probability=teacher_probability,
                    return_tensor_diagnostics=True,
                )
                # PRA 调度权重和 lambda_crcc 只缩放 CRCC 项，不改变监督项。
                loss = supervised_loss + weight * args.lambda_crcc * crcc_loss
            else:
                labeled_output = student(labeled_image)
                supervised_loss = supervised_segmentation_loss(
                    labeled_output["logits"], labels
                )
                crcc_loss = supervised_loss.new_zeros(())
                crcc_info = None
                loss = supervised_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.gradient_clip)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # 若数值溢出导致 GradScaler 跳过 optimizer.step()，其缩放因子会下降；
        # 此时 Student 未更新，因此 Teacher 也不得执行 EMA 更新。
        optimizer_step_succeeded = scaler.get_scale() >= scale_before_step
        if optimizer_step_succeeded:
            optimizer_steps += 1
            if ssl_enabled:
                update_ema_teacher(student, teacher, args.ema_decay)

        running["loss"] += loss.detach().float()
        running["supervised"] += supervised_loss.detach().float()
        running["crcc"] += crcc_loss.detach().float()
        if ssl_enabled:
            assert isinstance(crcc_info, torch.Tensor)
            running["crcc_weighted"] += (
                weight * args.lambda_crcc * crcc_loss.detach().float()
            )
            running["valid_crcc_images"] += crcc_info[0]
            running["clean_background_response"] += crcc_info[1]
            running["clean_foreground_response"] += crcc_info[2]
            running["background_response"] += crcc_info[3]
            running["foreground_response"] += crcc_info[4]
            running["response_separation"] += crcc_info[5]
            running["calibration_strength"] += crcc_info[6]
            running["teacher_pseudo_foreground_fraction"] += crcc_info[7]
            running["calibrated_target_foreground_fraction"] += crcc_info[8]
            running["student_foreground_fraction"] += crcc_info[9]
            running["crcc_confidence"] += crcc_info[10]
            running["crcc_foreground_loss"] += crcc_info[11]
            running["crcc_background_loss"] += crcc_info[12]
        running["weight"] += weight
        if args.progress:
            postfix = {
                "lr": f"{optimizer.param_groups[0]['lr']:.2g}",
                "mem": f"{_gpu_reserved_gb(device):.1f}G",
            }
            if ssl_enabled:
                postfix.update(
                    weight=f"{weight:.2f}",
                )
            progress.set_postfix(postfix, refresh=False)

    running_keys = list(running)
    running_values = torch.stack(
        [torch.as_tensor(running[key], device=device, dtype=torch.float32) for key in running_keys]
    ).cpu().tolist()
    metrics = {
        key: float(value) / steps for key, value in zip(running_keys, running_values)
    }
    metrics["valid_crcc_fraction"] = metrics["valid_crcc_images"] * steps / max(
        unlabeled_images_seen, 1
    )
    metrics["pseudo_foreground_fraction"] = metrics[
        "teacher_pseudo_foreground_fraction"
    ]
    metrics["optimizer_steps"] = float(optimizer_steps)
    return metrics


def _hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(math.hypot(*prediction.shape))
    prediction_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_prediction = distance_transform_edt(~prediction_surface)
    prediction_to_target = distance_to_target[prediction_surface]
    target_to_prediction = distance_to_prediction[target_surface]
    return max(
        float(np.percentile(prediction_to_target, 95)),
        float(np.percentile(target_to_prediction, 95)),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    compute_hd95: bool = False,
    channels_last: bool = False,
) -> dict[str, float]:
    model.eval()
    metric_sums = torch.zeros(5, device=device, dtype=torch.float64)
    hd95_sum = 0.0
    count = 0
    for batch in tqdm(loader, desc="Evaluation", leave=False, ncols=80):
        image = _move_image(batch["image"], device, channels_last)
        target = batch["mask"].to(device, non_blocking=True)
        logits = model(image)["logits"]
        probability = torch.sigmoid(logits)
        prediction = probability >= 0.5
        target_bool = target >= 0.5

        reduce_dims = (1, 2, 3)
        tp = (prediction & target_bool).sum(dim=reduce_dims).float()
        fp = (prediction & ~target_bool).sum(dim=reduce_dims).float()
        fn = (~prediction & target_bool).sum(dim=reduce_dims).float()
        eps = 1e-7
        metric_sums += torch.stack(
            (
                supervised_segmentation_loss(logits, target).double() * image.shape[0],
                ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).sum().double(),
                ((tp + eps) / (tp + fp + fn + eps)).sum().double(),
                ((tp + eps) / (tp + fp + eps)).sum().double(),
                ((tp + eps) / (tp + fn + eps)).sum().double(),
            )
        )
        if compute_hd95:
            for pred_item, target_item in zip(prediction[:, 0], target_bool[:, 0]):
                hd95_sum += _hd95(pred_item.cpu().numpy(), target_item.cpu().numpy())
        count += int(image.shape[0])

    count = max(count, 1)
    loss_sum, dice_sum, iou_sum, precision_sum, recall_sum = metric_sums.cpu().tolist()
    metrics = {
        "loss": loss_sum / count,
        "dice": dice_sum / count,
        "iou": iou_sum / count,
        "precision": precision_sum / count,
        "recall": recall_sum / count,
    }
    if compute_hd95:
        metrics["hd95"] = hd95_sum / count
    return metrics


def save_checkpoint(
    path: Path,
    epoch: int,
    student: nn.Module,
    teacher: nn.Module | None,
    crcc_module: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_dice: float,
    args: argparse.Namespace,
    *,
    best_epoch: int = -1,
    early_stop_best_dice: float = -1.0,
    bad_validations: int = 0,
    early_stopped: bool = False,
) -> None:
    """保存 Student、Teacher、CRM 状态及完整优化和续训元数据。"""

    payload = {
        "checkpoint_version": 5,
        "method_implementation_version": METHOD_IMPLEMENTATION_VERSION,
        "training_pipeline_version": TRAINING_PIPELINE_VERSION,
        "epoch": epoch,
        "student": student.state_dict(),
        "teacher": teacher.state_dict() if teacher is not None else None,
        "crcc_module": crcc_module.state_dict() if crcc_module is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "early_stop_best_dice": early_stop_best_dice,
        "bad_validations": bad_validations,
        "early_stopped": early_stopped,
        "training_mode": args.training_mode,
        "args": vars(args),
    }
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


RESUME_TRAINING_FIELDS = (
    "backbone",
    "pretrained",
    "image_size",
    "epochs",
    "batch_size",
    "labeled_batch_size",
    "unlabeled_batch_size",
    "num_workers",
    "ram_preload",
    "ram_preload_workers",
    "steps_per_epoch",
    "learning_rate",
    "weight_decay",
    "ema_decay",
    "warmup_epochs",
    "rampup_epochs",
    "lambda_crcc",
    "crcc_momentum",
    "crcc_min_separation",
    "crcc_calibration_mode",
    "crcc_confidence_weighting",
    "crcc_use_sac",
    "crcc_use_crm",
    "crcc_use_pra",
    "gradient_clip",
    "validation_interval",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "early_stopping_start_epoch",
    "amp",
    "channels_last",
    "tf32",
    "deterministic",
    "seed",
)


def _normalized_path(value: object) -> str:
    return str(Path(value).resolve()).casefold()


def validate_resume_checkpoint(
    args: argparse.Namespace,
    checkpoint: dict[str, object],
) -> None:
    """拒绝跨实验目录或改变训练配置的不安全续训。"""

    assert args.resume is not None
    resume_path = args.resume.resolve()
    output_path = args.output.resolve()
    if resume_path.name != "last.pt" or resume_path.parent != output_path:
        raise ValueError(
            "--resume must point to last.pt inside the same --output directory. "
            "Forking a checkpoint into a new run is not supported because best.pt "
            "and history provenance would be incomplete."
        )

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        raise ValueError("Checkpoint args metadata is missing or malformed")
    recorded_output = checkpoint_args.get("output")
    if recorded_output is not None and _normalized_path(recorded_output) != _normalized_path(
        args.output
    ):
        raise ValueError("Checkpoint was created in a different output directory")
    recorded_split = checkpoint_args.get("split_dir")
    if recorded_split is not None and _normalized_path(recorded_split) != _normalized_path(
        args.split_dir
    ):
        raise ValueError("Checkpoint split_dir differs from the current --split-dir")

    recorded_mode = str(
        checkpoint.get("training_mode", checkpoint_args.get("training_mode", "ssl"))
    )
    mismatches: list[str] = []
    if recorded_mode != args.training_mode:
        mismatches.append(
            f"training_mode: checkpoint={recorded_mode!r}, current={args.training_mode!r}"
        )
    if args.training_mode == "ssl":
        recorded_method_version = int(checkpoint.get("method_implementation_version", 1))
        if recorded_method_version != METHOD_IMPLEMENTATION_VERSION:
            mismatches.append(
                "method_implementation_version: "
                f"checkpoint={recorded_method_version!r}, "
                f"current={METHOD_IMPLEMENTATION_VERSION!r}"
            )
        recorded_pipeline_version = int(checkpoint.get("training_pipeline_version", 1))
        if recorded_pipeline_version != TRAINING_PIPELINE_VERSION:
            mismatches.append(
                "training_pipeline_version: "
                f"checkpoint={recorded_pipeline_version!r}, "
                f"current={TRAINING_PIPELINE_VERSION!r}"
            )

    legacy_defaults: dict[str, object] = {
        "labeled_batch_size": checkpoint_args.get("batch_size", 8),
        "unlabeled_batch_size": checkpoint_args.get("batch_size", 8),
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "early_stopping_start_epoch": 0,
        "crcc_calibration_mode": "adaptive",
        "crcc_confidence_weighting": True,
        "crcc_use_sac": True,
        "crcc_use_crm": True,
        "crcc_use_pra": True,
    }
    for field in RESUME_TRAINING_FIELDS:
        if field in checkpoint_args:
            recorded = checkpoint_args[field]
        elif field in legacy_defaults:
            recorded = legacy_defaults[field]
        else:
            # 允许加载创建于非关键元数据字段引入之前的检查点；原实现实际使用的
            # 所有字段仍必须存在。
            continue
        current = getattr(args, field)
        if recorded != current:
            mismatches.append(
                f"{field}: checkpoint={recorded!r}, current={current!r}"
            )
    if mismatches:
        raise ValueError(
            "Resume configuration differs from the checkpoint:\n- "
            + "\n- ".join(mismatches)
            + "\nUse the original command or start a new output directory."
        )

    checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
    history_path = output_path / "history.jsonl"
    if checkpoint_epoch > 0:
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Cannot verify resume provenance because history is missing: {history_path}"
            )
        history_lines = [
            line for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if not history_lines:
            raise ValueError("Cannot resume because history.jsonl is empty")
        last_history_epoch = int(json.loads(history_lines[-1])["epoch"])
        if last_history_epoch != checkpoint_epoch:
            raise ValueError(
                f"History ends at epoch {last_history_epoch}, but last.pt is epoch "
                f"{checkpoint_epoch}; refusing to append a mixed or duplicated history."
            )

    if float(checkpoint.get("best_dice", -1.0)) >= 0.0:
        best_path = output_path / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint records a best model but the file is missing: {best_path}"
            )


def prepare_output_directory(output: Path, resume: Path | None) -> None:
    """防止新实验向已有实验目录追加内容或覆盖其结果。"""

    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f"--output is not a directory: {output}")
        if resume is None and any(output.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output}. Use a new --output "
                "directory, or resume its own last.pt explicitly."
            )
    else:
        output.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = parse_args()
    if args.compare_method != "crcc":
        from compare.common.engine import run_comparison_training

        run_comparison_training(args)
        return
    prepare_output_directory(args.output, args.resume)
    set_global_seed(args.seed, deterministic=args.deterministic, tf32=args.tf32)
    device = resolve_device(args.device)

    loaders = build_loaders(args, device)
    student = ResNetUNet(args.backbone, pretrained=args.pretrained).to(device)
    if args.channels_last and device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)
    teacher: nn.Module | None = None
    crcc_module: CleanResponseCalibratedConsistency | None = None
    if args.training_mode == "ssl":
        teacher = create_ema_teacher(student).to(device)
        if args.channels_last and device.type == "cuda":
            teacher = teacher.to(memory_format=torch.channels_last)
        crcc_module = CleanResponseCalibratedConsistency(
            momentum=args.crcc_momentum,
            minimum_separation=args.crcc_min_separation,
            calibration_mode=args.crcc_calibration_mode,
            confidence_weighting=args.crcc_confidence_weighting,
            use_sac=args.crcc_use_sac,
            use_crm=args.crcc_use_crm,
            use_pra=args.crcc_use_pra,
        ).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_dice = 0, -1.0
    best_epoch = -1
    early_stop_best_dice = -1.0
    bad_validations = 0
    resume_was_early_stopped = False

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume_checkpoint(args, checkpoint)
        student.load_state_dict(checkpoint["student"])
        if args.training_mode == "ssl":
            assert teacher is not None and crcc_module is not None

            if checkpoint.get("teacher") is None or checkpoint.get("crcc_module") is None:
                raise ValueError("SSL checkpoint is missing Teacher or CRCC state")
            teacher.load_state_dict(checkpoint["teacher"])
            crcc_module.load_state_dict(checkpoint["crcc_module"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))
        if "best_epoch" in checkpoint:
            best_epoch = int(checkpoint["best_epoch"])
        elif best_dice >= 0.0:
            legacy_best = torch.load(
                args.output / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            best_epoch = int(legacy_best.get("epoch", -1)) + 1
        else:
            best_epoch = -1
        early_stop_best_dice = float(
            checkpoint.get("early_stop_best_dice", best_dice)
        )
        bad_validations = int(checkpoint.get("bad_validations", 0))
        resume_was_early_stopped = bool(checkpoint.get("early_stopped", False))
        print(f"Resumed from epoch {start_epoch}: {args.resume}")

    arguments_path = args.output / "arguments.json"
    if args.resume is None:
        with arguments_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                handle,
                indent=2,
            )
    elif not arguments_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume because the original arguments file is missing: {arguments_path}"
        )

    unlabeled_count = (
        len(loaders["unlabeled"].dataset) if "unlabeled" in loaders else 0
    )
    print(
        f"Device={device} | mode={args.training_mode} | "
        f"labeled={len(loaders['labeled'].dataset)} | "
        f"unlabeled={unlabeled_count} | val={len(loaders['val'].dataset)}"
    )
    print(
        f"Performance | labeled_batch={args.labeled_batch_size} | "
        f"unlabeled_batch={args.unlabeled_batch_size if args.training_mode == 'ssl' else 0} | "
        f"eval_batch={args.batch_size} | workers/stream={args.num_workers} | "
        f"prefetch={args.prefetch_factor if args.num_workers > 0 else 0} | "
        f"AMP={amp_enabled} | TF32={args.tf32 and device.type == 'cuda'} | "
        f"channels_last={args.channels_last and device.type == 'cuda'} | "
        f"deterministic={args.deterministic}"
    )
    history_path = args.output / "history.jsonl"
    if resume_was_early_stopped:
        print(
            "Checkpoint is already marked early-stopped; training is not resumed. "
            "Start a new training run without --resume to use a different stopping policy."
        )
    epoch_range = range(0) if resume_was_early_stopped else range(start_epoch, args.epochs)
    for epoch in epoch_range:
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            epoch,
            student,
            teacher,
            crcc_module,
            loaders,
            optimizer,
            scaler,
            device,
            args,
        )
        if train_metrics["optimizer_steps"] > 0:
            scheduler.step()
        else:
            print(
                f"Epoch {epoch + 1:03d}: optimizer step was skipped by AMP; "
                "the learning-rate scheduler was not advanced."
            )
        record: dict[str, object] = {
            "epoch": epoch + 1,
            "training_mode": args.training_mode,
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation_selection": "student",
        }

        stop_after_epoch = False
        if (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.epochs:
            val_metrics = evaluate(
                student,
                loaders["val"],
                device,
                args.compute_hd95,
                channels_last=args.channels_last,
            )
            record["val"] = val_metrics
            teacher_val_metrics = None
            if args.training_mode == "ssl" and args.validate_teacher:
                assert teacher is not None
                teacher_val_metrics = evaluate(
                    teacher,
                    loaders["val"],
                    device,
                    args.compute_hd95,
                    channels_last=args.channels_last,
                )
            record["val_teacher"] = teacher_val_metrics

            patience_eligible = (
                args.early_stopping_start_epoch == 0
                or epoch + 1 >= args.early_stopping_start_epoch
            )
            if patience_eligible:
                early_stop_best_dice, bad_validations = update_early_stopping_state(
                    val_metrics["dice"],
                    early_stop_best_dice,
                    bad_validations,
                    args.early_stopping_min_delta,
                )

            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                best_epoch = epoch + 1
                save_checkpoint(
                    args.output / "best.pt",
                    epoch,
                    student,
                    teacher,
                    crcc_module,
                    optimizer,
                    scheduler,
                    scaler,
                    best_dice,
                    args,
                    best_epoch=best_epoch,
                    early_stop_best_dice=early_stop_best_dice,
                    bad_validations=bad_validations,
                )
            stop_after_epoch = (
                args.early_stopping_patience > 0
                and patience_eligible
                and bad_validations >= args.early_stopping_patience
            )
            record["early_stopping"] = {
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "start_epoch": args.early_stopping_start_epoch,
                "active": args.early_stopping_patience > 0 and patience_eligible,
                "bad_validations": bad_validations,
                "best_epoch": best_epoch,
                "stopped": stop_after_epoch,
            }
            teacher_text = (
                f", Teacher Dice={teacher_val_metrics['dice']:.4f}"
                if teacher_val_metrics is not None
                else ""
            )
            print(
                f"Epoch {epoch + 1:03d}: loss={train_metrics['loss']:.4f}, "
                f"val Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}"
                f"{teacher_text}"
            )
        else:
            record["early_stopping"] = {
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "start_epoch": args.early_stopping_start_epoch,
                "active": False,
                "bad_validations": bad_validations,
                "best_epoch": best_epoch,
                "stopped": False,
            }

        save_checkpoint(
            args.output / "last.pt",
            epoch,
            student,
            teacher,
            crcc_module,
            optimizer,
            scheduler,
            scaler,
            best_dice,
            args,
            best_epoch=best_epoch,
            early_stop_best_dice=early_stop_best_dice,
            bad_validations=bad_validations,
            early_stopped=stop_after_epoch,
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if stop_after_epoch:
            print(
                f"Early stopping at epoch {epoch + 1}: no Student validation Dice "
                f"improvement for {bad_validations} validation events."
            )
            break

    if args.test_after_training:
        best_checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
        student.load_state_dict(best_checkpoint["student"])
        test_metrics = evaluate(
            student,
            loaders["test"],
            device,
            args.compute_hd95,
            channels_last=args.channels_last,
        )
        with (args.output / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
        print("Test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
