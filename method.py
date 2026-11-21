"""干净响应校准一致性（CRCC）的损失函数与外部模块。

整体数据流如下：有标注图像的 Student 预测和真实标签进入干净响应记忆
（CRM），得到背景响应、前景响应及其分离度；无标注弱视图的 EMA Teacher
软概率再结合这些响应进入分离感知校准（SAC），生成校准后的软目标；该软目标
最终约束无标注强视图的 Student 预测。渐进可靠激活（PRA）同时负责响应可靠性
门控以及训练阶段的一致性权重调度。

CRCC 分支只向无标注强视图的 Student logits 反向传播。EMA Teacher、校准软目标、
有标注 Student logits 在 CRM 中的副本以及所有响应统计量均不接收梯度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 当优化语义发生变化、已有 SSL 检查点不再适合安全续训时，递增此版本号。
METHOD_IMPLEMENTATION_VERSION = 5


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """计算二值分割的软 Dice 损失，输入为未经过 Sigmoid 的 logits。"""

    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def supervised_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """以 BCE 与软 Dice 之和约束有标注 Student 预测和真实标签。"""

    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss_with_logits(logits, target)


@dataclass
class ResponseConsistencyDiagnostics:
    """保存 CRCC 的标量诊断量；这些数值只用于日志，不参与损失计算。"""

    valid_images: int
    clean_background_response: float
    clean_foreground_response: float
    background_response: float
    foreground_response: float
    response_separation: float
    calibration_strength: float
    teacher_foreground_fraction: float
    target_foreground_fraction: float
    student_foreground_fraction: float
    mean_confidence: float
    foreground_loss: float
    background_loss: float
    loss: float


class ProgressiveReliabilityActivation(nn.Module):
    """渐进可靠激活（PRA）：共享的响应门控与训练权重调度器。

    ``response_ready`` 使用响应分离度判断 CRM 状态能否更新、SAC 校准能否启用；
    ``training_weight`` 则根据训练进度执行 warm-up 和线性 ramp-up。消融关闭 PRA
    时，响应只保留类别有效性检查，SAC 混合强度不再扣除分离阈值，训练端同时把
    一致性权重固定为一。该模块没有可训练参数，也不保存会随批次更新的张量状态。
    """

    def __init__(
        self,
        minimum_separation: float = 0.1,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_separation < 1.0:
            raise ValueError("minimum_separation must be in (0, 1)")
        self.minimum_separation = minimum_separation
        self.enabled = bool(enabled)

    def response_ready(
        self,
        separation: torch.Tensor,
        initialized: torch.Tensor,
    ) -> torch.Tensor:
        """返回标量布尔张量：状态有效且响应分离度达到阈值时才可靠。"""

        if not self.enabled:
            return initialized.bool()
        return initialized.bool() & (separation >= self.minimum_separation)

    def calibration_strength(self, separation: torch.Tensor) -> torch.Tensor:
        """返回 SAC 混合强度；关闭 PRA 时不再扣除可靠性阈值。"""

        separation = separation.clamp(0.0, 1.0)
        if not self.enabled:
            return separation
        return (
            (separation - self.minimum_separation)
            / (1.0 - self.minimum_separation)
        ).clamp(0.0, 1.0)

    @staticmethod
    def training_weight(
        progress: float,
        warmup_epochs: int,
        rampup_epochs: int,
    ) -> float:
        """返回 CRCC 的调度权重：warm-up 为零，随后线性增至一。"""

        if progress < warmup_epochs:
            return 0.0
        if rampup_epochs <= 0:
            return 1.0
        return float(min(1.0, (progress - warmup_epochs) / rampup_epochs))


@dataclass
class CleanResponseMemoryOutput:
    """CRM 输出的批次响应、记忆响应、响应分离度与初始化状态。"""

    clean_background_response: torch.Tensor
    clean_foreground_response: torch.Tensor
    background_response: torch.Tensor
    foreground_response: torch.Tensor
    separation: torch.Tensor
    initialized: torch.Tensor


class CleanResponseMemory(nn.Module):
    """干净响应记忆（CRM）：从有标注预测中维护类别条件 EMA 响应。

    输入是有标注图像的 Student logits、对应真实标签和 PRA。CRM 在真实背景像素
    上统计 Student 的平均前景概率作为背景响应，在真实前景像素上统计平均前景
    概率作为前景响应。只有当前批次同时具有有效前景和背景，且前景响应与背景
    响应的差达到 PRA 阈值时，才初始化或更新 EMA 记忆。

    三个注册缓冲区 ``background_response_ema``、``foreground_response_ema`` 和
    ``response_initialized`` 会随检查点保存，但全部在无梯度路径中更新。输出既
    包含当前批次的干净响应，也包含供 SAC 使用的记忆响应及其分离度。消融关闭
    CRM 时，模块只返回当前批次响应，并且不读取或更新三个跨批次缓冲区。
    """

    def __init__(self, momentum: float = 0.99, enabled: bool = True) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = momentum
        self.enabled = bool(enabled)
        self.register_buffer(
            "background_response_ema", torch.zeros((), dtype=torch.float32)
        )
        self.register_buffer(
            "foreground_response_ema", torch.ones((), dtype=torch.float32)
        )
        self.register_buffer("response_initialized", torch.zeros((), dtype=torch.bool))

    @staticmethod
    @torch.no_grad()
    def _class_response(
        probability: torch.Tensor,
        class_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """先按图像计算指定类别区域的平均响应，再对有效图像等权平均。"""

        mask = class_mask.float()
        counts = mask.flatten(1).sum(dim=1)
        per_image = (probability * mask).flatten(1).sum(dim=1) / counts.clamp_min(1.0)
        valid = counts > 0
        valid_count = valid.float().sum()
        response = (per_image * valid.float()).sum() / valid_count.clamp_min(1.0)
        response_valid = (valid_count > 0) & torch.isfinite(response)
        return response, response_valid

    @torch.no_grad()
    def forward(
        self,
        labeled_logits: torch.Tensor,
        labels: torch.Tensor,
        activation: ProgressiveReliabilityActivation,
    ) -> CleanResponseMemoryOutput:
        """更新 CRM，并返回供 SAC 使用的无梯度标量响应。

        ``labeled_logits`` 与 ``labels`` 通常均为 ``B_l×1×H×W``。前者先分离
        梯度并经 Sigmoid 转为前景概率；标签以 0.5 为界划分真实前景和背景。
        """

        probability = torch.sigmoid(labeled_logits.detach().float())
        foreground_mask = labels.detach() >= 0.5
        background_mask = ~foreground_mask
        clean_background, background_valid = self._class_response(
            probability, background_mask
        )
        clean_foreground, foreground_valid = self._class_response(
            probability, foreground_mask
        )
        classes_valid = background_valid & foreground_valid
        current_separation = clean_foreground - clean_background
        if not self.enabled:
            # CRM 关闭时只把当前批次响应交给 SAC，不更新或读取任何跨批次状态。
            return CleanResponseMemoryOutput(
                clean_background_response=clean_background,
                clean_foreground_response=clean_foreground,
                background_response=clean_background.detach().float(),
                foreground_response=clean_foreground.detach().float(),
                separation=current_separation.clamp(0.0, 1.0),
                initialized=classes_valid.detach().clone(),
            )
        update_valid = activation.response_ready(
            current_separation,
            initialized=classes_valid,
        )
        initialized_before = self.response_initialized.clone()
        background_candidate = torch.where(
            initialized_before,
            self.background_response_ema * self.momentum
            + clean_background * (1.0 - self.momentum),
            clean_background,
        )
        foreground_candidate = torch.where(
            initialized_before,
            self.foreground_response_ema * self.momentum
            + clean_foreground * (1.0 - self.momentum),
            clean_foreground,
        )
        self.background_response_ema.copy_(
            torch.where(update_valid, background_candidate, self.background_response_ema)
        )
        self.foreground_response_ema.copy_(
            torch.where(update_valid, foreground_candidate, self.foreground_response_ema)
        )
        self.response_initialized.logical_or_(update_valid)

        initialized = self.response_initialized.detach().clone()
        background_response = torch.where(
            initialized,
            self.background_response_ema.detach().float(),
            clean_background.detach().float(),
        )
        foreground_response = torch.where(
            initialized,
            self.foreground_response_ema.detach().float(),
            clean_foreground.detach().float(),
        )
        separation = (foreground_response - background_response).clamp(0.0, 1.0)
        return CleanResponseMemoryOutput(
            clean_background_response=clean_background,
            clean_foreground_response=clean_foreground,
            background_response=background_response,
            foreground_response=foreground_response,
            separation=separation,
            initialized=initialized,
        )


@dataclass
class AdaptiveCalibrationOutput:
    """SAC 输出：校准软目标、标量校准强度以及可靠状态。"""

    target: torch.Tensor
    calibration_strength: torch.Tensor
    ready: torch.Tensor


class SeparationAwareCalibration(nn.Module):
    """分离感知校准（SAC）：利用 CRM 响应校准 Teacher 软概率。

    输入为无标注弱视图的 Teacher 前景概率、CRM 输出和 PRA。背景响应被映射到
    0，前景响应被映射到 1；``fixed`` 模式完全使用该校准结果，``adaptive``
    模式根据响应分离度在原始 Teacher 概率与完全校准结果之间平滑混合，``none``
    模式保留原始 Teacher 概率。CRM 尚未可靠初始化时，输出回退为原始 Teacher
    概率。消融关闭 SAC 时则直接返回原始 Teacher 概率。该模块没有可训练参数或
    随批次更新的状态，并始终运行在无梯度路径。
    """

    def __init__(
        self,
        calibration_mode: str = "adaptive",
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if calibration_mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("calibration_mode must be one of: none, fixed, adaptive")
        self.calibration_mode = calibration_mode
        self.enabled = bool(enabled)

    @torch.no_grad()
    def forward(
        self,
        teacher_probability: torch.Tensor,
        memory: CleanResponseMemoryOutput,
        activation: ProgressiveReliabilityActivation,
    ) -> AdaptiveCalibrationOutput:
        """生成与 ``teacher_probability`` 同形状的校准软目标及标量状态。"""

        teacher = teacher_probability.detach().float().clamp(0.0, 1.0)
        if not self.enabled:
            zero = memory.separation.new_zeros(())
            return AdaptiveCalibrationOutput(
                target=teacher,
                calibration_strength=zero,
                ready=memory.initialized.new_ones((), dtype=torch.bool),
            )
        ready = activation.response_ready(memory.separation, memory.initialized)
        calibrated = (
            (teacher - memory.background_response)
            / memory.separation.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        if self.calibration_mode == "fixed":
            candidate_strength = memory.separation.new_ones(())
            candidate_target = calibrated
        elif self.calibration_mode == "adaptive":
            candidate_strength = activation.calibration_strength(memory.separation)
            candidate_target = (
                (1.0 - candidate_strength) * teacher
                + candidate_strength * calibrated
            ).clamp(0.0, 1.0)
        else:
            candidate_strength = memory.separation.new_zeros(())
            candidate_target = teacher
        calibration_strength = torch.where(
            ready, candidate_strength, memory.separation.new_zeros(())
        )
        target = torch.where(ready, candidate_target, teacher)
        return AdaptiveCalibrationOutput(
            target=target,
            calibration_strength=calibration_strength,
            ready=ready,
        )


class CleanResponseCalibratedConsistency(nn.Module):
    """组合 CRM、SAC 与 PRA，并计算无标注强弱视图一致性损失。

    有标注 Student logits 和真实标签先更新 CRM；无标注弱视图的 Teacher 软概率
    再经 SAC 得到软目标；该软目标最终通过软二元交叉熵约束无标注强视图的
    Student logits。可选置信度权重来自原始 Teacher 概率，而非校准后的目标。

    固定或自适应校准模式仅在对应的响应来源与可选 PRA 已就绪时接纳图像；此外，
    每张图像的有效像素权重总量还必须达到 ``minimum_effective_pixels``。三个
    ``use_*`` 开关只服务于受控消融，默认全开时保持正式 CRCC 数值语义。返回的
    主结果是标量 CRCC 损失，附加诊断量只用于记录。整个模块没有需要优化器更新
    的参数；真正的梯度终点只有 ``student_logits``。
    """

    _LEGACY_MEMORY_KEYS = (
        "background_response_ema",
        "foreground_response_ema",
        "response_initialized",
    )

    def __init__(
        self,
        momentum: float = 0.99,
        minimum_separation: float = 0.1,
        minimum_effective_pixels: float = 8.0,
        calibration_mode: str = "adaptive",
        confidence_weighting: bool = True,
        use_sac: bool = True,
        use_crm: bool = True,
        use_pra: bool = True,
    ) -> None:
        super().__init__()
        if minimum_effective_pixels <= 0.0:
            raise ValueError("minimum_effective_pixels must be positive")
        if use_crm and not use_sac:
            raise ValueError(
                "CRM requires SAC because response memory otherwise cannot affect the target"
            )
        if use_pra and not use_sac:
            raise ValueError(
                "PRA requires SAC because its reliability gate needs a calibrated target"
            )
        self.momentum = momentum
        self.minimum_separation = minimum_separation
        self.minimum_effective_pixels = minimum_effective_pixels
        self.calibration_mode = calibration_mode
        self.confidence_weighting = confidence_weighting
        self.use_sac = bool(use_sac)
        self.use_crm = bool(use_crm)
        self.use_pra = bool(use_pra)
        self.activation = ProgressiveReliabilityActivation(
            minimum_separation,
            enabled=self.use_pra,
        )
        # 即使 CRM 被关闭也保留相同缓冲区结构，确保默认完整方法与旧版检查点兼容。
        self.response_memory = CleanResponseMemory(momentum, enabled=self.use_crm)
        self.calibration = SeparationAwareCalibration(
            calibration_mode,
            enabled=self.use_sac,
        )

    @property
    def background_response_ema(self) -> torch.Tensor:
        return self.response_memory.background_response_ema

    @property
    def foreground_response_ema(self) -> torch.Tensor:
        return self.response_memory.foreground_response_ema

    @property
    def response_initialized(self) -> torch.Tensor:
        return self.response_memory.response_initialized

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # 正式版 v5 检查点将 CRM 缓冲区直接保存在 CRCC 模块下；这里把旧键映射到
        # 重构后的子模块，保持数值状态和训练语义不变。
        for key in self._LEGACY_MEMORY_KEYS:
            legacy_key = prefix + key
            nested_key = prefix + "response_memory." + key
            if legacy_key in state_dict:
                if nested_key not in state_dict:
                    state_dict[nested_key] = state_dict[legacy_key]
                del state_dict[legacy_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        labeled_logits: torch.Tensor,
        labels: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_probability: torch.Tensor,
        return_tensor_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ResponseConsistencyDiagnostics | torch.Tensor]:
        """执行 CRM→SAC→强视图一致性损失的完整 CRCC 前向。

        参数：
            labeled_logits: Student 对有标注图像的 logits，通常为
                ``B_l×1×H×W``；只用于无梯度响应统计。
            labels: 与 ``labeled_logits`` 同形状的二值真实标签。
            student_logits: Student 对无标注强视图的 logits，形状为
                ``B_u×1×H×W``，是 CRCC 唯一接收梯度的输入。
            teacher_probability: EMA Teacher 对同一无标注图像弱视图的软概率，
                与 ``student_logits`` 同形状且不接收梯度。
            return_tensor_diagnostics: 为真时返回便于训练循环聚合的诊断张量；
                否则返回诊断数据类。

        返回：
            标量 CRCC 损失，以及不参与反向传播的诊断信息。
        """

        if labeled_logits.shape != labels.shape:
            raise ValueError("labeled_logits and labels must have identical shapes")
        if student_logits.shape != teacher_probability.shape:
            raise ValueError(
                "student_logits and teacher_probability must have identical shapes"
            )
        if student_logits.ndim != 4 or student_logits.shape[1] != 1:
            raise ValueError("CRCC expects Bx1xHxW binary segmentation tensors")

        if self.use_sac:
            memory = self.response_memory(labeled_logits, labels, self.activation)
        else:
            # Baseline 直接使用 Teacher 软概率，不计算干净响应或改变 CRM 状态。
            zero = teacher_probability.detach().new_zeros((), dtype=torch.float32)
            memory = CleanResponseMemoryOutput(
                clean_background_response=zero,
                clean_foreground_response=zero,
                background_response=zero,
                foreground_response=zero,
                separation=zero,
                initialized=zero.bool(),
            )
        calibrated = self.calibration(
            teacher_probability,
            memory,
            self.activation,
        )
        student_logits_float = student_logits.float()
        with torch.no_grad():
            teacher = teacher_probability.detach().float().clamp(0.0, 1.0)
            student_probability = torch.sigmoid(student_logits_float.detach())
            confidence = (2.0 * teacher - 1.0).abs()
            loss_weight = (
                confidence
                if self.confidence_weighting
                else torch.ones_like(confidence)
            )

        target = calibrated.target
        foreground_term = loss_weight * target * F.softplus(-student_logits_float)
        background_term = (
            loss_weight * (1.0 - target) * F.softplus(student_logits_float)
        )
        confidence_sum = loss_weight.flatten(1).sum(dim=1)
        per_image_foreground = foreground_term.flatten(1).sum(dim=1) / (
            confidence_sum.clamp_min(1e-6)
        )
        per_image_background = background_term.flatten(1).sum(dim=1) / (
            confidence_sum.clamp_min(1e-6)
        )
        valid_image = confidence_sum >= self.minimum_effective_pixels
        if (
            self.use_sac
            and self.use_pra
            and self.calibration_mode in {"fixed", "adaptive"}
        ):
            valid_image = valid_image & calibrated.ready
        valid_count = valid_image.float().sum()
        valid_scale = (valid_count > 0).float()
        foreground_loss = (
            (per_image_foreground * valid_image.float()).sum()
            / valid_count.clamp_min(1.0)
        ) * valid_scale
        background_loss = (
            (per_image_background * valid_image.float()).sum()
            / valid_count.clamp_min(1.0)
        ) * valid_scale
        loss = foreground_loss + background_loss

        diagnostic_tensor = torch.stack(
            (
                valid_count,
                memory.clean_background_response,
                memory.clean_foreground_response,
                memory.background_response,
                memory.foreground_response,
                memory.separation,
                calibrated.calibration_strength,
                (teacher >= 0.5).float().mean(),
                target.mean(),
                (student_probability >= 0.5).float().mean(),
                confidence.mean(),
                foreground_loss.detach(),
                background_loss.detach(),
                loss.detach(),
            )
        ).detach()
        if return_tensor_diagnostics:
            return loss, diagnostic_tensor
        diagnostic_values = diagnostic_tensor.cpu().tolist()
        diagnostics = ResponseConsistencyDiagnostics(
            valid_images=int(diagnostic_values[0]),
            clean_background_response=float(diagnostic_values[1]),
            clean_foreground_response=float(diagnostic_values[2]),
            background_response=float(diagnostic_values[3]),
            foreground_response=float(diagnostic_values[4]),
            response_separation=float(diagnostic_values[5]),
            calibration_strength=float(diagnostic_values[6]),
            teacher_foreground_fraction=float(diagnostic_values[7]),
            target_foreground_fraction=float(diagnostic_values[8]),
            student_foreground_fraction=float(diagnostic_values[9]),
            mean_confidence=float(diagnostic_values[10]),
            foreground_loss=float(diagnostic_values[11]),
            background_loss=float(diagnostic_values[12]),
            loss=float(diagnostic_values[13]),
        )
        return loss, diagnostics
