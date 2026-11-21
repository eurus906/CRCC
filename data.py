"""数据集读取、预加载以及空间对齐的弱/强视图构造。

本文件直接读取 ``split_dataset.py`` 生成的划分清单：

* ``train_labeled.tsv``、``val.tsv``、``test.tsv``：图像路径<TAB>掩码路径；
* ``train_unlabeled.tsv``：仅含图像路径。

有标注样本返回图像及其真实二值掩码，二者始终应用相同的翻转和旋转。无标注
样本由同一幅图像产生弱视图与强视图：两者先共享完全相同的空间变换，强视图
只额外接受颜色、模糊和噪声等外观增强。因此，弱视图 Teacher 预测与强视图
Student 预测在像素位置上天然对齐，无需逆空间变换。

启用共享内存预加载并延迟增强时，本文件返回原始 uint8 张量；对应的批量空间
对齐与弱/强视图生成在 ``train.py`` 的 GPU 增强函数中完成，数据语义保持一致。
"""

from __future__ import annotations

import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path: str | Path, require_mask: bool) -> List[Tuple[str, Optional[str]]]:
    """读取 TSV 清单，并根据是否需要掩码校验列数。"""

    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    records: List[Tuple[str, Optional[str]]] = []
    with manifest.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if require_mask and len(fields) != 2:
                raise ValueError(
                    f"{manifest}:{line_number} must contain image<TAB>mask, got {len(fields)} columns"
                )
            if not require_mask and len(fields) not in (1, 2):
                raise ValueError(
                    f"{manifest}:{line_number} must contain an image path, got {len(fields)} columns"
                )
            image_path = fields[0]
            mask_path = fields[1] if require_mask else None
            records.append((image_path, mask_path))

    if not records:
        raise ValueError(f"Manifest is empty: {manifest}")
    return records


def _load_rgb_uint8(path: str, image_size: int) -> torch.Tensor:
    """读取 RGB 图像，以双线性插值缩放并返回 ``3×H×W`` uint8 张量。"""

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return TF.pil_to_tensor(image)


def _load_mask_uint8(path: str, image_size: int) -> torch.Tensor:
    """读取掩码，以最近邻插值缩放、二值化并返回 ``1×H×W`` uint8 张量。"""

    with Image.open(path) as mask:
        # Kvasir 掩码可能是 RGB JPEG，而 ISIC/DSB 掩码通常已是灰度模式。
        mask = mask.convert("L")
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        tensor = TF.pil_to_tensor(mask)
        return (tensor >= 128).to(torch.uint8)


def _load_rgb(path: str, image_size: int) -> torch.Tensor:
    """读取并缩放一幅 RGB 图像，不创建磁盘缓存。"""

    return _load_rgb_uint8(path, image_size).float().div_(255.0)


def _load_mask(path: str, image_size: int) -> torch.Tensor:
    """读取、最近邻缩放并二值化一个掩码，不创建磁盘缓存。"""

    return _load_mask_uint8(path, image_size).float()


def _preload_shared_uint8(
    records: List[Tuple[str, Optional[str]]],
    image_size: int,
    include_masks: bool,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """一次性读取缩放后的样本并放入 Windows 共享内存，绝不写入磁盘。"""

    images = torch.empty(
        (len(records), 3, image_size, image_size), dtype=torch.uint8
    ).share_memory_()
    masks = (
        torch.empty((len(records), 1, image_size, image_size), dtype=torch.uint8).share_memory_()
        if include_masks
        else None
    )

    def load_one(item: tuple[int, Tuple[str, Optional[str]]]) -> None:
        index, (image_path, mask_path) = item
        images[index].copy_(_load_rgb_uint8(image_path, image_size))
        if masks is not None:
            if mask_path is None:
                raise ValueError("A labeled RAM-preload record is missing its mask")
            masks[index].copy_(_load_mask_uint8(mask_path, image_size))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        deque(executor.map(load_one, enumerate(records)), maxlen=0)
    return images, masks


def _sample_geometry(training: bool) -> Tuple[bool, bool, int]:
    """采样水平翻转、垂直翻转和九十度整数倍旋转参数。"""

    if not training:
        return False, False, 0
    return random.random() < 0.5, random.random() < 0.5, random.randint(0, 3)


def _apply_geometry(
    image: torch.Tensor,
    mask: Optional[torch.Tensor],
    geometry: Tuple[bool, bool, int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """向图像和可选掩码应用同一组空间变换，保持逐像素对齐。"""

    horizontal, vertical, quarter_turns = geometry
    if horizontal:
        image = TF.hflip(image)
        mask = TF.hflip(mask) if mask is not None else None
    if vertical:
        image = TF.vflip(image)
        mask = TF.vflip(mask) if mask is not None else None
    if quarter_turns:
        image = torch.rot90(image, quarter_turns, dims=(-2, -1))
        mask = torch.rot90(mask, quarter_turns, dims=(-2, -1)) if mask is not None else None
    return image, mask


def _strong_appearance(image: torch.Tensor) -> torch.Tensor:
    """执行适度的医学图像外观增强，不依赖额外软件包。"""

    if random.random() < 0.8:
        image = TF.adjust_brightness(image, random.uniform(0.75, 1.25))
        image = TF.adjust_contrast(image, random.uniform(0.75, 1.25))
    if random.random() < 0.5:
        image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
    if random.random() < 0.25:
        image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.1, 1.2])
    if random.random() < 0.5:
        noise_std = random.uniform(0.0, 0.04)
        image = image + torch.randn_like(image) * noise_std
    return image.clamp_(0.0, 1.0)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    """使用 ImageNet 均值和标准差归一化 RGB 图像。"""

    return TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)


class LabeledSegmentationDataset(Dataset):
    """有标注分割数据集，返回空间对齐的图像与真实二值掩码。

    常规路径在样本级同步应用几何增强并返回归一化图像 ``image`` 和掩码
    ``mask``。共享内存延迟增强路径返回 ``image_uint8`` 与 ``mask_uint8``，由
    ``train.py`` 在 GPU 上对整批图像和掩码应用同一组几何参数。
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        training: bool = True,
        preload_to_ram: bool = False,
        preload_workers: int = 20,
        defer_training_augmentation: bool = False,
    ) -> None:
        self.records = read_manifest(manifest, require_mask=True)
        self.image_size = image_size
        self.training = training
        self.defer_training_augmentation = defer_training_augmentation
        self.preloaded_images: torch.Tensor | None = None
        self.preloaded_masks: torch.Tensor | None = None
        if preload_to_ram:
            self.preloaded_images, self.preloaded_masks = _preload_shared_uint8(
                self.records,
                image_size,
                include_masks=True,
                workers=preload_workers,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.records[index]
        assert mask_path is not None
        if self.preloaded_images is None:
            image = _load_rgb(image_path, self.image_size)
            mask = _load_mask(mask_path, self.image_size)
        else:
            assert self.preloaded_masks is not None
            if self.training and self.defer_training_augmentation:
                return {
                    "image_uint8": self.preloaded_images[index],
                    "mask_uint8": self.preloaded_masks[index],
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            image = self.preloaded_images[index].float().div_(255.0)
            mask = self.preloaded_masks[index].float()
        image, mask = _apply_geometry(image, mask, _sample_geometry(self.training))
        assert mask is not None
        return {
            "image": _normalize(image),
            "mask": mask,
            "image_path": image_path,
            "mask_path": mask_path,
        }


class UnlabeledPairDataset(Dataset):
    """无标注数据集，返回同一图像的空间对齐弱视图与强视图。

    常规路径先对原图应用一次共享几何变换，再分别构造仅归一化的 ``weak`` 与
    额外进行外观增强的 ``strong``。共享内存延迟增强路径只返回
    ``image_uint8``，弱/强视图由 ``train.py`` 在 GPU 上按相同原则生成。
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 256,
        preload_to_ram: bool = False,
        preload_workers: int = 20,
        defer_augmentation: bool = False,
    ) -> None:
        self.records = read_manifest(manifest, require_mask=False)
        self.image_size = image_size
        self.defer_augmentation = defer_augmentation
        self.preloaded_images: torch.Tensor | None = None
        if preload_to_ram:
            self.preloaded_images, _ = _preload_shared_uint8(
                self.records,
                image_size,
                include_masks=False,
                workers=preload_workers,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, _ = self.records[index]
        if self.preloaded_images is not None and self.defer_augmentation:
            return {"image_uint8": self.preloaded_images[index], "image_path": image_path}
        image = (
            _load_rgb(image_path, self.image_size)
            if self.preloaded_images is None
            else self.preloaded_images[index].float().div_(255.0)
        )
        image, _ = _apply_geometry(image, None, _sample_geometry(training=True))
        weak = _normalize(image.clone())
        strong = _normalize(_strong_appearance(image.clone()))
        return {"weak": weak, "strong": strong, "image_path": image_path}


def seed_worker(worker_id: int) -> None:
    """为每个 DataLoader 工作进程设置可复现的 Python/NumPy 随机状态。"""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass


def find_split_files(
    split_dir: str | Path,
    require_unlabeled: bool = True,
) -> dict[str, Path]:
    """定位一个折次/标注比例目录中的标准数据划分文件。"""

    root = Path(split_dir)
    files = {
        "labeled": root / "train_labeled.tsv",
        "unlabeled": root / "train_unlabeled.tsv",
        "val": root / "val.tsv",
        "test": root / "test.tsv",
    }
    required_keys = ("labeled", "val", "test")
    if require_unlabeled:
        required_keys = (*required_keys, "unlabeled")
    missing = [str(files[key]) for key in required_keys if not files[key].is_file()]
    if missing:
        raise FileNotFoundError("Missing split files:\n" + "\n".join(missing))
    return files
