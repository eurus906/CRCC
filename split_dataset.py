"""创建确定性的五折划分，并生成相互嵌套的 5% 与 10% 标注子集。

支持以下平铺数据集布局（目录名示例）：
    dataset/
      images/
      masks/

也支持以下嵌套数据集布局（目录名示例）：
    dataset/
      train/images, train/masks
      val/images,   val/masks
      test/images,  test/masks

图像与掩码按文件名主干配对。验证集始终保留标签，无标注训练清单只写图像路径。
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_DIR_NAMES = {"image", "images", "img", "imgs"}
MASK_DIR_NAMES = {"mask", "masks", "label", "labels", "gt", "gts"}
SPLIT_ALIASES = {
    "train": {"train", "training", "traindata", "trainset"},
    "val": {"val", "valid", "validation", "valdata", "valset"},
    "test": {"test", "testing", "testdata", "testset"},
}


def indexed_files(folder: Path):
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate filename stem in {folder}: {path.stem}")
            result[path.stem] = path.resolve()
    return result


def write_manifest(path: Path, records, include_mask: bool):
    lines = []
    for image, mask in records:
        lines.append(f"{image}\t{mask}" if include_mask else str(image))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_child_directory(folder: Path, accepted_names):
    if not folder.is_dir():
        return None
    children = {
        child.name.casefold(): child
        for child in folder.iterdir()
        if child.is_dir()
    }
    for name in accepted_names:
        if name.casefold() in children:
            return children[name.casefold()]
    return None


def find_split_directory(dataset_root: Path, requested_name: str):
    direct = dataset_root / requested_name
    if direct.is_dir():
        return direct
    aliases = SPLIT_ALIASES.get(requested_name.casefold(), {requested_name.casefold()})
    return find_child_directory(dataset_root, aliases)


def data_directories(folder: Path):
    image_dir = find_child_directory(folder, IMAGE_DIR_NAMES)
    mask_dir = find_child_directory(folder, MASK_DIR_NAMES)
    return image_dir, mask_dir


def paired_directory(folder: Path):
    image_dir, mask_dir = data_directories(folder)
    if image_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"{folder} must contain an image directory "
            f"({sorted(IMAGE_DIR_NAMES)}) and a mask directory "
            f"({sorted(MASK_DIR_NAMES)})"
        )
    images, masks = indexed_files(image_dir), indexed_files(mask_dir)
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(
            f"Unpaired files in {folder}: {len(missing_masks)} images without masks "
            f"and {len(missing_images)} masks without images"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


def ensure_unique_stems(pairs, description: str):
    stems = [image.stem for image, _ in pairs]
    duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(
            f"Duplicate image stems across {description}: {preview}. "
            "Rename them so prediction filenames remain unique."
        )


def foreground_fraction(mask_path: Path) -> float:
    with Image.open(mask_path) as source:
        histogram = source.convert("L").histogram()
    total = sum(histogram)
    return sum(histogram[128:]) / max(total, 1)


def balanced_folds(pairs, fold_count: int, rng: random.Random):
    """按前景占比平衡样本对并返回各折，且不把外层测试标签泄漏到训练。"""

    shuffled = list(pairs)
    rng.shuffle(shuffled)
    ranked = sorted(
        shuffled, key=lambda pair: foreground_fraction(pair[1]), reverse=True
    )
    bins = [[] for _ in range(fold_count)]
    foreground_sums = [0.0] * fold_count
    for pair in ranked:
        area = foreground_fraction(pair[1])
        fold_index = min(
            range(fold_count),
            key=lambda index: (len(bins[index]), foreground_sums[index]),
        )
        bins[fold_index].append(pair)
        foreground_sums[fold_index] += area
    return bins


def stratified_holdout(pairs, count: int):
    """按前景面积均匀选择指定数量的留出集，并返回留出集与剩余样本。"""

    ranked = sorted(pairs, key=lambda pair: foreground_fraction(pair[1]))
    selected_indices = {
        min(len(ranked) - 1, int((index + 0.5) * len(ranked) / count))
        for index in range(count)
    }
    # 正常情况下舍入后的索引互不重复；对极小数据集仍进行防御性补足。
    candidate = 0
    while len(selected_indices) < count:
        if candidate not in selected_indices:
            selected_indices.add(candidate)
        candidate += 1
    holdout = [pair for index, pair in enumerate(ranked) if index in selected_indices]
    remaining = [pair for index, pair in enumerate(ranked) if index not in selected_indices]
    return holdout, remaining


def main():
    """生成五折半监督数据清单及其汇总信息。

    输入来自命令行指定的数据集根目录、折数、随机种子、标注比例和验证比例；
    输出为每折每比例的有标注、无标注、验证和外层测试 TSV，以及 ``summary.json``。
    本函数只负责数据来源发现、配对与确定性划分，不执行图像增强、模型训练、
    检查点选择或测试指标计算。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("splits"), type=Path)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--ratios", default=(5, 10), nargs="+", type=int)
    parser.add_argument(
        "--validation-fraction",
        default=0.1,
        type=float,
        help="Independent fraction of each outer-training set used for checkpoint selection",
    )
    parser.add_argument(
        "--stratify-foreground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance outer folds by mask foreground fraction",
    )
    parser.add_argument(
        "--fold-splits",
        default=("train", "val"),
        nargs="+",
        help="Nested folders combined for five-fold splitting",
    )
    parser.add_argument(
        "--external-split",
        default="test",
        help="Nested folder preserved as an external test set",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    root_image_dir, root_mask_dir = data_directories(dataset_root)
    if root_image_dir is not None and root_mask_dir is not None:
        layout = "flat"
        pairs = paired_directory(dataset_root)
        external_pairs = []
        fold_sources = ["root"]
    else:
        layout = "nested"
        pairs = []
        fold_sources = []
        for split_name in args.fold_splits:
            split_root = find_split_directory(dataset_root, split_name)
            if split_root is not None:
                split_pairs = paired_directory(split_root)
                pairs.extend(split_pairs)
                fold_sources.append(split_root.name)
        # 若缺少常规划分名称，则发现所有直接子数据目录，但排除名称明确表示
        # 测试集的目录。
        if not pairs:
            discovered = []
            for child in dataset_root.iterdir() if dataset_root.is_dir() else []:
                if not child.is_dir():
                    continue
                image_dir, mask_dir = data_directories(child)
                is_test = child.name.casefold() in SPLIT_ALIASES["test"]
                if image_dir is not None and mask_dir is not None and not is_test:
                    discovered.append(child)
            for split_root in sorted(discovered):
                pairs.extend(paired_directory(split_root))
                fold_sources.append(split_root.name)
        if not pairs:
            children = (
                ", ".join(sorted(child.name for child in dataset_root.iterdir()))
                if dataset_root.is_dir()
                else "<dataset root does not exist>"
            )
            raise FileNotFoundError(
                "No usable data folders found under "
                f"{dataset_root}. Detected entries: {children}"
            )
        external_root = find_split_directory(dataset_root, args.external_split)
        external_pairs = (
            paired_directory(external_root) if external_root is not None else []
        )

    ensure_unique_stems(pairs, "five-fold sources")
    ensure_unique_stems(external_pairs, "external test source")
    if len(pairs) < args.folds:
        raise ValueError(f"Need at least {args.folds} paired samples")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in (0, 1)")

    rng = random.Random(args.seed)
    if args.stratify_foreground:
        fold_bins = balanced_folds(pairs, args.folds, rng)
    else:
        rng.shuffle(pairs)
        fold_bins = [pairs[index:: args.folds] for index in range(args.folds)]

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_root": str(dataset_root),
        "layout": layout,
        "fold_sources": fold_sources,
        "external_source": args.external_split if external_pairs else None,
        "external_samples": len(external_pairs),
        "seed": args.seed,
        "folds": args.folds,
        "stratify_foreground": args.stratify_foreground,
        "validation_fraction": args.validation_fraction,
        "samples": len(pairs),
        "ratios": {},
    }
    if external_pairs:
        write_manifest(output / "external_test.tsv", external_pairs, True)

    for fold_index in range(args.folds):
        outer_test = fold_bins[fold_index]
        outer_training = [
            pair
            for index, fold in enumerate(fold_bins)
            if index != fold_index
            for pair in fold
        ]
        validation_count = max(
            2, round(len(outer_training) * args.validation_fraction)
        )
        validation, training = stratified_holdout(
            outer_training, validation_count
        )
        # 使用每折固定的稳定顺序，使 5% 标注集严格成为 10% 标注集的子集。
        fold_rng = random.Random(args.seed + fold_index + 1)
        fold_rng.shuffle(training)

        for ratio in sorted(set(args.ratios)):
            if not 0 < ratio < 100:
                raise ValueError(f"Label ratio must be in (0, 100): {ratio}")
            labeled_count = max(1, math.ceil(len(training) * ratio / 100.0))
            labeled = training[:labeled_count]
            unlabeled = training[labeled_count:]
            ratio_dir = output / f"fold_{fold_index + 1}" / f"ratio_{ratio:02d}"
            ratio_dir.mkdir(parents=True, exist_ok=True)
            write_manifest(ratio_dir / "train_labeled.tsv", labeled, True)
            write_manifest(ratio_dir / "train_unlabeled.tsv", unlabeled, False)
            write_manifest(ratio_dir / "val.tsv", validation, True)
            # 外层测试折只在该组训练全部结束后用于评测。
            write_manifest(ratio_dir / "test.tsv", outer_test, True)
            summary["ratios"][f"fold_{fold_index + 1}_ratio_{ratio:02d}"] = {
                "labeled_budget": len(labeled),
                "train_labeled": len(labeled),
                "independent_validation": len(validation),
                "unlabeled": len(unlabeled),
                "outer_test": len(outer_test),
            }

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
