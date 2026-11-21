"""在任意 ``image<TAB>mask`` 清单上评估已保存的 Student 检查点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import LabeledSegmentationDataset, seed_worker
from model import ResNetUNet
from train import evaluate, resolve_device


def parse_args() -> argparse.Namespace:
    """解析独立评测参数并返回命令行命名空间，不读取或修改训练状态。"""

    parser = argparse.ArgumentParser(description="Evaluate a saved CRCC model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-hd95", action="store_true")
    return parser.parse_args()


def main() -> None:
    """执行单检查点评测。

    输入来自命令行指定的检查点、带标签清单和运行设备；输出为指标字典，
    同时写入 JSON 并打印到终端。本函数复用 ``train.evaluate`` 的验证逻辑，
    但不负责训练调度、损失反向传播、优化器更新或 EMA Teacher 更新。
    """

    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = checkpoint.get("args", {})
    backbone = training_args.get("backbone", "resnet50")
    image_size = int(training_args.get("image_size", 256))

    model = ResNetUNet(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(checkpoint["student"])
    dataset = LabeledSegmentationDataset(args.manifest, image_size=image_size, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    metrics = evaluate(model, loader, device, compute_hd95=args.compute_hd95)

    output = args.output or args.checkpoint.parent / f"{args.manifest.stem}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
