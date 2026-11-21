"""顺序调度五折训练，并可在训练完成后运行综合 CSV 评测。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from fold_utils import normalize_ratio, split_directory, validate_folds
from compare.registry import comparison_method_version
from train import TRAINING_PIPELINE_VERSION


FORBIDDEN_FORWARDED_OPTIONS = {
    "--split-dir",
    "--output",
    "--resume",
    "--seed",
    "--test-after-training",
    "--test-manifest",
}

CONTROLLED_TRAIN_FIELDS = {
    "split_dir",
    "output",
    "resume",
    "seed",
    "test_after_training",
    "test_manifest",
}


def _effective_train_configuration(parsed: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(parsed).items()
        if key not in CONTROLLED_TRAIN_FIELDS
    }


def _validate_forwarded_train_arguments(forwarded: Sequence[str]) -> argparse.Namespace:
    """在写入运行元数据前校验要转发给 ``train.py`` 的参数。

    输入为原始参数令牌序列，输出为经 ``train.py`` 参数解析器补全并校验的命名
    空间；本函数只验证参数契约，不创建实验目录、启动训练或改写检查点。
    """

    from train import build_argument_parser, finalize_training_args

    train_parser = build_argument_parser(exit_on_error=False)
    try:
        parsed, unexpected = train_parser.parse_known_args(
            ["--split-dir", "__fivefold_validation__", *forwarded]
        )
    except argparse.ArgumentError as error:
        raise ValueError(f"Invalid forwarded train.py arguments: {error}") from error
    if not unexpected:
        return finalize_training_args(parsed)
    unexpected_text = " ".join(unexpected)
    raise ValueError(
        f"Arguments not recognized by train.py: {unexpected_text}. "
        "Check option spelling and whether an option received too many values."
    )


def _explicit_train_destinations(
    forwarded: Sequence[str],
    train_parser: argparse.ArgumentParser,
) -> set[str]:
    """返回用户在转发令牌中显式填写的训练参数目标名集合。

    输入为原始转发令牌和 ``train.py`` 参数解析器，输出只包含用户主动指定的
    参数目标名。该集合用于区分“应从已有实验恢复的省略参数”和“必须继续执行
    冲突校验的显式参数”，不会解析或修改参数值。
    """

    destinations: set[str] = set()
    option_actions = train_parser._option_string_actions
    for token in forwarded:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


def _serialize_restored_train_action(
    action: argparse.Action,
    value: object,
) -> list[str]:
    """把一个已保存训练参数转换为可重新转发给 ``train.py`` 的令牌。"""

    long_options = [option for option in action.option_strings if option.startswith("--")]
    if not long_options or value is None:
        return []
    positive_option = next(
        (option for option in long_options if not option.startswith("--no-")),
        long_options[0],
    )
    if isinstance(action, argparse.BooleanOptionalAction):
        if bool(value):
            return [positive_option]
        negative_option = next(
            (option for option in long_options if option.startswith("--no-")),
            None,
        )
        if negative_option is None:
            raise ValueError(f"Cannot restore false value for {action.dest}")
        return [negative_option]
    if isinstance(action, argparse._StoreTrueAction):
        return [positive_option] if bool(value) else []
    if isinstance(action, argparse._StoreFalseAction):
        return [positive_option] if not bool(value) else []
    if isinstance(action, argparse._AppendAction):
        if not isinstance(value, list):
            raise ValueError(f"Saved repeatable parameter {action.dest} is not a list")
        restored: list[str] = []
        for item in value:
            restored.extend([positive_option, str(item)])
        return restored
    if action.nargs not in (None, "?"):
        raise ValueError(f"Unsupported saved parameter shape for {action.dest}")
    return [positive_option, str(value)]


def _saved_resume_effective_configuration(
    args: argparse.Namespace,
    train_parser: argparse.ArgumentParser,
) -> dict[str, object] | None:
    """读取本次续跑范围内一致的已保存训练参数。

    输入为五折调度参数和训练参数解析器，输出为第一个可用实验配置中仍由当前
    ``train.py`` 接收的参数。如果多个比例或种子保存了不同训练语义，则在启动
    任何训练或测试前拒绝合并，避免一条命令错误覆盖多组实验来源。
    """

    restorable_destinations = {
        action.dest
        for action in train_parser._actions
        if action.dest not in CONTROLLED_TRAIN_FIELDS and action.dest != "help"
    }
    reference: dict[str, object] | None = None
    reference_path: Path | None = None
    for ratio in args.ratios:
        for seed in args.seeds:
            path = (
                args.output_root
                / ratio
                / f"seed_{seed}"
                / "fivefold_configuration.json"
            )
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                recorded = json.load(handle)
            recorded, _ = _upgrade_legacy_configuration(recorded)
            effective = recorded.get("effective_train_configuration")
            if not isinstance(effective, dict):
                raise ValueError(f"Invalid effective training configuration: {path}")
            projected = {
                key: value
                for key, value in effective.items()
                if key in restorable_destinations
            }
            if reference is None:
                reference = projected
                reference_path = path
                continue
            if projected != reference:
                raise ValueError(
                    "Requested resume configurations use different training parameters: "
                    f"{reference_path} and {path}. Run them in separate commands."
                )
    return reference


def _restore_omitted_resume_arguments(
    args: argparse.Namespace,
    forwarded: Sequence[str],
) -> list[str]:
    """从已有五折配置恢复用户在续跑命令中省略的训练参数。

    用户显式填写的参数始终保持原值，稍后的来源校验仍会拒绝真实冲突；只有未
    填写的参数才从已有配置恢复。因此测试已训练模型时无需重复记忆 compare 方法、
    方法参数或梯度累积设置，未完成折也会沿用原训练语义继续运行。
    """

    if not args.resume_existing:
        return list(forwarded)
    from train import build_argument_parser

    train_parser = build_argument_parser(exit_on_error=False)
    saved = _saved_resume_effective_configuration(args, train_parser)
    if saved is None:
        return list(forwarded)
    explicit = _explicit_train_destinations(forwarded, train_parser)
    restored: list[str] = []
    for action in train_parser._actions:
        if (
            action.dest in CONTROLLED_TRAIN_FIELDS
            or action.dest == "help"
            or action.dest in explicit
            or action.dest not in saved
        ):
            continue
        restored.extend(_serialize_restored_train_action(action, saved[action.dest]))
    return [*restored, *forwarded]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """解析五折调度参数及待转发的训练参数，并返回二者组成的元组。"""

    parser = argparse.ArgumentParser(
        description=(
            "Train fold_1 ... fold_5 sequentially. Unknown options are forwarded to train.py."
        )
    )
    parser.add_argument("--splits-root", type=Path, required=True)
    parser.add_argument(
        "--ratio",
        nargs="+",
        required=True,
        help="One or more ratios trained sequentially: --ratio 1 5",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1337],
        help="Repeat all folds for each training seed",
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--run-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run outer-fold and optional external tests after all folds finish",
    )
    parser.add_argument("--external-manifest", type=Path, default=None)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--test-workers", type=int, default=8)
    parser.add_argument("--test-device", default="auto")
    parser.add_argument("--overwrite-test-results", action="store_true")
    args, forwarded = parser.parse_known_args()
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    try:
        args.ratios = [normalize_ratio(ratio) for ratio in args.ratio]
        args.folds = validate_folds(args.folds)
    except ValueError as error:
        parser.error(str(error))
    if len(args.ratios) != len(set(args.ratios)):
        parser.error("--ratio must not contain duplicate ratios")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must not contain duplicates")
    option_names = {token.split("=", 1)[0] for token in forwarded if token.startswith("--")}
    forbidden = sorted(option_names & FORBIDDEN_FORWARDED_OPTIONS)
    if forbidden:
        parser.error(
            "These options are controlled by train_5fold.py and must be removed: "
            + ", ".join(forbidden)
        )
    try:
        forwarded = _restore_omitted_resume_arguments(args, forwarded)
        forwarded_train_args = _validate_forwarded_train_arguments(forwarded)
    except ValueError as error:
        parser.error(str(error))
    args.forwarded_training_mode = forwarded_train_args.training_mode
    args.forwarded_compare_method = forwarded_train_args.compare_method
    args.effective_train_configuration = _effective_train_configuration(
        forwarded_train_args
    )
    return args, forwarded


def _write_status(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = [
        "ratio",
        "seed",
        "fold",
        "split_dir",
        "run_dir",
        "status",
        "exit_code",
        "checkpoint",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_status(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["fold"]): dict(row) for row in csv.DictReader(handle)}


def _training_configuration(
    args: argparse.Namespace,
    seed: int,
    ratio: str,
) -> dict[str, object]:
    return {
        "splits_root": str(args.splits_root.resolve()),
        "ratio": ratio,
        "folds": args.folds,
        "seed": seed,
        "output_root": str(args.output_root.resolve()),
        "method_implementation_version": (
            comparison_method_version(args.forwarded_compare_method)
            if args.forwarded_training_mode == "ssl"
            else 0
        ),
        "training_pipeline_version": TRAINING_PIPELINE_VERSION,
        "effective_train_configuration": args.effective_train_configuration,
    }


def _upgrade_legacy_configuration(
    configuration: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """规范化旧版转发令牌配置，返回升级后的配置及是否发生变更。"""

    if "effective_train_configuration" in configuration:
        effective = configuration.get("effective_train_configuration")
        if not isinstance(effective, dict):
            return configuration, False
        upgraded = dict(configuration)
        upgraded_effective = dict(effective)
        compatibility_defaults = {
            "compare_method": "crcc",
            "compare_weight": 1.0,
            "gradient_accumulation_steps": 1,
            "method_param": [],
            "method_parameters": {},
            "crcc_use_sac": True,
            "crcc_use_crm": True,
            "crcc_use_pra": True,
        }
        changed = False
        for key, value in compatibility_defaults.items():
            if key not in upgraded_effective:
                upgraded_effective[key] = value
                changed = True
        if changed:
            upgraded["effective_train_configuration"] = upgraded_effective
        return upgraded, changed
    legacy_tokens = configuration.get("forwarded_train_arguments")
    if not isinstance(legacy_tokens, list) or not all(
        isinstance(token, str) for token in legacy_tokens
    ):
        return configuration, False
    parsed = _validate_forwarded_train_arguments(legacy_tokens)
    upgraded = dict(configuration)
    upgraded.pop("forwarded_train_arguments", None)
    upgraded["effective_train_configuration"] = _effective_train_configuration(parsed)
    return upgraded, True


def _align_removed_effective_fields(
    recorded: dict[str, object],
    current: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """在共同有效参数完全一致时，移除旧配置中已退出解析器的多余字段。

    该兼容步骤只允许旧配置多出当前训练器已无法接收的字段；当前配置中的每个
    顶层来源字段和每个有效训练参数都必须在旧配置中存在且取值相同。因此它不会
    放宽 backbone、batch、损失权重、随机种子或方法版本等训练语义保护。
    """

    recorded_effective = recorded.get("effective_train_configuration")
    current_effective = current.get("effective_train_configuration")
    if not isinstance(recorded_effective, dict) or not isinstance(current_effective, dict):
        return recorded, False

    recorded_outer = {
        key: value for key, value in recorded.items() if key != "effective_train_configuration"
    }
    current_outer = {
        key: value for key, value in current.items() if key != "effective_train_configuration"
    }
    if recorded_outer != current_outer:
        return recorded, False
    if any(recorded_effective.get(key) != value for key, value in current_effective.items()):
        return recorded, False

    extra_fields = set(recorded_effective) - set(current_effective)
    if not extra_fields:
        return recorded, False
    aligned = dict(recorded)
    aligned["effective_train_configuration"] = dict(current_effective)
    return aligned, True


def _align_relocated_output_root(
    path: Path,
    recorded: dict[str, object],
    current: dict[str, object],
) -> tuple[dict[str, object], bool]:
    """在实验目录整体迁移且其余来源完全一致时更新保存的输出根目录。

    输入为配置文件实际路径、保存配置和当前配置，输出为对齐后的配置及变更标志。
    只有配置文件确实位于当前 ``output_root/ratio/seed`` 下，并且除输出根目录之外
    的训练参数、数据来源、随机种子和方法版本均一致时才允许迁移。
    """

    if recorded.get("output_root") == current.get("output_root"):
        return recorded, False
    recorded_without_output = {
        key: value for key, value in recorded.items() if key != "output_root"
    }
    current_without_output = {
        key: value for key, value in current.items() if key != "output_root"
    }
    if recorded_without_output != current_without_output:
        return recorded, False
    output_root = current.get("output_root")
    ratio = current.get("ratio")
    seed = current.get("seed")
    if not isinstance(output_root, str) or not isinstance(ratio, str):
        return recorded, False
    expected_path = (
        Path(output_root) / ratio / f"seed_{seed}" / "fivefold_configuration.json"
    )
    if expected_path.resolve() != path.resolve():
        return recorded, False
    aligned = dict(recorded)
    aligned["output_root"] = output_root
    return aligned, True


def _prepare_configuration(
    path: Path,
    configuration: dict[str, object],
    resume_existing: bool,
    directory_was_nonempty: bool,
) -> None:
    """核对续训配置来源并在新实验中写入规范配置。

    输入为配置文件路径、当前配置及续训状态，输出为空；职责限于实验来源一致性
    保护，不负责解析模型、加载权重或执行任何训练步骤。
    """

    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            recorded_configuration = json.load(handle)
        recorded_configuration, was_upgraded = _upgrade_legacy_configuration(
            recorded_configuration
        )
        recorded_configuration, output_root_aligned = _align_relocated_output_root(
            path,
            recorded_configuration,
            configuration,
        )
        recorded_configuration, removed_fields_aligned = _align_removed_effective_fields(
            recorded_configuration,
            configuration,
        )
        was_upgraded = was_upgraded or output_root_aligned or removed_fields_aligned
        if recorded_configuration != configuration:
            raise ValueError(
                f"Resume configuration differs from {path}. "
                "Use a new output directory for changed hyperparameters."
            )
        if was_upgraded:
            temporary_path = path.with_name(path.name + ".tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(recorded_configuration, handle, ensure_ascii=False, indent=2)
            temporary_path.replace(path)
        return
    if resume_existing and directory_was_nonempty:
        raise FileNotFoundError(
            f"Refusing to resume a non-empty legacy directory without {path.name}: {path.parent}. "
            "Its checkpoint provenance cannot be verified; use a new output directory."
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, ensure_ascii=False, indent=2)


def _run_test(
    args: argparse.Namespace,
    seed_root: Path,
    seed: int,
    ratio: str,
    external: bool,
) -> int:
    """为一个比例和随机种子启动独立的 ``test.py`` 评测子进程。

    输入包含调度参数、种子输出目录、种子、比例及内外部评测标志，输出为子进程
    退出码。本函数只拼装和派发命令，不加载模型，也不计算病例级或汇总指标。
    """

    test_script = Path(__file__).with_name("test.py")
    eval_scope = "external" if external else "outer_fold"
    output_name = "external_original" if external else "outer_original"
    command = [
        sys.executable,
        str(test_script),
        "--splits-root",
        str(args.splits_root),
        "--run-root",
        str(seed_root),
        "--ratio",
        ratio,
        "--folds",
        *[str(fold) for fold in args.folds],
        "--output-dir",
        str(seed_root / "metrics" / output_name),
        "--batch-size",
        str(args.test_batch_size),
        "--num-workers",
        str(args.test_workers),
        "--device",
        args.test_device,
    ]
    if external:
        assert args.external_manifest is not None
        command.extend(["--external-manifest", str(args.external_manifest)])
    if args.overwrite_test_results:
        command.append("--overwrite")
    print(f"\nTesting ratio {ratio}, seed {seed} ({eval_scope})")
    return subprocess.run(command, check=False).returncode


def main() -> None:
    """执行完整的多比例、多种子、五折顺序训练调度。

    输入来自本脚本命令行及转发给 ``train.py`` 的参数；输出通过子进程生成各折
    检查点，并写入配置与状态 CSV，按需再调用 ``test.py``。本函数负责来源校验、
    续训决策、失败传播和任务顺序，不实现网络前向、损失计算、优化器更新、EMA
    更新或评测指标公式；这些职责分别属于主训练与测试脚本。
    """

    args, forwarded = parse_args()
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
                "External provenance mismatch. Expected exactly: "
                f"{expected_external}"
            )

    # 在启动可能耗时很长的序列前，先验证所有比例和折的数据文件。
    for ratio in args.ratios:
        for fold in args.folds:
            split_dir = split_directory(args.splits_root, fold, ratio)
            required = [
                split_dir / "train_labeled.tsv",
                split_dir / "val.tsv",
                split_dir / "test.tsv",
            ]
            if args.forwarded_training_mode == "ssl":
                required.append(split_dir / "train_unlabeled.tsv")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError("Incomplete fold split:\n" + "\n".join(missing))

    # 在写入任何配置或状态文件前统一拒绝冲突，避免后续比例的冲突使本次调用
    # 已经启动前面的比例而留下半完成状态。
    seed_was_nonempty: dict[tuple[str, int], bool] = {}
    for ratio in args.ratios:
        ratio_root = args.output_root / ratio
        for seed in args.seeds:
            seed_root = ratio_root / f"seed_{seed}"
            key = (ratio, seed)
            seed_was_nonempty[key] = seed_root.exists() and any(seed_root.iterdir())
            if seed_was_nonempty[key] and not args.resume_existing:
                raise FileExistsError(
                    f"Seed experiment directory is not empty: {seed_root}\n"
                    "Use --resume-existing only with the same training configuration."
                )

    train_script = Path(__file__).with_name("train.py")
    status_paths: list[Path] = []
    for ratio_index, ratio in enumerate(args.ratios, start=1):
        ratio_root = args.output_root / ratio
        ratio_root.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Ratio {ratio_index}/{len(args.ratios)}: {ratio} ===")
        for seed in args.seeds:
            seed_root = ratio_root / f"seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            status_path = seed_root / "fivefold_training_status.csv"
            command_path = seed_root / "fivefold_configuration.json"
            configuration = _training_configuration(args, seed, ratio)
            _prepare_configuration(
                command_path,
                configuration,
                args.resume_existing,
                seed_was_nonempty[(ratio, seed)],
            )
            status_by_fold = _read_status(status_path)
            status_paths.append(status_path)
            for fold in args.folds:
                split_dir = split_directory(args.splits_root, fold, ratio)
                run_dir = seed_root / f"fold_{fold}"
                last_checkpoint = run_dir / "last.pt"
                best_checkpoint = run_dir / "best.pt"
                existing_status = status_by_fold.get(fold, {})
                if (
                    args.resume_existing
                    and existing_status.get("status") == "completed"
                    and best_checkpoint.is_file()
                ):
                    print(
                        f"\nSkipping completed ratio {ratio}, seed {seed}, fold {fold}: "
                        f"{best_checkpoint}"
                    )
                    continue
                command = [
                    sys.executable,
                    str(train_script),
                    "--split-dir",
                    str(split_dir),
                    "--output",
                    str(run_dir),
                    "--seed",
                    str(seed),
                    *forwarded,
                ]
                if args.resume_existing and last_checkpoint.is_file():
                    command.extend(["--resume", str(last_checkpoint)])
                elif args.resume_existing and run_dir.exists() and any(run_dir.iterdir()):
                    raise FileNotFoundError(
                        f"Cannot resume {run_dir}: it contains files but last.pt is missing"
                    )

                print(f"\nTraining ratio {ratio}, seed {seed}, fold {fold}: {split_dir}")
                result = subprocess.run(command, check=False)
                status = (
                    "completed"
                    if result.returncode == 0 and best_checkpoint.is_file()
                    else "failed"
                )
                row = {
                    "ratio": ratio,
                    "seed": seed,
                    "fold": fold,
                    "split_dir": str(split_dir),
                    "run_dir": str(run_dir),
                    "status": status,
                    "exit_code": result.returncode,
                    "checkpoint": str(best_checkpoint) if best_checkpoint.is_file() else "",
                    "error": ""
                    if status == "completed"
                    else "training failed or best.pt was not created",
                }
                status_by_fold[fold] = row
                _write_status(
                    status_path,
                    [status_by_fold[key] for key in sorted(status_by_fold)],
                )
                if status != "completed":
                    raise RuntimeError(
                        f"Ratio {ratio}, seed {seed}, fold {fold} failed with exit code "
                        f"{result.returncode}; status saved to {status_path}"
                    )

            if args.run_test:
                exit_code = _run_test(
                    args, seed_root, seed, ratio, external=False
                )
                if exit_code != 0:
                    raise RuntimeError(
                        f"Outer-fold test failed for ratio {ratio}, seed {seed} "
                        f"(exit code {exit_code})"
                    )
                if args.external_manifest is not None:
                    exit_code = _run_test(
                        args, seed_root, seed, ratio, external=True
                    )
                    if exit_code != 0:
                        raise RuntimeError(
                            f"External test failed for ratio {ratio}, seed {seed} "
                            f"(exit code {exit_code})"
                        )

    print("\nAll requested ratios and folds completed. Status CSV files:")
    for path in status_paths:
        print(path)


if __name__ == "__main__":
    main()
