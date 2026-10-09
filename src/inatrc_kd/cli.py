"""Command-line entrypoint; notebook subprocesses always launch exactly one run."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from .config import load_config, validate_output_location


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InaTRC YOLO26 KD — foundations and Stage B pilots")
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", choices=("teacher", "student"), default="student")
    common.add_argument("--student", choices=("s", "n"), default="n")
    common.add_argument("--method", choices=("none", "native", "crosskd", "csakd"), default="none")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    common.add_argument("--device", default="cpu")
    common.add_argument("--output-root", default="outputs")
    common.add_argument("--dataset-root")
    common.add_argument("--teacher-ckpt")
    common.add_argument("--weights", help="Optional local pretrained YOLO26 .pt for full/offline runs")
    common.add_argument("--expected-images", type=int)
    common.add_argument("--config-dir")
    common.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
    commands.add_parser("run", parents=[common]).add_argument("--upload-checkpoint-each-epoch", action="store_true")
    audit = commands.add_parser("audit", parents=[common])
    audit.add_argument("--ignore-expected-count", action="store_true",
                       help="Explicitly audit an unknown dataset version without the configured total count")
    preflight = commands.add_parser("preflight", parents=[common])
    preflight.add_argument("--require-gpu", action="store_true")
    preflight.add_argument("--models", nargs="+", choices=("m", "s", "n"))
    preflight.add_argument("--native-probe", choices=("s", "n"))
    preflight.add_argument("--custom-probe", choices=("crosskd", "csakd"))
    preflight.add_argument("--probe-amp", action="store_true")
    evaluation = commands.add_parser("evaluate", parents=[common])
    evaluation.add_argument("--checkpoint", required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--output-root", default="outputs")
    summary.add_argument("--include-smoke", action="store_true")
    return parser


def _setup_runtime(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    settings_dir = output / ".ultralytics"
    matplotlib_dir = output / ".matplotlib"
    settings_dir.mkdir(exist_ok=True)
    matplotlib_dir.mkdir(exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output_root).expanduser().resolve()
    validate_output_location(output, getattr(args, "dataset_root", None))
    _setup_runtime(output)
    if args.command == "summarize":
        from .summary import summarize_runs
        print(summarize_runs(output, include_smoke=args.include_smoke))
        return
    if args.command == "preflight":
        from .preflight import environment_report, inspect_model, native_gradient_probe
        result = {"environment": environment_report(args.device, require_gpu=args.require_gpu)}
        if args.models or args.native_probe or args.custom_probe:
            import torch
            torch.set_num_threads(min(4, torch.get_num_threads()))
        if args.models:
            result["models"] = [inspect_model(scale=scale, device=args.device) for scale in args.models]
        if args.native_probe:
            result["native"] = native_gradient_probe(student=args.native_probe, device=args.device, amp=args.probe_amp)
        if args.custom_probe:
            from .kd.probe import custom_gradient_probe
            result["custom"] = custom_gradient_probe(args.custom_probe, student=args.student, device=args.device,
                                                     amp=args.probe_amp, seed=args.seed)
        from .io import write_json
        write_json(output / "preflight.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return
    config = load_config(**{name: getattr(args, name) for name in
                          ("stage", "student", "method", "seed", "mode", "device", "dataset_root",
                           "teacher_ckpt", "output_root", "config_dir", "wandb_mode", "expected_images", "weights")})
    if args.command == "audit":
        if not args.dataset_root:
            raise ValueError("audit requires --dataset-root")
        from .data import audit_dataset
        result = audit_dataset(args.dataset_root, output / f"audit-{uuid.uuid4().hex[:8]}",
                               expected_images=None if args.ignore_expected_count else config["dataset"]["expected_images"])
        print(json.dumps({key: value for key, value in result.items() if key != "bbox_distribution"},
                         indent=2, ensure_ascii=False, allow_nan=False))
    elif args.command == "run":
        if args.upload_checkpoint_each_epoch:
            config["tracking"]["upload_checkpoint_each_epoch"] = True
            config["training"]["save_period"] = 1
        from .train import run_experiment
        result = run_experiment(config)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif args.command == "evaluate":
        if not args.dataset_root:
            raise ValueError("evaluate requires --dataset-root")
        from ultralytics import settings
        settings.update({"wandb": False})
        from .data import audit_dataset, write_runtime_yaml
        from .evaluate import evaluate_checkpoint
        run_dir = output / f"evaluation-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir()
        audit_dataset(args.dataset_root, run_dir / "reports", expected_images=config["dataset"]["expected_images"])
        data_yaml = write_runtime_yaml(args.dataset_root, run_dir / "data_runtime.yaml")
        result = evaluate_checkpoint(args.checkpoint, data_yaml, run_dir, config)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
