"""Gabungkan YAML di configs/ dengan pilihan pengguna; periksa sebelum training."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def validate_output_location(output_root: str | Path, dataset_root: str | Path | None) -> None:
    """Cegah output/cache ditulis ke dalam dataset asli."""
    if dataset_root is None:
        return
    output = Path(output_root).expanduser().resolve()
    source = Path(dataset_root).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source dataset root")


def load_config(*, config_dir: str | Path | None = None, stage: str = "student", student: str = "n",
                method: str = "none", seed: int = 42, mode: str = "smoke", device: str = "cpu",
                dataset_root: str | None = None, teacher_ckpt: str | None = None,
                output_root: str | Path = "outputs", wandb_mode: str | None = None,
                expected_images: int | None = None, weights: str | None = None) -> dict:
    directory = Path(config_dir or REPO_ROOT / "configs")
    sections = {name: yaml.safe_load((directory / f"{name}.yaml").read_text(encoding="utf-8"))
                for name in ("dataset", "training", "kd", "tracking")}
    presets = sections.pop("training")
    # Smoke memakai preset full sebagai dasar, lalu menimpa parameter dengan nilai ringan.
    training = deepcopy(presets["full"])
    if mode == "smoke":
        training.update(presets["smoke"])
    config = {**sections, "training": training, "evaluation": deepcopy(presets["evaluation"]),
              "stage": stage, "student": student, "method": method, "seed": seed, "mode": mode,
              "device": str(device), "output_root": str(Path(output_root).resolve())}
    if dataset_root is not None:
        config["dataset"]["root"] = str(Path(dataset_root).expanduser().resolve())
    if teacher_ckpt is not None:
        config["kd"]["teacher_checkpoint"] = str(Path(teacher_ckpt).expanduser().resolve())
    if expected_images is not None:
        config["dataset"]["expected_images"] = expected_images
    if wandb_mode is not None:
        config["tracking"]["mode"] = wandb_mode
    for key, variable in (("project", "WANDB_PROJECT"), ("entity", "WANDB_ENTITY")):
        if os.getenv(variable):
            config["tracking"][key] = os.environ[variable]
    if config["tracking"]["upload_checkpoint_each_epoch"]:
        config["training"]["save_period"] = 1
    config["training"]["seed"] = seed
    # YAML berarti model acak untuk diagnosis; .pt berarti pretrained untuk penelitian.
    config["model"] = f"yolo26{'m' if stage == 'teacher' else student}.{'yaml' if mode == 'smoke' else 'pt'}"
    if weights is not None:
        if mode != "full":
            raise ValueError("--weights is for full pretrained runs; smoke uses random YAML models")
        checkpoint = Path(weights).expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ValueError(f"--weights must be an existing pretrained .pt file: {checkpoint}")
        config["model"] = str(checkpoint)
    if mode == "smoke":
        config["evaluation"].update(pilot_latency=False)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    from .data import CLASS_NAMES
    validate_output_location(config["output_root"], config["dataset"]["root"])
    for name, choices in (("stage", ("teacher", "student")), ("student", ("s", "n")),
                          ("method", ("none", "native", "crosskd", "csakd")), ("mode", ("smoke", "full"))):
        if config[name] not in choices:
            raise ValueError(f"Invalid {name}={config[name]!r}; choose {choices}")
    if config["stage"] == "teacher" and config["method"] != "none":
        raise ValueError("Teacher training requires --method none")
    if config["dataset"]["names"] != list(CLASS_NAMES):
        raise ValueError("InaTRC class order must match the reference protocol")
    if config["tracking"]["mode"] not in ("disabled", "offline", "online"):
        raise ValueError("W&B mode must be disabled, offline, or online")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
        raise ValueError("Seed must be a nonnegative integer")
    training = config["training"]
    if training["batch"] < 1 or int(training["batch"]) != training["batch"]:
        raise ValueError("An explicit positive integer batch is required; AutoBatch is disabled")
    if training["epochs"] < 1 or training["imgsz"] % 32:
        raise ValueError("epochs must be positive and imgsz a multiple of 32")
    if config["evaluation"]["split"] != "val" or config["evaluation"]["nms"] is not None:
        raise ValueError("Stage A evaluates val only with the fixed nms=None baseline protocol")
    if config["mode"] == "full" and not config["dataset"]["root"]:
        raise ValueError("Full mode requires --dataset-root; it never uses synthetic data")
