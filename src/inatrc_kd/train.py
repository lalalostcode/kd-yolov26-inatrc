"""Alur utama: cek environment/data → training → muat best.pt → evaluasi → simpan hasil."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds
from ultralytics.utils.torch_utils import unwrap_model

from .config import REPO_ROOT, validate_config
from .data import audit_dataset, create_synthetic_dataset, write_runtime_yaml, verify_dataset_unchanged
from .evaluate import evaluate_checkpoint
from .io import sha256, source_provenance, write_json
from .kd import configure_kd
from .pipeline import SafeDetectionTrainer
from .preflight import environment_report, inspect_model, make_synthetic_teacher, _state_hash
from .tracking import EpochRecorder, ExperimentTracker


def run_experiment(config: dict) -> dict:
    validate_config(config)
    # Logging W&B hanya melalui tracker manual, agar tidak terbentuk dua run.
    settings.update({"wandb": False})
    output = Path(config["output_root"])
    scale = "m" if config["stage"] == "teacher" else config["student"]
    name = (f"{config['stage']}-yolo26{scale}-{config['method']}-seed{config['seed']}-{config['mode']}-"
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}")
    run_dir = output / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "reports").mkdir()
    started = time.perf_counter()
    tracker = ExperimentTracker(config, run_dir)
    metadata = {"status": "preflight", "nonfinal": config["mode"] == "smoke",
                "test_evaluated": False, "source": source_provenance(REPO_ROOT),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": config["stage"], "student": config["student"], "method": config["method"],
                "seed": config["seed"], "mode": config["mode"], "run_dir": str(run_dir)}
    provenance_file = os.getenv("INATRC_SOURCE_PROVENANCE")
    if provenance_file:
        metadata["notebook_source"] = json.loads(Path(provenance_file).read_text(encoding="utf-8"))
    try:
        environment = environment_report(config["device"], require_gpu=config["device"] != "cpu")
        metadata["environment"] = environment
        if config["device"] == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        if config["dataset"]["root"]:
            actual_audit = audit_dataset(config["dataset"]["root"], run_dir / "reports" / "inatrc",
                                         expected_images=config["dataset"]["expected_images"])
            metadata["inatrc_dataset_fingerprint"] = actual_audit["fingerprint"]
        # Smoke selalu memakai data buatan; dataset asli yang terpasang tetap diaudit.
        if config["mode"] == "smoke":
            data_root = create_synthetic_dataset(run_dir / "synthetic_data", seed=config["seed"])
            audit = audit_dataset(data_root, run_dir / "reports" / "training_data", expected_images=20)
            config["initialization"] = "random_yaml_diagnostics_only"
        else:
            data_root = Path(config["dataset"]["root"])
            audit = actual_audit
            config["initialization"] = "pretrained"
        config["training_dataset_root"] = str(data_root)
        config["dataset_fingerprint"] = audit["fingerprint"]
        data_yaml = write_runtime_yaml(data_root, run_dir / "data_runtime.yaml", audit=audit)
        init_seeds(config["seed"], deterministic=config["training"]["deterministic"])
        # Teacher acak hanya alat diagnosis smoke, bukan teacher penelitian.
        if config["method"] != "none" and config["mode"] == "smoke" and not config["kd"]["teacher_checkpoint"]:
            config["kd"]["teacher_checkpoint"] = str(make_synthetic_teacher(run_dir / "diagnostic_teacher.pt", seed=config["seed"]))
        kd_args = configure_kd(config)
        metadata["architecture_probe"] = inspect_model(
            scale=scale, imgsz=config["training"]["imgsz"], device=config["device"],
            batch_size=config["training"]["batch"], amp=config["training"]["amp"],
        )
        model = YOLO(config["model"])
        if config["mode"] == "full":
            yaml_file = Path(str(model.model.yaml.get("yaml_file", ""))).stem
            if yaml_file not in ("yolo26", f"yolo26{scale}") or model.model.yaml.get("scale") != scale:
                raise ValueError(f"Initial pretrained weights must match YOLO26{scale}")
            if int(model.model.model[-1].nc) != 80:
                raise ValueError("Research initialization must use COCO pretrained weights, not a fine-tuned baseline checkpoint")
            checkpoint = Path(model.ckpt_path)
            metadata["initial_checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)}
        if kd_args:
            metadata["teacher_checkpoint_sha256"] = sha256(kd_args["distill_model"])
        train_args = {**config["training"], **kd_args, "data": str(data_yaml),
                      "device": config["device"], "project": str(output), "name": name, "exist_ok": True,
                      "pretrained": config["mode"] == "full", "nms": None,
                      "conf": config["evaluation"]["conf"], "iou": config["evaluation"]["iou"],
                      "max_det": config["evaluation"]["max_det"], "augment": False}
        config["ultralytics_effective_config"] = vars(get_cfg(overrides=train_args))
        # Simpan parameter yang benar-benar dipakai, termasuk default dari Ultralytics.
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        write_json(run_dir / "metadata.json", metadata)
        tracker.start(metadata)
        recorder = EpochRecorder(config, run_dir, tracker)
        recorder.attach(model)

        def record_runtime_initialization(trainer):
            # Hash awal membantu membandingkan inisialisasi antar eksperimen.
            runtime = unwrap_model(trainer.model)
            student = getattr(runtime, "student_model", runtime)
            metadata["initial_student_state_sha256"] = _state_hash(student)
            if hasattr(runtime, "teacher_model"):
                metadata["initial_teacher_state_sha256"] = _state_hash(runtime.teacher_model)
                optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
                if not all(id(p) in optimized for p in runtime.projector.parameters()):
                    raise RuntimeError("KD adapter missing from optimizer")
                metadata["adapter_parameters"] = sum(p.numel() for p in runtime.projector.parameters())

        model.add_callback("on_pretrain_routine_end", record_runtime_initialization)
        if config["device"] == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        else:
            torch.cuda.reset_peak_memory_stats()
        training_started = time.perf_counter()
        # Teacher/baseline/Native berbagi trainer; custom KD hanya mengganti bagian KD.
        trainer = SafeDetectionTrainer
        if config["method"] in ("crosskd", "csakd"):
            from .kd.trainer import CustomKDTrainer
            from .kd.common import SOURCE_REVISIONS
            trainer = CustomKDTrainer
            metadata["kd_adaptation"] = {"method": config["method"], "options": config["kd"][config["method"]],
                                         "status": "stage_b_pilot_not_final", "guide": "docs/stage_b_design.md",
                                         "sources": SOURCE_REVISIONS}
            write_json(run_dir / "metadata.json", metadata)
        model.train(trainer=trainer, **train_args)
        metadata["training_seconds"] = time.perf_counter() - training_started
        metadata["amp_requested"] = config["training"]["amp"]
        metadata["amp_enabled"] = bool(model.trainer.amp)
        metadata["optimizer_updates"] = int(model.trainer.ema.updates)
        if metadata["optimizer_updates"] < 1:
            raise RuntimeError("Training completed without an optimizer update")
        runtime = unwrap_model(model.trainer.model)
        if hasattr(runtime, "teacher_model"):
            metadata["teacher_state_unchanged"] = _state_hash(runtime.teacher_model) == metadata["initial_teacher_state_sha256"]
            metadata["teacher_frozen"] = all(not p.requires_grad and p.grad is None for p in runtime.teacher_model.parameters())
            if not metadata["teacher_state_unchanged"] or not metadata["teacher_frozen"]:
                raise RuntimeError("Teacher parameters/buffers changed during KD training")
        if config["device"] != "cpu":
            metadata["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            metadata["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        if Path(model.trainer.save_dir).resolve() != run_dir.resolve():
            raise RuntimeError("Ultralytics changed the experiment output directory")
        for relative in ("weights/best.pt", "results.csv", "args.yaml"):
            if not (run_dir / relative).is_file():
                raise RuntimeError(f"Missing mandatory training output {relative}")
        metadata.update(recorder.verify(run_dir / "results.csv"))
        saved_args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
        for key in config["training"]:
            if saved_args[key] != config["training"][key]:
                raise RuntimeError(f"Training protocol changed: {key}")
        metrics = evaluate_checkpoint(run_dir / "weights" / "best.pt", data_yaml, run_dir, config)
        # Hash seluruh file lagi; byte identik tidak perlu di-decode dan divalidasi ulang.
        verify_dataset_unchanged(audit)
        summary = {"status": "completed", "run_dir": str(run_dir),
                   "best_pt": str(run_dir / "weights" / "best.pt"), "metrics": metrics,
                   "mode": config["mode"], "nonfinal": config["mode"] == "smoke",
                   "best_epoch": recorder.best_epoch, "test_evaluated": False}
        metadata.update(status="completed", total_seconds=time.perf_counter() - started,
                        artifact_upload_seconds=tracker.upload_seconds,
                        wandb_url=tracker.run.url if tracker.run is not None else None)
        write_json(run_dir / "metadata.json", metadata)
        write_json(run_dir / "run_summary.json", summary)
        tracker.finish("completed", metrics=metrics)
        metadata.update(total_seconds=time.perf_counter() - started, artifact_upload_seconds=tracker.upload_seconds)
        write_json(run_dir / "metadata.json", metadata)
        write_json(output / "latest_run.json", summary)
        return summary
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        key = os.getenv("WANDB_API_KEY")
        if key:
            error = error.replace(key, "[REDACTED]")
        metadata.update(status="failed", error=error, total_seconds=time.perf_counter() - started)
        write_json(run_dir / "metadata.json", metadata)
        write_json(run_dir / "run_summary.json", {"status": "failed", "run_dir": str(run_dir), "error": error})
        tracker.finish("failed", error=error)
        if isinstance(exc, torch.OutOfMemoryError):
            raise RuntimeError("GPU OOM at configured batch/imgsz; protocol was not automatically changed") from exc
        raise
