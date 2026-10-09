"""Muat ulang best.pt dan hitung metrik pada val dengan protokol yang tetap."""
from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from .io import sha256, write_json
from .pipeline import SafeDetectionValidator


def evaluate_checkpoint(checkpoint: str | Path, data_yaml: str | Path, run_dir: str | Path, config: dict) -> dict:
    checkpoint, run_dir = Path(checkpoint), Path(run_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint}")
    evaluation, training = config["evaluation"], config["training"]
    if evaluation["split"] != "val" or evaluation["nms"] is not None:
        raise ValueError("Only the fixed val/nms=None protocol is available in Stage A")
    # Evaluasi model tersimpan secara mandiri; teacher/adapter KD tidak diperlukan.
    model = YOLO(str(checkpoint))
    native = model.model
    head = native.model[-1]
    if int(head.nc) != 5 or [native.names[i] for i in range(5)] != config["dataset"]["names"]:
        raise RuntimeError("Checkpoint class names/head do not match the InaTRC protocol")
    if hasattr(native, "teacher_model") or hasattr(native, "projector") or hasattr(native, "student_model"):
        raise RuntimeError("Checkpoint still contains the KD training wrapper")
    audit = {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint),
             "size_mb": checkpoint.stat().st_size / 1e6,
             "parameters": sum(parameter.numel() for parameter in native.parameters()),
             "nc": int(head.nc), "teacher_present": False, "projector_present": False}
    # Salinan sebelum fusion menjaga penghitungan FLOPs memakai model checkpoint asli.
    complexity_probe = deepcopy(native).float().eval()
    result = model.val(
        validator=SafeDetectionValidator, data=str(data_yaml), split="val",
        imgsz=training["imgsz"], batch=training["batch"], workers=training["workers"],
        device=config["device"], plots=training["plots"], save_json=False,
        nms=None, conf=evaluation["conf"], iou=evaluation["iou"], max_det=evaluation["max_det"],
        augment=False, quantize=evaluation["quantize"],
        project=str(run_dir / "validation"), name="best-val", exist_ok=False,
    )
    box = result.box
    metrics = {"split": "val", "test_evaluated": False, "mode": config["mode"],
               "nonfinal": config["mode"] == "smoke", "nms": None,
               "precision": float(box.mp), "recall": float(box.mr),
               "mAP50": float(box.map50), "mAP50_95": float(box.map),
               "checkpoint": audit, "validation_dir": str(result.save_dir),
               "validation_precision": "float32" if evaluation["quantize"] is None else evaluation["quantize"]}
    for key in ("precision", "recall", "mAP50", "mAP50_95"):
        if not math.isfinite(metrics[key]):
            raise RuntimeError(f"Non-finite validation metric {key}")
    indices = {int(class_id): position for position, class_id in enumerate(box.ap_class_index)}
    per_class = []
    ap = np.asarray(box.all_ap)
    for class_id, name in enumerate(config["dataset"]["names"]):
        position = indices.get(class_id)
        # Kelas tanpa AP valid tetap null, agar tidak dianggap memperoleh nilai nol.
        row = {"class_id": class_id, "class_name": name,
               **dict.fromkeys(("precision", "recall", "AP50", "AP75", "AP50_95"))}
        if position is not None:
            values = (box.p[position], box.r[position], box.ap50[position], ap[position, 5], ap[position].mean())
            for key, value in zip(("precision", "recall", "AP50", "AP75", "AP50_95"), values, strict=True):
                row[key] = float(value)
                if not math.isfinite(row[key]):
                    raise RuntimeError(f"Non-finite {key} for class {class_id}")
        per_class.append(row)
    metrics["per_class"] = per_class
    complexity = {"parameters": audit["parameters"], "imgsz": training["imgsz"],
                  "GFLOPs_estimate": None,
                  "convention": "2 * THOP MACs / 1e9, float32 unfused checkpoint; estimate"}
    gflops = float(get_flops(complexity_probe, imgsz=training["imgsz"]))
    if math.isfinite(gflops) and gflops > 0:
        complexity["GFLOPs_estimate"] = gflops
    else:
        complexity["profile_status"] = "unavailable_from_upstream_profiler"
    del complexity_probe
    metrics["complexity"] = complexity
    metrics["speed_ms_per_image"] = {key: float(value) for key, value in result.speed.items()}
    if evaluation["pilot_latency"]:
        from .data import IMAGE_EXTENSIONS
        import yaml
        data = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        image = next(iter(sorted(path for path in Path(data["val"]).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)))
        times = []
        for index in range(evaluation["warmup_calls"] + evaluation["timed_calls"]):
            if str(config["device"]) != "cpu":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(str(image), imgsz=training["imgsz"], device=config["device"],
                          nms=None, verbose=False, save=False)
            if str(config["device"]) != "cpu":
                torch.cuda.synchronize()
            if index >= evaluation["warmup_calls"]:
                times.append((time.perf_counter() - start) * 1000)
        metrics["pilot_latency"] = {"batch": 1, "imgsz": training["imgsz"],
                                    "median_wall_ms": median(times), "mean_wall_ms": float(np.mean(times)),
                                    "p95_wall_ms": float(np.percentile(times, 95)),
                                    "note": "Single val image; includes I/O, Python, preprocessing and postprocessing; hardware-specific estimate"}
    pd.DataFrame(per_class).to_csv(run_dir / "per_class.csv", index=False)
    write_json(run_dir / "checkpoint_audit.json", audit)
    write_json(run_dir / "complexity.json", complexity)
    write_json(run_dir / "metrics.json", metrics)
    return metrics
