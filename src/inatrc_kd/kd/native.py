"""Use Ultralytics Native KD without reimplementing its loss or trainer."""

from __future__ import annotations

import math

from inatrc_kd.preflight import check_teacher, ordered_names


def native_train_kwargs(config: dict) -> dict:
    kd = config.get("kd", {})
    checkpoint = kd.get("teacher_checkpoint")
    if not checkpoint:
        raise ValueError(f"{config.get('method', 'native')} KD requires kd.teacher_checkpoint pointing to a local YOLO26m best.pt.")
    teacher = check_teacher(checkpoint)
    expected_names = ordered_names(config["dataset"]["names"])
    if teacher["names"] != expected_names:
        raise ValueError(
            f"Teacher and dataset class IDs/names differ. Teacher={teacher['names']}; dataset={expected_names}."
        )
    if config.get("mode", "smoke") == "full" and teacher["diagnostics_only"]:
        raise ValueError("A synthetic diagnostics-only teacher cannot be used for full/research training.")
    weight = float(kd.get("dis", 6.0))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("kd.dis must be finite and positive; use method='none' for the control.")
    return {"distill_model": teacher["path"], "dis": weight}
