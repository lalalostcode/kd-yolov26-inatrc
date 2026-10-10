"""Gunakan API Native KD Ultralytics untuk loss dan trainer resminya."""

from __future__ import annotations

import math

from inatrc_kd.preflight import check_teacher, ordered_names


def native_train_kwargs(config: dict) -> dict:
    kd = config.get("kd", {})
    checkpoint = kd.get("teacher_checkpoint")
    if not checkpoint:
        raise ValueError(f"{config.get('method', 'native')} KD requires kd.teacher_checkpoint pointing to a local YOLO26m best.pt.")
    benchmark = bool(config.get("benchmark_test", False))
    teacher = check_teacher(checkpoint, expected_nc=80) if benchmark else check_teacher(checkpoint)
    if benchmark:
        evidence = teacher.get("benchmark_metadata") or {}
        if (evidence.get("status") != "PASS" or evidence.get("stage") != "teacher"
                or evidence.get("dataset") != "coco8.yaml" or evidence.get("seed") != 42
                or evidence.get("completed_epochs") != 1 or evidence.get("optimizer_updates", 0) < 1):
            raise ValueError("Benchmark KD requires the trained COCO8 teacher checkpoint from notebook 01, not raw pretrained/synthetic weights.")
    # Urutan ID kelas harus sama agar target teacher cocok dengan label student.
    expected_names = ordered_names(config["dataset"]["names"])
    if teacher["names"] != expected_names:
        raise ValueError(
            f"Teacher and dataset class IDs/names differ. Teacher={teacher['names']}; dataset={expected_names}."
        )
    # Teacher sintetis hanya untuk smoke test, bukan eksperimen penelitian pada dataset asli.
    if config.get("mode", "smoke") == "full" and teacher["diagnostics_only"]:
        raise ValueError("A synthetic diagnostics-only teacher cannot be used for full/research training.")
    weight = float(kd.get("dis", 6.0))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("kd.dis must be finite and positive; use method='none' for the control.")
    # Serahkan checkpoint dan bobot loss ke API KD resmi; algoritmenya tetap milik Ultralytics.
    return {"distill_model": teacher["path"], "dis": weight}
