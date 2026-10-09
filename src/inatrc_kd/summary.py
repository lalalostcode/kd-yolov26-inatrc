"""Gabungkan metrik run yang selesai ke CSV; hasil smoke disertakan hanya jika diminta."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def summarize_runs(output_root: str | Path, *, include_smoke: bool = False) -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.glob("*/run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["status"] != "completed" or (summary["mode"] == "smoke" and not include_smoke):
            continue
        config = yaml.safe_load((path.parent / "resolved_config.yaml").read_text(encoding="utf-8"))
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        rows.append({"run_dir": str(path.parent), "stage": config["stage"],
                     "model": config["model"], "method": config["method"], "seed": config["seed"],
                     "mode": config["mode"], "dataset_fingerprint": config["dataset_fingerprint"],
                     "teacher_sha256": metadata.get("teacher_checkpoint_sha256"),
                     "best_pt": summary["best_pt"],
                     **{key: summary["metrics"][key] for key in ("mAP50_95", "mAP50", "precision", "recall")}})
    destination = root / ("diagnostic_summary.csv" if include_smoke else "experiment_summary.csv")
    columns = ["run_dir", "stage", "model", "method", "seed", "mode", "dataset_fingerprint", "teacher_sha256",
               "best_pt", "mAP50_95", "mAP50", "precision", "recall"]
    pd.DataFrame(rows, columns=columns).to_csv(destination, index=False)
    return destination
