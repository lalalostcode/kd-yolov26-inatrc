import json
from pathlib import Path

import pandas as pd
import yaml

from inatrc_kd.summary import summarize_runs


def _run(root: Path, name: str, mode: str, status: str = "completed"):
    directory = root / name
    directory.mkdir()
    summary = {"status": status, "mode": mode, "best_pt": "weights/best.pt",
               "metrics": {"mAP50_95": 0.1, "mAP50": 0.2, "precision": 0.3, "recall": 0.4}}
    (directory / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config = {"stage": "student", "model": "yolo26n.pt", "method": "none", "seed": 42,
              "mode": mode, "dataset_fingerprint": "test-hash"}
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (directory / "metadata.json").write_text("{}", encoding="utf-8")


def test_research_summary_excludes_synthetic_and_failed_runs(tmp_path):
    _run(tmp_path, "real", "full")
    _run(tmp_path, "synthetic", "smoke")
    _run(tmp_path, "failed", "full", "failed")
    result = pd.read_csv(summarize_runs(tmp_path))
    assert len(result) == 1 and result.iloc[0]["mode"] == "full"
    assert result.iloc[0]["dataset_fingerprint"] == "test-hash"
    diagnostics = pd.read_csv(summarize_runs(tmp_path, include_smoke=True))
    assert len(diagnostics) == 2 and set(diagnostics["mode"]) == {"full", "smoke"}
