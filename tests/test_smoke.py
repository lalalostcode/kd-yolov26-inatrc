"""Meaningful checkpoint regression through the actual pinned training pipeline."""
from pathlib import Path
import json

import pandas as pd
import pytest

from inatrc_kd.config import load_config
from inatrc_kd.train import run_experiment
from inatrc_kd.preflight import make_synthetic_teacher


@pytest.fixture(scope="module")
def fixed_pilot_teacher(tmp_path_factory):
    return make_synthetic_teacher(tmp_path_factory.mktemp("fixed-pilot-teacher") / "teacher-m.pt", seed=42)


@pytest.mark.smoke
@pytest.mark.parametrize("student,method", [(s, m) for s in ("s", "n") for m in ("none", "native", "crosskd", "csakd")])
def test_training_checkpoint_roundtrip(tmp_path, method, student, fixed_pilot_teacher):
    config = load_config(method=method, student=student, output_root=tmp_path / "experiments",
                         teacher_ckpt=str(fixed_pilot_teacher) if method != "none" else None)
    summary = run_experiment(config)
    assert summary["status"] == "completed" and summary["nonfinal"]
    assert summary["test_evaluated"] is False
    run_dir = Path(summary["run_dir"])
    for file in ("weights/best.pt", "metrics.json", "per_class.csv", "resolved_config.yaml", "metadata.json", "results.csv"):
        assert (run_dir / file).is_file()
    assert len(pd.read_csv(run_dir / "results.csv")) == 1
    assert len(pd.read_csv(run_dir / "epoch_history.csv")) == 1
    assert summary["metrics"]["checkpoint"]["nc"] == 5
    assert summary["metrics"]["checkpoint"]["teacher_present"] is False
    assert summary["metrics"]["checkpoint"]["projector_present"] is False
    assert len(summary["metrics"]["per_class"]) == 5
    assert not list((run_dir / "synthetic_data").rglob("*.cache"))
    assert not list((run_dir / "weights").glob("epoch*.pt"))
    assert json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["optimizer_updates"] > 0
    if method != "none":
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["teacher_frozen"] and metadata["teacher_state_unchanged"]
