"""Protocol, method gates and tracking regressions without model training."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from inatrc_kd.config import load_config, validate_config
from inatrc_kd.data import CLASS_NAMES
from inatrc_kd.kd import configure_kd
from inatrc_kd.tracking import EpochRecorder, ExperimentTracker, numeric


def test_default_config_is_small_nonfinal_and_optional_tracking():
    config = load_config()
    assert (config["stage"], config["student"], config["method"], config["seed"], config["mode"]) == (
        "student", "n", "none", 42, "smoke"
    )
    assert config["model"] == "yolo26n.yaml"
    assert config["training"]["epochs"] == 1
    assert config["training"]["batch"] == 2
    assert config["training"]["imgsz"] < 640
    assert config["training"]["fraction"] == 1.0
    assert config["tracking"]["mode"] == "disabled"
    assert config["tracking"]["upload_checkpoint_each_epoch"] is False
    assert config["training"]["save_period"] == -1
    assert config["evaluation"]["split"] == "val"
    assert config["evaluation"]["nms"] is None
    assert config["evaluation"]["pilot_latency"] is False
    assert config["dataset"]["names"] == list(CLASS_NAMES)


def test_full_config_requires_explicit_dataset():
    with pytest.raises(ValueError, match="requires --dataset-root"):
        load_config(mode="full")


def test_full_can_use_explicit_local_pretrained_weights_without_loading_them(tmp_path):
    checkpoint = tmp_path / "yolo26n.pt"
    checkpoint.write_bytes(b"Config validation only; not a loadable model")
    config = load_config(mode="full", dataset_root=str(tmp_path), weights=str(checkpoint))
    assert config["model"] == str(checkpoint.resolve())
    assert config["training"]["epochs"] == 100
    with pytest.raises(ValueError, match="smoke uses random YAML"):
        load_config(weights=str(checkpoint))
    with pytest.raises(ValueError, match="existing pretrained .pt"):
        load_config(mode="full", dataset_root=str(tmp_path), weights=str(tmp_path / "missing.pt"))


@pytest.mark.parametrize("stage, student, model", [
    ("teacher", "n", "yolo26m.pt"), ("student", "s", "yolo26s.pt"),
    ("student", "n", "yolo26n.pt"),
])
def test_full_preserves_reference_protocol(tmp_path, stage, student, model):
    config = load_config(mode="full", dataset_root=str(tmp_path), stage=stage, student=student)
    assert config["model"] == model
    expected = {
        "imgsz": 640, "epochs": 100, "batch": 16, "patience": 25,
        "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "amp": True, "workers": 2, "cache": False,
        "close_mosaic": 10, "flipud": 0.0, "mixup": 0.0, "cutmix": 0.0,
        "deterministic": True, "compile": False, "plots": True, "val": True,
        "fraction": 1.0, "seed": 42,
    }
    assert {key: config["training"][key] for key in expected} == expected
    assert config["evaluation"]["split"] == "val"
    assert config["dataset"]["expected_images"] == 3250


def test_config_overrides_are_independent_and_tracking_has_no_identity(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "my-research")
    changed = load_config(student="s", seed=123, output_root=tmp_path, wandb_mode="offline", expected_images=20)
    assert changed["training"]["seed"] == changed["seed"] == 123
    assert changed["tracking"]["project"] == "my-research"
    assert changed["tracking"]["entity"] is None
    assert changed["tracking"]["mode"] == "offline"
    assert changed["dataset"]["expected_images"] == 20
    changed["training"]["epochs"] = 999
    assert load_config()["training"]["epochs"] == 1


@pytest.mark.parametrize("change, message", [
    ({"stage": "teacher", "method": "native"}, "Teacher training"),
    ({"student": "m"}, "Invalid student"),
    ({"method": "invented"}, "Invalid method"),
    ({"mode": "automatic"}, "Invalid mode"),
    ({"seed": True}, "nonnegative integer"),
    ({"seed": -1}, "nonnegative integer"),
])
def test_invalid_selections_fail_before_training(change, message):
    with pytest.raises(ValueError, match=message):
        load_config(**change)


@pytest.mark.parametrize("field, value, message", [
    ("batch", -1, "positive integer batch"), ("batch", 0, "positive integer batch"),
    ("imgsz", 161, "multiple of 32"), ("epochs", 0, "epochs must be positive"),
])
def test_unsafe_training_settings_are_rejected(field, value, message):
    config = load_config()
    config["training"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize("method", ["crosskd", "csakd"])
def test_custom_kd_is_an_explicit_gate(method):
    with pytest.raises(ValueError, match="requires.*teacher_checkpoint"):
        configure_kd(load_config(method=method))
    assert configure_kd(load_config()) == {}


def _trainer(run_dir, *, epoch=0, fitness=0.25, best_fitness=0.25, native=False):
    losses = {"train/box_loss": 1.0, "train/cls_loss": 2.0}
    if native:
        losses["train/dis_loss"] = 0.5
    return SimpleNamespace(
        epoch=epoch, fitness=fitness, best_fitness=best_fitness, wdir=run_dir / "weights",
        device=SimpleNamespace(type="cpu"), tloss=None,
        label_loss_items=Mock(return_value=losses),
        metrics={"metrics/mAP50-95(B)": fitness}, lr={"lr/pg0": 0.001},
        validator=SimpleNamespace(metrics=SimpleNamespace(box=SimpleNamespace(
            ap_class_index=np.array([1]), p=np.array([0.4]), r=np.array([0.5]),
            ap50=np.array([0.6]), ap=np.array([0.3]), all_ap=np.full((1, 10), 0.3),
        ))),
    )


def _record_epoch(config, run_dir, *, native=False):
    tracker = Mock()
    recorder = EpochRecorder(config, run_dir, tracker)
    trainer = _trainer(run_dir, native=native)
    recorder.on_epoch_start(trainer)
    recorder.on_save(trainer)
    recorder.on_fit_epoch_end(trainer)
    return recorder, tracker, trainer


def test_epoch_metrics_do_not_require_epoch_checkpoint_files(tmp_path):
    config = load_config()
    recorder, tracker, trainer = _record_epoch(config, tmp_path)
    assert config["training"]["save_period"] == -1
    assert not (trainer.wdir / "epoch0.pt").exists()
    assert recorder.best_epoch == recorder.last_logged_epoch == 1
    recorder.on_fit_epoch_end(trainer)  # duplicate from final_eval must not create a fake epoch
    tracker.log.assert_called_once()
    tracker.epoch_artifact.assert_not_called()
    assert len(pd.read_csv(tmp_path / "epoch_history.csv")) == 1
    history = recorder.rows[0]
    assert history["epoch"] == 1
    assert history["per_class/1/AP50_95"] == pytest.approx(0.3)
    assert "per_class/0/AP50_95" not in history
    assert not any(key.startswith("system/gpu_") for key in history)
    results = tmp_path / "results.csv"
    pd.DataFrame({"epoch": [1], "metrics/mAP50-95(B)": [0.25]}).to_csv(results, index=False)
    assert recorder.verify(results)["best_epoch"] == 1


def test_fit_event_without_a_saved_epoch_is_ignored(tmp_path):
    tracker = Mock()
    recorder = EpochRecorder(load_config(), tmp_path, tracker)
    recorder.on_fit_epoch_end(_trainer(tmp_path))
    assert not recorder.rows
    tracker.log.assert_not_called()


@pytest.mark.parametrize("method, has_kd_column, fails", [
    ("none", False, False), ("none", True, True),
    ("native", False, True), ("native", True, False),
])
def test_history_proves_selected_kd_condition(tmp_path, method, has_kd_column, fails):
    recorder, _, _ = _record_epoch(load_config(method=method), tmp_path, native=method == "native")
    history = {"epoch": [1], "metrics/mAP50-95(B)": [0.25]}
    if has_kd_column:
        history["train/dis_loss"] = [0.5]
    results = tmp_path / "results.csv"
    pd.DataFrame(history).to_csv(results, index=False)
    if fails:
        with pytest.raises(RuntimeError, match="KD loss|lacks dis_loss"):
            recorder.verify(results)
    else:
        assert recorder.verify(results)["completed_epochs"] == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_epoch_scalars_fail_loudly(tmp_path, invalid):
    with pytest.raises(RuntimeError, match="Non-finite"):
        numeric(invalid, name="loss")
    recorder = EpochRecorder(load_config(), tmp_path, Mock())
    with pytest.raises(RuntimeError, match="Non-finite"):
        recorder.on_save(_trainer(tmp_path, fitness=invalid))


def _fake_wandb(monkeypatch):
    run = SimpleNamespace(id="test-run", url="https://wandb.invalid/run", summary={},
                          log=Mock(), log_artifact=Mock(return_value=SimpleNamespace(wait=Mock())), finish=Mock())
    artifacts = []

    def artifact(name, type, metadata=None):
        created = SimpleNamespace(name=name, type=type, metadata=metadata, add_file=Mock())
        artifacts.append(created)
        return created

    module = SimpleNamespace(run=None, login=Mock(), init=Mock(return_value=run), Artifact=artifact)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module, run, artifacts


def test_disabled_tracker_never_logs_in_or_starts_a_run(monkeypatch, tmp_path):
    module, run, artifacts = _fake_wandb(monkeypatch)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker = ExperimentTracker(load_config(), tmp_path)
    tracker.start({"environment": {}})
    tracker.log({"loss": 1.0}, step=1)
    tracker.epoch_artifact(tmp_path / "missing.pt", 1, True)
    tracker.finish("completed")
    module.login.assert_not_called()
    module.init.assert_not_called()
    run.log.assert_not_called()
    run.finish.assert_not_called()
    assert not artifacts


def test_online_tracker_has_one_run_and_final_best_artifact(monkeypatch, tmp_path):
    module, run, artifacts = _fake_wandb(monkeypatch)
    monkeypatch.setenv("WANDB_API_KEY", "test-secret-never-print")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "best.pt").write_bytes(b"fixture checkpoint")
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    tracker = ExperimentTracker(load_config(wandb_mode="online"), tmp_path)
    tracker.start({"nonfinal": True})
    tracker.log({"epoch": 1, "loss": 1.0}, step=1)
    tracker.epoch_artifact(tmp_path / "missing-epoch.pt", 1, True)  # upload default false
    tracker.finish("completed", metrics={"mAP50_95": 0.25, "mAP50": 0.5, "precision": 0.6, "recall": 0.7})
    module.login.assert_called_once_with(key="test-secret-never-print", relogin=False)
    module.init.assert_called_once()
    assert "test-secret-never-print" not in repr(module.init.call_args)
    assert run.summary["research/status"] == "completed"
    assert run.summary["research/test_evaluated"] is False
    assert run.summary["validation/mAP50_95"] == 0.25
    assert len(artifacts) == 1
    assert any(call.kwargs.get("name") == "weights/best.pt" for call in artifacts[0].add_file.call_args_list)
    assert run.log_artifact.call_args.kwargs["aliases"] == ["best-checkpoint", "final"]
    run.log_artifact.return_value.wait.assert_called_once()
    run.finish.assert_called_once_with(exit_code=0)


def test_online_missing_secret_and_duplicate_run_fail_before_init(monkeypatch, tmp_path):
    module, _, _ = _fake_wandb(monkeypatch)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker = ExperimentTracker(load_config(wandb_mode="online"), tmp_path)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        tracker.start({})
    module.run = object()
    with pytest.raises(RuntimeError, match="existing W&B run"):
        tracker.start({})
    module.login.assert_not_called()
    module.init.assert_not_called()


def test_failed_run_closes_once_and_does_not_upload_final_model(monkeypatch, tmp_path):
    _, run, artifacts = _fake_wandb(monkeypatch)
    tracker = ExperimentTracker(load_config(wandb_mode="offline"), tmp_path)
    tracker.start({})
    tracker.finish("failed", error="diagnostic failure")
    tracker.finish("failed", error="diagnostic failure")  # exception cleanup is idempotent
    assert run.summary["research/status"] == "failed"
    assert run.summary["research/error"] == "diagnostic failure"
    run.finish.assert_called_once_with(exit_code=1)
    assert not artifacts


def test_final_artifact_failure_records_failure_and_cleans_up_once(monkeypatch, tmp_path):
    _, run, _ = _fake_wandb(monkeypatch)
    tracker = ExperimentTracker(load_config(wandb_mode="offline"), tmp_path)
    tracker.start({})
    run.log_artifact.side_effect = RuntimeError("artifact upload failed")
    with pytest.raises(RuntimeError, match="artifact upload failed"):
        tracker.finish("completed")
    tracker.finish("failed", error="artifact upload failed")
    assert run.summary["research/status"] == "failed"
    run.finish.assert_called_once_with(exit_code=1)
