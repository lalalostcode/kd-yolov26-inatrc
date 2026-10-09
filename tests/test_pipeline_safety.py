"""Research protocol failures must not repair data or restart optimization."""
from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest
import torch

from inatrc_kd.cli import main
from inatrc_kd.config import load_config, validate_config
from inatrc_kd.pipeline import DetectionTrainer, SafeDetectionTrainer


def test_first_epoch_oom_cannot_reduce_explicit_batch(monkeypatch):
    trainer = object.__new__(SafeDetectionTrainer)
    trainer.args = SimpleNamespace(batch=16)
    trainer.batch_size = 16
    trainer._oom_retries = 0
    upstream = Mock()
    monkeypatch.setattr(DetectionTrainer, "run_callbacks", upstream)
    trainer.run_callbacks("on_train_epoch_start")
    upstream.assert_called_once_with("on_train_epoch_start")
    assert trainer._oom_retries == 3  # pinned trainer's stop condition before batch mutation
    assert trainer.args.batch == trainer.batch_size == 16
    trainer.run_callbacks("on_fit_epoch_end")
    assert trainer._oom_retries == 3


def test_recovery_never_rebuilds_optimizer_or_dataloaders(monkeypatch):
    trainer = object.__new__(SafeDetectionTrainer)
    upstream = Mock(return_value="initial-build")
    monkeypatch.setattr(DetectionTrainer, "_build_train_pipeline", upstream)
    assert trainer._build_train_pipeline() == "initial-build"
    trainer._oom_retries = 1
    with pytest.raises(RuntimeError, match="OOM recovery is disabled"):
        trainer._build_train_pipeline()
    upstream.assert_called_once()


@pytest.mark.parametrize("epoch", [0, 5])
@pytest.mark.parametrize("field", ["loss", "fitness", "loss_items"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_values_fail_even_in_first_epoch(epoch, field, value):
    trainer = object.__new__(SafeDetectionTrainer)
    trainer.loss = torch.tensor(1.0)
    trainer.fitness = 0.2
    trainer.loss_items = {"box_loss": torch.tensor(1.0)}
    setattr(trainer, field, {"box_loss": torch.tensor(value)} if field == "loss_items" else value)
    with pytest.raises(RuntimeError, match="automatic recovery is disabled"):
        trainer._handle_nan_recovery(epoch)


def test_finite_epoch_does_not_request_recovery():
    trainer = object.__new__(SafeDetectionTrainer)
    trainer.loss = torch.tensor(1.0)
    trainer.fitness = 0.0
    trainer.loss_items = {"box_loss": torch.tensor(1.0)}
    assert trainer._handle_nan_recovery(0) is False


@pytest.mark.parametrize("command", ["audit", "run", "evaluate", "preflight"])
@pytest.mark.parametrize("nested", [False, True])
def test_cli_rejects_source_outputs_before_any_write(tmp_path, command, nested):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    sentinel = dataset / "source.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    output = dataset / "outputs" if nested else dataset
    args = [command, "--dataset-root", str(dataset), "--output-root", str(output)]
    if command == "evaluate":
        args += ["--checkpoint", "best.pt"]
    with pytest.raises(ValueError, match="outside the source dataset"):
        main(args)
    assert sorted(path.name for path in dataset.iterdir()) == ["source.txt"]
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def test_direct_runner_validates_mutated_source_output_before_writes(tmp_path):
    from inatrc_kd.train import run_experiment
    config = load_config()
    config["dataset"]["root"] = str(tmp_path / "dataset")
    config["output_root"] = str(tmp_path / "dataset" / "outputs")
    with pytest.raises(ValueError, match="outside the source dataset"):
        validate_config(config)
    with pytest.raises(ValueError, match="outside the source dataset"):
        run_experiment(config)
    assert not (tmp_path / "dataset").exists()


@pytest.mark.smoke
def test_injected_first_forward_oom_fails_without_halving_batch(tmp_path, monkeypatch):
    from inatrc_kd.train import run_experiment
    original = SafeDetectionTrainer.run_callbacks
    observed = []

    def callbacks(trainer, event):
        original(trainer, event)
        if event == "on_train_start":
            observed.append(trainer)
            monkeypatch.setattr(trainer.model, "forward", Mock(
                side_effect=torch.OutOfMemoryError("out of memory: injected CPU policy test")))

    monkeypatch.setattr(SafeDetectionTrainer, "run_callbacks", callbacks)
    output = tmp_path / "experiments"
    with pytest.raises(RuntimeError, match="protocol was not automatically changed"):
        run_experiment(load_config(output_root=output))
    trainer, = observed
    assert trainer.batch_size == trainer.args.batch == 2
    assert trainer._oom_retries == 3
    assert not (output / "latest_run.json").exists()
    metadata_path, = output.glob("*/metadata.json")
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["status"] == "failed"
