"""One optional manual W&B run; epoch accounting also works completely offline."""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pandas as pd

from .io import write_json


def numeric(value, *, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {name}: {result}")
    return result


class ExperimentTracker:
    def __init__(self, config: dict, run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.run = None
        self.finished = False
        self.upload_seconds = 0.0

    def start(self, metadata: dict) -> None:
        if self.config["tracking"]["mode"] == "disabled":
            return
        import wandb
        if wandb.run is not None:
            raise RuntimeError("An existing W&B run is active; refusing a duplicate run")
        if self.config["tracking"]["mode"] == "online":
            key = os.getenv("WANDB_API_KEY")
            if not key:
                raise RuntimeError("W&B online requires WANDB_API_KEY from environment or Kaggle Secrets")
            wandb.login(key=key, relogin=False)
        self.run = wandb.init(
            mode=self.config["tracking"]["mode"], entity=self.config["tracking"]["entity"],
            project=self.config["tracking"]["project"], group=self.config["tracking"]["group"],
            name=self.run_dir.name, job_type=f"{self.config['stage']}-{self.config['method']}",
            tags=["InaTRC", self.config["mode"], f"seed-{self.config['seed']}"],
            dir=str(self.run_dir), config={**self.config, "metadata": metadata}, reinit=False,
        )
        self.run.summary["research/status"] = "running"
        self.run.summary["research/test_evaluated"] = False

    def log(self, values: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def epoch_artifact(self, checkpoint: Path, epoch: int, is_best: bool) -> None:
        if self.run is None or not self.config["tracking"]["upload_checkpoint_each_epoch"]:
            return
        import wandb
        started = time.perf_counter()
        artifact = wandb.Artifact(f"epochs-{self.run.id}", type="model", metadata={"epoch": epoch})
        artifact.add_file(str(checkpoint), name="checkpoint.pt")
        logged = self.run.log_artifact(artifact, aliases=[f"epoch-{epoch:03d}"] + (["best"] if is_best else []))
        if self.config["tracking"]["mode"] == "online":
            logged.wait()
        self.upload_seconds += time.perf_counter() - started

    def finish(self, status: str, metrics: dict | None = None, error: str | None = None) -> None:
        if self.run is None or self.finished:
            return
        import wandb
        exit_status = status
        upload_started = None
        try:
            self.run.summary["research/status"] = status
            self.run.summary["research/test_evaluated"] = False
            if error:
                self.run.summary["research/error"] = error
            if metrics is not None:
                for key in ("mAP50_95", "mAP50", "precision", "recall"):
                    self.run.summary[f"validation/{key}"] = metrics[key]
                self.run.summary["checkpoint/best_pt"] = str(self.run_dir / "weights" / "best.pt")
            if status == "completed":
                upload_started = time.perf_counter()
                artifact = wandb.Artifact(f"final-{self.run.id}", type="model",
                                          metadata={"mode": self.config["mode"], "test_evaluated": False})
                for name in ("weights/best.pt", "metrics.json", "per_class.csv", "results.csv",
                             "resolved_config.yaml", "metadata.json", "run_summary.json"):
                    path = self.run_dir / name
                    if path.is_file():
                        artifact.add_file(str(path), name=name)
                for folder in ("reports", "validation"):
                    for path in sorted((self.run_dir / folder).rglob("*")):
                        if path.is_file():
                            artifact.add_file(str(path), name=path.relative_to(self.run_dir).as_posix())
                logged = self.run.log_artifact(artifact, aliases=["best-checkpoint", "final"])
                if self.config["tracking"]["mode"] == "online":
                    logged.wait()
        except BaseException as exc:
            exit_status = "failed"
            self.run.summary["research/status"] = "failed"
            self.run.summary["research/error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if upload_started is not None:
                self.upload_seconds += time.perf_counter() - upload_started
            self.finished = True
            self.run.finish(exit_code=0 if exit_status == "completed" else 1)


class EpochRecorder:
    """Track actual saved epochs, including when save_period=-1 and W&B is disabled."""
    def __init__(self, config: dict, run_dir: Path, tracker: ExperimentTracker):
        self.config, self.run_dir, self.tracker = config, run_dir, tracker
        self.last_saved_epoch = 0
        self.last_logged_epoch = 0
        self.best_epoch = None
        self.best_fitness = None
        self.epoch_started = None
        self.rows: list[dict] = []
        self.checkpoints: list[dict] = []

    def on_epoch_start(self, trainer) -> None:
        self.epoch_started = time.perf_counter()

    def on_save(self, trainer) -> None:
        epoch = int(trainer.epoch + 1)
        fitness = numeric(trainer.fitness, name="fitness")
        best = numeric(trainer.best_fitness, name="best_fitness")
        is_best = fitness is not None and best is not None and fitness == best
        if is_best:
            self.best_epoch, self.best_fitness = epoch, best
        self.last_saved_epoch = epoch
        self.checkpoints.append({"epoch": epoch, "fitness": fitness, "best_fitness": best,
                                 "is_best_at_save": is_best})
        pd.DataFrame(self.checkpoints).to_csv(self.run_dir / "checkpoint_manifest.csv", index=False)
        if self.config["tracking"]["upload_checkpoint_each_epoch"]:
            checkpoint = Path(trainer.wdir) / f"epoch{trainer.epoch}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"Missing requested epoch checkpoint: {checkpoint}")
            self.tracker.epoch_artifact(checkpoint, epoch, is_best)

    def on_fit_epoch_end(self, trainer) -> None:
        epoch = int(trainer.epoch + 1)
        if epoch != self.last_saved_epoch or epoch <= self.last_logged_epoch:
            return  # final_eval emits this event a second time
        values = {"epoch": epoch}
        for group in (trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.metrics or {}, trainer.lr or {}):
            for key, value in group.items():
                value = numeric(value, name=key)
                if value is not None:
                    values[key] = value
        values["fit/fitness"] = numeric(trainer.fitness, name="fitness")
        values["fit/best_fitness"] = numeric(trainer.best_fitness, name="best_fitness")
        if self.epoch_started is not None:
            values["fit/epoch_seconds"] = time.perf_counter() - self.epoch_started
        import torch
        if trainer.device.type == "cuda":
            values["system/gpu_memory_allocated_gib"] = torch.cuda.memory_allocated(trainer.device) / 2**30
            values["system/gpu_memory_reserved_gib"] = torch.cuda.memory_reserved(trainer.device) / 2**30
        box = getattr(getattr(getattr(trainer, "validator", None), "metrics", None), "box", None)
        if box is not None:
            for position, class_id in enumerate(box.ap_class_index):
                for metric, array in (("precision", box.p), ("recall", box.r), ("AP50", box.ap50), ("AP50_95", box.ap)):
                    values[f"per_class/{int(class_id)}/{metric}"] = numeric(array[position], name=metric)
        self.rows.append(values)
        pd.DataFrame(self.rows).to_csv(self.run_dir / "epoch_history.csv", index=False)
        self.tracker.log(values, step=epoch)
        self.last_logged_epoch = epoch

    def attach(self, model) -> None:
        model.add_callback("on_train_epoch_start", self.on_epoch_start)
        model.add_callback("on_model_save", self.on_save)
        model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def verify(self, results_csv: Path) -> dict:
        history = pd.read_csv(results_csv)
        history.columns = history.columns.str.strip()
        if len(history) != len(self.rows) or len(history) != self.last_saved_epoch:
            raise RuntimeError("CSV, epoch logger, and checkpoint events disagree")
        column = "metrics/mAP50-95(B)"
        if column not in history or not all(math.isfinite(float(v)) for v in history[column]):
            raise RuntimeError("Training history lacks finite mAP50-95")
        if self.config["method"] == "none" and any("dis_loss" in col for col in history):
            raise RuntimeError("Baseline unexpectedly contains KD loss")
        if self.config["method"] != "none" and not any("dis_loss" in col for col in history):
            raise RuntimeError("KD history lacks dis_loss; refusing a silent baseline")
        for method, components in (("crosskd", ("kd_cls_loss", "kd_box_loss")),
                                   ("csakd", ("kd_agka_loss", "kd_cross_loss"))):
            if self.config["method"] == method and not all(any(name in column for column in history) for name in components):
                raise RuntimeError(f"{method} history lacks algorithm-specific loss components")
        if self.best_epoch is None:
            raise RuntimeError("No best checkpoint epoch was recorded")
        summary = {"completed_epochs": len(history), "best_epoch": self.best_epoch,
                   "best_fitness": self.best_fitness,
                   "early_stopped": len(history) < self.config["training"]["epochs"]}
        write_json(self.run_dir / "training_summary.json", summary)
        return summary
