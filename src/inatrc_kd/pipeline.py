"""Small isolated Ultralytics extensions to keep caches outside source data.

Dataset constructor arguments follow the pinned upstream build_yolo_dataset;
the detector, optimizer, losses and Native KD remain upstream implementations.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path

import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.torch_utils import unwrap_model

from .data import SafeYOLODataset


def safe_dataset(args, img_path, batch, data, *, mode, stride, cache_dir):
    if args.fraction != 1.0 or args.classes is not None or args.single_cls:
        raise ValueError("Stage A requires all audited images/classes; filtering is disabled")
    if Path(str(img_path)).resolve() == (Path(data["path"]) / "test" / "images").resolve():
        raise ValueError("Stage A never constructs a test-set dataloader")
    return SafeYOLODataset(
        img_path=img_path, imgsz=args.imgsz, batch_size=batch, augment=mode == "train",
        hyp=copy(args), rect=args.rect or mode == "val", cache=False,
        single_cls=False, stride=stride, pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ", task="detect", classes=None, data=data, fraction=1.0,
        audit_cache_dir=cache_dir,
    )


class SafeDetectionValidator(DetectionValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        if self.args.split != "val":
            raise ValueError("Stage A allows validation split only")
        return safe_dataset(self.args, img_path, batch or self.args.batch, self.data,
                            mode="val", stride=self.stride, cache_dir=self.save_dir / "cache")


class SafeDetectionTrainer(DetectionTrainer):
    def run_callbacks(self, event: str):
        super().run_callbacks(event)
        if event == "on_train_epoch_start":
            # v8.4.155 only retries OOM when this counter is below three.
            # Set it after upstream initializes it, before any forward/backward.
            # Thus OOM raises before upstream can halve the explicit batch.
            self._oom_retries = 3

    def _build_train_pipeline(self):
        if getattr(self, "_oom_retries", 0):
            raise RuntimeError("OOM recovery is disabled; keep the configured batch and start a separate run")
        return super()._build_train_pipeline()

    def _handle_nan_recovery(self, epoch):
        values = [self.loss, self.fitness]
        if isinstance(self.loss_items, dict):
            values.extend(self.loss_items.values())
        for value in values:
            if value is not None and not bool(torch.isfinite(torch.as_tensor(value)).all()):
                raise RuntimeError(f"Nonfinite loss/fitness at epoch {epoch + 1}; automatic recovery is disabled")
        return False

    def build_dataset(self, img_path, mode="train", batch=None):
        model = unwrap_model(self.model) if self.model is not None else None
        stride = max(int(model.stride.max()) if model is not None else 0, 32)
        return safe_dataset(self.args, img_path, batch or self.batch_size, self.data,
                            mode=mode, stride=stride, cache_dir=self.save_dir / "cache")

    def get_validator(self):
        original = super().get_validator()
        return SafeDetectionValidator(dataloader=original.dataloader, save_dir=original.save_dir,
                                      args=original.args, _callbacks=original.callbacks)
