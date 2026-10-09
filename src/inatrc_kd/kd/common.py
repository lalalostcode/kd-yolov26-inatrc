"""Shared custom lifecycle; official Native implementation stays untouched."""
from __future__ import annotations
import math
import torch
from torch import nn
from ultralytics.nn.modules.head import Detect

SOURCE_REVISIONS = {
    "crosskd": {"repository": "https://github.com/jbwang1997/CrossKD",
                "commit": "87e156a46259e3a516b65fc6a049b200b9ab0d54"},
    "csakd": {"repository": "https://github.com/KefanZhan/YOLOv8-KD",
              "commit": "3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8"},
    "ultralytics": {"repository": "https://github.com/ultralytics/ultralytics", "tag": "v8.4.155",
                    "commit": "590427b052171d75dadd813d5f2a9d49a94dfa9d"},
}


def detect_head(model):
    heads = [module for module in model.model if isinstance(module, Detect)]
    if len(heads) != 1:
        raise ValueError("Custom KD requires exactly one Detect head")
    return heads[0]


def align_channels(source, target):
    return nn.Identity() if source == target else nn.Conv2d(source, target, 1, bias=False)


def finite_options(options, names):
    for name in names:
        value = float(options[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"KD {name} must be finite and positive")


def configure_custom(config):
    from .native import native_train_kwargs
    kwargs = native_train_kwargs(config)
    return {"distill_model": kwargs["distill_model"]}


def combined_loss(wrapper, batch, preds, weights):
    """Detection loss once; release graph-holding hook caches even on exceptions."""
    wrapper._teacher_feats.clear()
    wrapper._student_feats.clear()
    try:
        if not wrapper.training:
            regular, items = wrapper.student_model.loss(batch, preds)
            parts = {key: regular.new_zeros(()) for key in weights}
        else:
            with torch.no_grad():
                wrapper.teacher_model(batch["img"])
            preds = wrapper.student_model(batch["img"])
            regular, items = wrapper.student_model.loss(batch, preds)
            with torch.autocast(device_type=batch["img"].device.type, enabled=False):
                parts = wrapper.kd_loss()
        total = sum(parts[key] * weights[key] for key in weights)
        if not bool(torch.isfinite(total)):
            raise RuntimeError("Custom KD loss is not finite; KD will not be disabled")
        items.update({key: value.detach() for key, value in parts.items()})
        items["dis_loss"] = total.detach()
        return torch.cat((regular, total.reshape(1) * batch["img"].shape[0])), items
    finally:
        wrapper._teacher_feats.clear()
        wrapper._student_feats.clear()
