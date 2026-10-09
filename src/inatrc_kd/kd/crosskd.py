"""YOLO26 CrossKD adaptation: intermediate student head -> frozen teacher suffix."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.distill_model import DistillationModel, FeatureHook
from ultralytics.utils.tal import make_anchors, dist2bbox
from .common import detect_head, configure_custom, align_channels, finite_options


def configure_crosskd(config):
    options = config["kd"]["crosskd"]
    finite_options(options, ("cls_weight", "box_weight", "focal_gamma"))
    if options["branch"] != "one2many" or options["split_index"] != 1:
        raise ValueError("CrossKD pilot requires one2many, split_index=1; other routes are not validated")
    return configure_custom(config)


def quality_focal_loss(logits, target_logits, gamma=1.0):
    target = target_logits.detach().float().sigmoid()
    pred = logits.float()
    return F.binary_cross_entropy_with_logits(pred, target, reduction="none") * (pred.sigmoid() - target).abs().pow(gamma)


def scale_align(student, teacher):
    """Per-channel B/H/W statistics as official CrossKD align_scale."""
    student, teacher = student.float(), teacher.detach().float()
    axes = (0, 2, 3)
    return ((student - student.mean(axes, keepdim=True)) / (student.std(axes, keepdim=True) + 1e-6) *
            teacher.std(axes, keepdim=True) + teacher.mean(axes, keepdim=True))


def decoded_xyxy(raw_boxes, features, strides):
    anchors, stride = make_anchors(features, strides, 0.5)
    return dist2bbox(raw_boxes.float().transpose(1, 2), anchors.float(), xywh=False, dim=-1) * stride.float()


def aligned_giou(pred, target):
    # Ordering is only for the KD geometry; detection predictions/GT loss stay unchanged.
    p0, p1 = torch.minimum(pred[..., :2], pred[..., 2:]), torch.maximum(pred[..., :2], pred[..., 2:])
    t0, t1 = torch.minimum(target[..., :2], target[..., 2:]), torch.maximum(target[..., :2], target[..., 2:])
    ap, at = (p1 - p0).prod(-1), (t1 - t0).prod(-1)
    intersection = (torch.minimum(p1, t1) - torch.maximum(p0, t0)).clamp_min(0).prod(-1)
    union = ap + at - intersection
    enclosing = (torch.maximum(p1, t1) - torch.minimum(p0, t0)).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-7) - (enclosing - union) / enclosing.clamp_min(1e-7)


class CrossKDModel(DistillationModel):
    """Reuse upstream lifecycle/EMA stripping; replace only custom KD computation."""
    def __init__(self, teacher_model, student_model, options):
        super().__init__(teacher_model, student_model)
        self.options = dict(options)
        adapters = {}
        for branch in ("box", "cls"):
            for level in range(detect_head(self.student_model).nl):
                source, target = self._student_feats[(branch, level)], self._teacher_feats[(branch, level)]
                if source.shape[-2:] != target.shape[-2:]:
                    raise ValueError("CrossKD spatial mismatch; resizing is not allowed")
                adapters[f"{branch}_{level}"] = align_channels(source.shape[1], target.shape[1])
        self.projector = nn.ModuleDict(adapters).to(next(student_model.parameters()).device)
        self._student_feats.clear()
        self._teacher_feats.clear()

    def _register_feature_hooks(self):
        super()._register_feature_hooks()
        for model, cache, handles in ((self.student_model, self._student_feats, self._student_hooks),
                                      (self.teacher_model, self._teacher_feats, self._teacher_hooks)):
            if model is None:
                continue
            head = detect_head(model)
            if head.reg_max != 1 or list(head.f) != self.feats_idx[:-1]:
                raise ValueError("CrossKD requires matching YOLO26 Detect graph and reg_max=1")
            for name, blocks in (("box", head.cv2), ("cls", head.cv3)):
                for level, block in enumerate(blocks):
                    self._clear_feature_hooks(block[0])
                    handles.append(block[0].register_forward_hook(FeatureHook(cache, (name, level))))

    def kd_loss(self):
        head = detect_head(self.teacher_model)
        target = self.decouple_outputs(self._teacher_feats[self.feats_idx[-1]], branch="one2many")
        outputs = {}
        for name, blocks in (("box", head.cv2), ("cls", head.cv3)):
            parts = []
            for level, block in enumerate(blocks):
                feature = self.projector[f"{name}_{level}"](self._student_feats[(name, level)].float())
                feature = scale_align(feature, self._teacher_feats[(name, level)])
                # Teacher frozen/eval, with autograd active through its suffix.
                for layer in list(block.children())[1:]:
                    feature = layer(feature)
                parts.append(feature.flatten(2))
            outputs[name] = torch.cat(parts, dim=2)
        target_logits = target["scores"].detach()
        confidence = target_logits.float().sigmoid().amax(1)
        classification = quality_focal_loss(outputs["cls"], target_logits, self.options["focal_gamma"]).mean()
        features = [self._teacher_feats[index] for index in self.feats_idx[:-1]]
        boxes = decoded_xyxy(outputs["box"], features, head.stride)
        teacher_boxes = decoded_xyxy(target["boxes"].detach(), features, head.stride)
        regression = ((1 - aligned_giou(boxes, teacher_boxes)) * confidence).sum() / confidence.sum().clamp_min(1e-7)
        return {"kd_cls_loss": classification, "kd_box_loss": regression}

    def loss(self, batch, preds=None):
        from .common import combined_loss
        return combined_loss(self, batch, preds,
                             {"kd_cls_loss": self.options["cls_weight"], "kd_box_loss": self.options["box_weight"]})
