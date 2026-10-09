"""CSAKD dengan varian paper/penulis dan pasangan modul yang dipilih dari graph YOLO."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.nn.distill_model import DistillationModel
from .common import detect_head, configure_custom, align_channels, finite_options


def configure_csakd(config):
    options = config["kd"]["csakd"]
    finite_options(options, ("alpha", "beta"))
    if options["variant"] not in ("paper", "author"):
        raise ValueError("CSAKD variant must explicitly be paper or author")
    return configure_custom(config)


def module_indices(model):
    # Pilih pasangan backbone/neck dari graph YOLO, bukan indeks layer yang di-hardcode.
    boundary = len(model.yaml["backbone"])
    lateral = set()
    for module in model.model[boundary:]:
        refs = module.f if isinstance(module.f, list) else [module.f]
        lateral.update(index for index in refs if 0 <= index < boundary)
    neck = list(detect_head(model).f)
    result = sorted(lateral) + neck
    if len(lateral) != 3 or len(neck) != 3 or len(set(result)) != 6:
        raise ValueError("CSAKD requires three backbone lateral and three neck module pairs")
    return result


class InputHook:
    def __init__(self, cache, key):
        self.cache, self.key = cache, key

    def __call__(self, module, inputs):
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("CSAKD selected module must have one tensor input")
        self.cache[self.key] = inputs[0]


def csakd_losses(student, teacher, cross, variant="paper"):
    """AAWM + AGKA; BCHW row attention is an explicit dimensional adaptation."""
    if not (student.shape == teacher.shape == cross.shape):
        raise ValueError("CSAKD requires identical aligned BCHW shapes")
    student, teacher, cross = student.float(), teacher.detach().float(), cross.float()
    dimension = math.sqrt(student.shape[1])
    similarity = (teacher.mean((2, 3)) * student.mean((2, 3))).sum(1) / dimension
    # Varian paper dan kode penulis berbeda dalam attention, normalisasi, dan reduksi loss.
    if variant == "paper":
        attention = torch.softmax(teacher.relu() @ cross.relu().transpose(-2, -1), dim=-1) / dimension
        guided = student + attention @ student
        agka = torch.linalg.vector_norm((guided - teacher).flatten(1), dim=1).mean()
        gap = torch.linalg.vector_norm((cross - student).flatten(1), dim=1)
        supervision = ((1 - similarity.sigmoid()) * gap).mean()
    elif variant == "author":
        attention = torch.softmax((cross.relu() @ teacher.relu().transpose(-2, -1)) / dimension, dim=-1)
        guided = F.normalize((student + attention @ student) / 2, dim=1)
        agka = F.mse_loss(guided, F.normalize(teacher, dim=1))
        supervision = (1 - similarity.mean().sigmoid()) * F.mse_loss(
            F.normalize(student, dim=1), F.normalize(cross, dim=1))
    else:
        raise ValueError("CSAKD variant must explicitly be paper or author")
    return agka, supervision


class CSAKDModel(DistillationModel):
    @staticmethod
    def get_distill_layers(model):
        return module_indices(model) + [detect_head(model).i]

    def _register_feature_hooks(self):
        super()._register_feature_hooks()
        for model, cache, handles in ((self.student_model, self._student_feats, self._student_hooks),
                                      (self.teacher_model, self._teacher_feats, self._teacher_hooks)):
            if model is None:
                continue
            if self.feats_idx[:-1] != module_indices(model):
                raise ValueError("CSAKD teacher/student module graph does not match")
            for index in self.feats_idx[:-1]:
                module = model.model[index]
                for key, hook in list(module._forward_pre_hooks.items()):
                    if isinstance(hook, InputHook):
                        del module._forward_pre_hooks[key]
                handles.append(module.register_forward_pre_hook(InputHook(cache, ("input", index))))

    def __init__(self, teacher_model, student_model, options):
        super().__init__(teacher_model, student_model)
        self.options = dict(options)
        inputs, outputs = [], []
        for index in self.feats_idx[:-1]:
            si, ti = self._student_feats[("input", index)], self._teacher_feats[("input", index)]
            so, to = self._student_feats[index], self._teacher_feats[index]
            if si.shape[-2:] != ti.shape[-2:] or so.shape[-2:] != to.shape[-2:]:
                raise ValueError("CSAKD spatial mismatch; resizing is not allowed")
            inputs.append(align_channels(si.shape[1], ti.shape[1]))
            outputs.append(align_channels(so.shape[1], to.shape[1]))
        # Adapter input/output dibuat sekali sebelum optimizer, lalu dipakai ulang tiap batch.
        self.projector = nn.ModuleDict({"input": nn.ModuleList(inputs), "output": nn.ModuleList(outputs)}).to(
            next(student_model.parameters()).device)
        self._teacher_feats.clear()
        self._student_feats.clear()

    def kd_loss(self):
        targets = [self._teacher_feats[index] for index in self.feats_idx[:-1]]
        terms = []
        for position, (index, target) in enumerate(zip(self.feats_idx[:-1], targets, strict=True)):
            current = self.projector["output"][position](self._student_feats[index].float())
            incoming = self.projector["input"][position](self._student_feats[("input", index)].float())
            # Cross feature berasal dari input student; teacher beku, jalur gradient tetap aktif.
            cross = self.teacher_model.model[index](incoming)
            terms.append(csakd_losses(current, target, cross, self.options["variant"]))
        return {"kd_agka_loss": sum(pair[0] for pair in terms), "kd_cross_loss": sum(pair[1] for pair in terms)}

    def loss(self, batch, preds=None):
        from .common import combined_loss
        return combined_loss(self, batch, preds,
                             {"kd_agka_loss": self.options["alpha"], "kd_cross_loss": self.options["beta"]})
