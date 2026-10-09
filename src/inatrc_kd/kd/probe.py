"""Synthetic custom-KD-only gradient/optimizer/hook checks for Stage B."""
from __future__ import annotations
import torch
from inatrc_kd.config import load_config
from inatrc_kd.preflight import (_model, _batch, _device, _state_hash, _gradient_summary,
                                calibrate_diagnostic_bn)
from .common import detect_head


def hook_count(model):
    return sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in model.modules())


def custom_gradient_probe(method, student="n", imgsz=160, device="cpu", amp=False, seed=42):
    from .crosskd import CrossKDModel
    from .csakd import CSAKDModel
    if method not in ("crosskd", "csakd") or student not in ("s", "n"):
        raise ValueError("Custom diagnostic requires crosskd/csakd and student s/n")
    selected = _device(device)
    torch.manual_seed(seed)
    config = load_config(method=method, student=student, seed=seed)
    source = _model(student, imgsz, selected)
    teacher = calibrate_diagnostic_bn(_model("m", imgsz, selected), imgsz)
    cls = CrossKDModel if method == "crosskd" else CSAKDModel
    wrapper = cls(teacher, source, config["kd"][method]).train()
    before = _state_hash(wrapper.teacher_model)
    optimizer = torch.optim.AdamW((p for p in wrapper.parameters() if p.requires_grad), lr=0.001)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    adapter_in_optimizer = all(id(p) in optimized for p in wrapper.projector.parameters())
    enabled = bool(amp and selected.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    live = []

    def capture(module, inputs, output):
        output.retain_grad()
        live.append(output)

    handles = [source.model[index].register_forward_hook(capture) for index in detect_head(source).f]
    counts, gradients, losses, components, shapes = [], [], [], [], []
    try:
        for _ in range(2):
            wrapper.zero_grad(set_to_none=True)
            live.clear()
            with torch.autocast(device_type=selected.type, enabled=enabled):
                loss, items = wrapper(_batch(imgsz, selected))
            if not bool(torch.isfinite(loss).all()) or float(items["dis_loss"]) <= 0:
                raise RuntimeError("Custom diagnostic needs finite detection and positive KD loss")
            scaler.scale(loss[-1]).backward()  # KD alone must train the student; GT loss cannot hide a broken route.
            if enabled:
                scaler.unscale_(optimizer)
            student_grad = _gradient_summary(wrapper.student_model.parameters())
            adapter_grad = _gradient_summary(wrapper.projector.parameters())
            neck_grad = [{"finite": bool(torch.isfinite(feature.grad).all()),
                          "nonzero": bool(feature.grad.abs().sum() > 0)} for feature in live]
            if not all(g["finite"] and g["nonzero"] for g in [student_grad, adapter_grad, *neck_grad]):
                raise RuntimeError(f"Custom KD-only routing failed: {student_grad}, {adapter_grad}, {neck_grad}")
            gradients.append({"student": student_grad, "adapter": adapter_grad, "neck": neck_grad})
            counts.append(hook_count(wrapper))
            losses.append(float(items["dis_loss"]))
            components.append({key: float(value) for key, value in items.items() if key.startswith("kd_")})
            shapes = [list(value.shape) for value in live]
            if wrapper._student_feats or wrapper._teacher_feats:
                raise RuntimeError("Custom KD retained hook tensors after forward")
            torch.nn.utils.clip_grad_norm_((p for p in wrapper.parameters() if p.requires_grad), 10.0)
            scaler.step(optimizer)
            scaler.update()
        result = {"method": method, "student": student, "teacher": "m", "seed": seed,
                  "diagnostics_only": True, "pretrained": False, "device": str(selected),
                  "amp_requested": amp, "amp_enabled": enabled, "imgsz": imgsz, "batch": 2,
                  "options": config["kd"][method], "feature_layers": wrapper.feats_idx,
                  "feature_shapes": shapes, "kd_losses": losses, "components": components,
                  "gradients": gradients, "adapter_in_optimizer": adapter_in_optimizer,
                  "adapter_parameters": sum(p.numel() for p in wrapper.projector.parameters()),
                  "teacher_frozen": all(not p.requires_grad for p in wrapper.teacher_model.parameters()),
                  "teacher_eval": not wrapper.teacher_model.training,
                  "teacher_grads_absent": all(p.grad is None for p in wrapper.teacher_model.parameters()),
                  "teacher_state_unchanged": _state_hash(wrapper.teacher_model) == before,
                  "hook_counts": counts, "hooks_stable": len(set(counts)) == 1,
                  "caches_empty": not wrapper._student_feats and not wrapper._teacher_feats}
    finally:
        for handle in handles:
            handle.remove()
        wrapper._remove_feature_hooks()
        wrapper._student_feats.clear()
        wrapper._teacher_feats.clear()
        live.clear()
    result["hooks_after_cleanup"] = hook_count(wrapper)
    if result["hooks_after_cleanup"] or not all(result[key] for key in (
            "adapter_in_optimizer", "teacher_frozen", "teacher_eval", "teacher_grads_absent",
            "teacher_state_unchanged", "hooks_stable", "caches_empty")):
        raise RuntimeError(f"Custom KD lifecycle failed: {result}")
    return result
