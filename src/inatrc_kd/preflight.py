"""Periksa dependency/GPU serta bentuk output dan gradient model sebelum training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

INATRC_NAMES = [
    "Kelas-1 (Non-truk)",
    "Kelas-2 (Truk 2 sumbu)",
    "Kelas-3 (Truk 3 sumbu)",
    "Kelas-4 (Truk 4 sumbu)",
    "Kelas-5 (Truk >=5 sumbu)",
]


def _device(device: str, require_gpu: bool = False):
    import torch

    requested = str(device).lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif requested.isdecimal():
        requested = f"cuda:{requested}"
    selected = torch.device(requested)
    if require_gpu and selected.type != "cuda":
        raise RuntimeError("A CUDA GPU is required; select device='0' or 'cuda:0'.")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested GPU device {device!r}, but torch.cuda.is_available() is False. "
                f"Active Python: {sys.executable}; Torch: {torch.__version__}; CUDA: {torch.version.cuda}."
            )
        index = selected.index if selected.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist ({torch.cuda.device_count()} available).")
        selected = torch.device("cuda", index)
    elif selected.type != "cpu":
        raise ValueError("Stage A supports a single CPU or CUDA device: 'cpu', '0', or 'cuda:0'.")
    return selected


def _validate_runtime_requirements(packages=("torch", "torchvision")) -> list[dict[str, str]]:
    """Check active mandatory Requires-Dist entries, including a borrowed Kaggle CUDA stack."""
    from packaging.requirements import Requirement

    checked = []
    for package in packages:
        try:
            requirements = importlib.metadata.requires(package) or []
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Cannot inspect runtime dependency metadata for {package!r}.") from exc
        for declaration in requirements:
            requirement = Requirement(declaration)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                active_version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"Active {package} Requires-Dist {declaration!r} is not satisfied: "
                    f"{requirement.name} is missing. Reconcile the uv environment dependencies "
                    "without automatically replacing the inherited CUDA Torch stack."
                ) from exc
            if requirement.specifier and not requirement.specifier.contains(active_version, prereleases=True):
                raise RuntimeError(
                    f"Active {package} Requires-Dist {declaration!r} is not satisfied: "
                    f"active {requirement.name}=={active_version}. Reconcile the uv environment "
                    "dependencies without automatically replacing the inherited CUDA Torch stack."
                )
            checked.append({"package": package, "requirement": declaration, "active_version": active_version})
    return checked


def _validate_python_torch_pair(torch_version: str, vision_version: str, python_version=None):
    """Periksa pasangan resmi Torch/torchvision beserta dukungan versi Python."""
    from packaging.version import Version

    python_version = tuple(python_version or sys.version_info[:2])
    if not (3, 11) <= python_version < (3, 14):
        raise RuntimeError(f"Supported Python versions are 3.11, 3.12 and 3.13; found {python_version}.")
    torch_release, vision_release = Version(torch_version), Version(vision_version)
    torch_pair, vision_pair = torch_release.release[:2], vision_release.release[:2]
    if (torch_release.is_prerelease or torch_release.is_devrelease
            or vision_release.is_prerelease or vision_release.is_devrelease
            or torch_pair[0] != 2 or not 4 <= torch_pair[1] <= 13
            or vision_pair != (0, torch_pair[1] + 15)):
        raise RuntimeError(
            f"Unsupported Torch/torchvision pair: {torch_version}/{vision_version}. "
            "Require stable Torch 2.4 through 2.13 with torchvision 0.19 through 0.28 respectively."
        )
    # Tabel resmi torchvision membatasi pasangan 2.4–2.6 sampai Python 3.12.
    if python_version >= (3, 13) and torch_pair < (2, 7):
        raise RuntimeError(
            f"Python 3.13 requires Torch >=2.7 and torchvision >=0.22; found {torch_version}/{vision_version}. "
            "Use a compatible preinstalled runtime; do not automatically replace CUDA wheels."
        )
    return torch_pair, vision_pair


def environment_report(device: str = "cpu", require_gpu: bool = False,
                       dependency_checks: list[dict] | None = None) -> dict[str, Any]:
    """Check the active interpreter, Torch/CUDA, and torchvision NMS on the requested device."""
    import torch
    selected = _device(device, require_gpu)
    try:
        import torchvision
    except Exception as exc:
        raise RuntimeError(
            f"torchvision cannot load with active Torch {torch.__version__} from {torch.__file__}. "
            "Check the Torch/torchvision version pair and wheel build."
        ) from exc
    versions = {}
    for package in ("ultralytics", "wandb", "torch", "torchvision", "numpy", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    torch_pair, vision_pair = _validate_python_torch_pair(torch.__version__, torchvision.__version__)
    for package, expected in (("ultralytics", "8.4.155"), ("wandb", "0.28.1")):
        if versions[package] != expected:
            raise RuntimeError(f"Expected {package}=={expected}; active version is {versions[package]!r}.")
    requirement_checks = _validate_runtime_requirements() if dependency_checks is None else dependency_checks
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "torchvision_version": torchvision.__version__,
        "torchvision_file": torchvision.__file__,
        "package_versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": str(device),
        "resolved_device": str(selected),
        "torchvision_pair_compatible": True,
        "dependency_check_passed": True,
        "runtime_dependency_checks": requirement_checks,
        "torch_differs_from_local_cpu_lock": torch_pair != (2, 10) or vision_pair != (0, 25),
        "torch_runtime_policy": (
            "kaggle_inherited_cuda_pair" if Path("/kaggle/working").exists()
            else "local_cpu_lock" if torch_pair == (2, 10) and vision_pair == (0, 25)
            else "compatible_alternate_pair"
        ),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_gib": round(properties.total_memory / 2**30, 3),
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    try:
        # Jalankan operasi sungguhan; CUDA terdeteksi saja belum menjamin wheel cocok.
        import numpy as np
        probe = torch.tensor([1.0, 2.0], device=selected)
        result = float(probe.square().sum().cpu())
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]], device=selected)
        scores = torch.tensor([0.9, 0.8], device=selected)
        kept = torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist()
        if result != 5.0 or kept != [0, 1]:
            raise RuntimeError(f"Unexpected tensor/NMS output: {result}, {kept}")
        # Loader memakai NumPy ↔ Torch; periksa ABI setelah dependency venv terpasang.
        array = np.array([1.0, 2.0], dtype=np.float32)
        roundtrip = torch.from_numpy(array).to(selected).square().cpu().numpy()
        if not np.array_equal(roundtrip, np.array([1.0, 4.0], dtype=np.float32)):
            raise RuntimeError("Unexpected NumPy/Torch roundtrip output")
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)
        report["tensor_probe_passed"] = True
        report["torchvision_nms_passed"] = True
        report["numpy_torch_bridge_passed"] = True
    except Exception as exc:
        raise RuntimeError(
            f"Torch/torchvision/NumPy operation check failed on {selected}. "
            f"Torch {torch.__version__}, torchvision {torchvision.__version__}, "
            f"NumPy {versions['numpy']}, CUDA {torch.version.cuda}."
        ) from exc
    return report


def _model(scale: str, imgsz: int, device):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    if scale not in {"m", "s", "n"}:
        raise ValueError("Model scale must be 'm', 's', or 'n'.")
    if imgsz < 32 or imgsz % 32:
        raise ValueError("Diagnostic imgsz must be a positive multiple of 32.")
    model = DetectionModel(f"yolo26{scale}.yaml", nc=5, verbose=False).to(device)
    model.names = dict(enumerate(INATRC_NAMES))
    model.args = get_cfg(overrides={"imgsz": imgsz, "dis": 6.0, "device": str(device)})
    return model


def _batch(imgsz: int, device, batch_size: int = 2):
    import torch

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Diagnostic batch_size must be a positive integer.")
    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.6, 0.2, 0.3]], device=device)
    indices = torch.arange(batch_size, device=device)
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, device=device),
        "batch_idx": indices.float(),
        "cls": (indices % 5).float().view(-1, 1),
        "bboxes": boxes[indices % 2],
    }


def _tensor_shapes(value):
    import torch

    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: _tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_shapes(item) for item in value]
    return type(value).__name__


def _gradient_summary(parameters) -> dict[str, Any]:
    import torch

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    total = sum(float(gradient.detach().abs().sum().cpu()) for gradient in gradients)
    return {"tensors": len(gradients), "finite": finite, "absolute_sum": total, "nonzero": total > 0}


def inspect_model(
    scale: str = "n", imgsz: int = 160, device: str = "cpu", batch_size: int = 2, amp: bool = False
) -> dict[str, Any]:
    """Probe graph inputs, raw head branches, and loss/backward at the requested batch/input size."""
    import torch
    from ultralytics.nn.modules.head import Detect

    selected = _device(device)
    amp_enabled = bool(amp and selected.type == "cuda")
    model = _model(scale, imgsz, selected)
    head = next(module for module in model.model if isinstance(module, Detect))
    features = []

    def capture_inputs(module, inputs):
        features[:] = [list(feature.shape) for feature in inputs[0]]

    original_hooks = len(head._forward_pre_hooks)
    # Hook mengamati fitur P3/P4/P5 yang benar-benar masuk ke head Detect.
    handle = head.register_forward_pre_hook(capture_inputs)
    try:
        model.train()
        batch = _batch(imgsz, selected, batch_size)
        with torch.autocast(device_type=selected.type, enabled=amp_enabled):
            predictions = model(batch["img"])
            train_shapes = _tensor_shapes(predictions)
            loss, _ = model.loss(batch, predictions)
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError(f"YOLO26{scale} diagnostic detection loss is non-finite.")
        if amp_enabled:
            optimizer = torch.optim.AdamW(parameter for parameter in model.parameters() if parameter.requires_grad)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            scaler.scale(loss.sum()).backward()
            scaler.unscale_(optimizer)
        else:
            loss.sum().backward()
        gradient = _gradient_summary(model.parameters())
        if not gradient["finite"] or not gradient["nonzero"]:
            raise RuntimeError(f"YOLO26{scale} detection gradients failed: {gradient}")
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=selected.type, enabled=amp_enabled):
            eval_shapes = _tensor_shapes(model(batch["img"]))
        report = {
            "model": f"yolo26{scale}",
            "diagnostics_only": True,
            "pretrained": False,
            "device": str(selected),
            "batch": batch_size,
            "imgsz": imgsz,
            "amp_requested": bool(amp),
            "amp_enabled": amp_enabled,
            "nc": int(head.nc),
            "reg_max": int(head.reg_max),
            "end2end": bool(model.yaml.get("end2end", False)),
            "inference_end2end": bool(head.end2end),
            "detect_index": int(head.i),
            "detect_from": list(head.f),
            "strides": head.stride.cpu().tolist(),
            "features": features.copy(),
            "train_output_shapes": train_shapes,
            "eval_output_shapes": eval_shapes,
            "loss": float(loss.detach().sum().cpu()),
            "student_gradient": gradient,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    report["hooks_removed"] = len(head._forward_pre_hooks) == original_hooks
    if report["nc"] != 5 or report["reg_max"] != 1 or not report["end2end"] or not report["hooks_removed"]:
        raise RuntimeError(f"Unexpected YOLO26 head contract: {report}")
    return report


def _state_hash(model) -> str:
    fingerprint = hashlib.sha256()
    for name, value in model.state_dict().items():
        fingerprint.update(name.encode("utf-8"))
        fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
    return fingerprint.hexdigest()


def native_gradient_probe(
    student: str = "n", imgsz: int = 160, device: str = "cpu", amp: bool = False
) -> dict[str, Any]:
    """Exercise the official Native wrapper on two batches; this is not a research training run."""
    import torch
    from ultralytics.nn.distill_model import DistillationModel

    if student not in {"s", "n"}:
        raise ValueError("Native diagnostic student must be 's' or 'n'.")
    selected = _device(device)
    amp_enabled = bool(amp and selected.type == "cuda")
    student_model = _model(student, imgsz, selected)
    teacher_model = _model("m", imgsz, selected)
    wrapper = DistillationModel(teacher_model=teacher_model, student_model=student_model).train()
    teacher_hash = _state_hash(wrapper.teacher_model)
    registered = {id(parameter) for parameter in wrapper.parameters()}
    projectors_registered = all(id(parameter) in registered for parameter in wrapper.projector.parameters())
    optimizer = torch.optim.AdamW(parameter for parameter in wrapper.parameters() if parameter.requires_grad)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    projectors_in_optimizer = all(id(parameter) in optimized for parameter in wrapper.projector.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    losses, kd_losses, hook_counts = [], [], []
    student_gradients, projector_gradients = [], []
    teacher_grads_absent = True
    try:
        for _ in range(2):
            wrapper.zero_grad(set_to_none=True)
            with torch.autocast(device_type=selected.type, enabled=amp_enabled):
                loss, loss_items = wrapper(_batch(imgsz, selected))
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Official Native KD produced a non-finite diagnostic loss.")
            scaler.scale(loss.sum()).backward()
            if amp_enabled:
                scaler.unscale_(optimizer)
            student_gradient = _gradient_summary(wrapper.student_model.parameters())
            projector_gradient = _gradient_summary(wrapper.projector.parameters())
            if not all(item["finite"] and item["nonzero"] for item in (student_gradient, projector_gradient)):
                raise RuntimeError(f"Native KD gradient routing failed: {student_gradient}, {projector_gradient}")
            teacher_grads_absent &= all(parameter.grad is None for parameter in wrapper.teacher_model.parameters())
            losses.append(float(loss.detach().sum().cpu()))
            kd_losses.append(float(loss_items["dis_loss"].detach().sum().cpu()))
            student_gradients.append(student_gradient)
            projector_gradients.append(projector_gradient)
            hook_counts.append(sum(len(module._forward_hooks) for module in wrapper.modules()))
            if amp_enabled:
                scaler.update()
        report = {
            "method": "native",
            "student": f"yolo26{student}",
            "teacher": "yolo26m",
            "diagnostics_only": True,
            "pretrained": False,
            "device": str(selected),
            "batch": 2,
            "imgsz": imgsz,
            "amp_requested": bool(amp),
            "amp_enabled": amp_enabled,
            "feature_layers": wrapper.feats_idx,
            "projectors_registered": projectors_registered,
            "projectors_in_optimizer": projectors_in_optimizer,
            "teacher_frozen": all(not parameter.requires_grad for parameter in wrapper.teacher_model.parameters()),
            "teacher_eval": not wrapper.teacher_model.training,
            "teacher_grads_absent": teacher_grads_absent,
            "teacher_state_unchanged": _state_hash(wrapper.teacher_model) == teacher_hash,
            "losses": losses,
            "kd_losses": kd_losses,
            "student_gradients": student_gradients,
            "projector_gradients": projector_gradients,
            "hook_counts": hook_counts,
            "hook_count_stable": len(set(hook_counts)) == 1,
        }
    finally:
        wrapper._remove_feature_hooks()
        wrapper._teacher_feats.clear()
        wrapper._student_feats.clear()
        wrapper.zero_grad(set_to_none=True)
    report["hooks_after_cleanup"] = sum(len(module._forward_hooks) for module in wrapper.modules())
    gates = ("projectors_registered", "projectors_in_optimizer", "teacher_frozen", "teacher_eval", "teacher_grads_absent", "teacher_state_unchanged", "hook_count_stable")
    if not all(report[key] for key in gates) or report["hooks_after_cleanup"] != 0:
        raise RuntimeError(f"Official Native KD diagnostic contract failed: {report}")
    return report


def ordered_names(names) -> list[str]:
    """Normalize YOLO class names while retaining the meaning of class IDs."""
    if isinstance(names, dict):
        indexed = {int(index): str(name) for index, name in names.items()}
        if set(indexed) != set(range(len(indexed))):
            raise ValueError("Class names must have contiguous IDs starting at zero.")
        return [indexed[index] for index in range(len(indexed))]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise ValueError("Class names must be a list or an indexed mapping.")


def check_teacher(path, *, expected_nc: int = 5) -> dict[str, Any]:
    """Teacher InaTRC tetap lima kelas; diagnostik COCO8 meminta 80 secara eksplisit."""
    from ultralytics import YOLO

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint_path}")
    teacher = YOLO(str(checkpoint_path))
    model = teacher.model
    head = model.model[-1]
    yaml_file = str(model.yaml.get("yaml_file", getattr(model, "yaml_file", "")))
    family = Path(yaml_file).stem.lower()
    scale = str(model.yaml.get("scale", ""))
    if (
        family not in {"yolo26", "yolo26m"}
        or scale != "m"
        or teacher.task != "detect"
        or int(head.nc) != expected_nc
        or int(head.reg_max) != 1
        or getattr(head, "one2one_cv2", None) is None
    ):
        raise ValueError(
            f"Teacher must be YOLO26m fine-tuned for {expected_nc} classes; "
            f"found YAML={yaml_file!r}, scale={scale!r}, nc={getattr(head, 'nc', None)}."
        )
    names = ordered_names(model.names)
    if len(names) != expected_nc:
        raise ValueError(f"Teacher class-name count is {len(names)}; expected {expected_nc}.")
    fingerprint = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    checkpoint = teacher.ckpt or {}
    report = {
        "path": str(checkpoint_path),
        "sha256": fingerprint.hexdigest(),
        "size_bytes": checkpoint_path.stat().st_size,
        "family": "yolo26",
        "scale": scale,
        "nc": int(head.nc),
        "names": names,
        "diagnostics_only": bool(checkpoint.get("inatrc_diagnostics_only", False)),
        "diagnostic_metadata": checkpoint.get("inatrc_diagnostics", None),
        "benchmark_metadata": checkpoint.get("coco8_benchmark", None),
    }
    return report


def calibrate_diagnostic_bn(model, imgsz=160):
    """One unsupervised synthetic forward, no optimizer; only diagnostic BN buffers change."""
    import torch
    norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    momentum = [module.momentum for module in norms]
    parameter = next(model.parameters())
    try:
        model.train()
        for module in norms:
            module.momentum = 1.0
        with torch.no_grad():
            model(torch.rand(2, 3, imgsz, imgsz, device=parameter.device, dtype=parameter.dtype))
    finally:
        for module, value in zip(norms, momentum, strict=True):
            module.momentum = value
        model.eval()
    return model


def make_synthetic_teacher(path, seed: int = 42) -> Path:
    """Save a random, BN-calibrated YOLO26m for KD diagnosis, never a research teacher."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing teacher checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yolo = YOLO("yolo26m.yaml", verbose=False)
        yolo.model = DetectionModel(yolo.model.yaml.copy(), nc=5, verbose=False)
        yolo.model.names = dict(enumerate(INATRC_NAMES))
        calibrate_diagnostic_bn(yolo.model)
        metadata = {
            "diagnostics_only": True,
            "training_completed": False,
            "dataset": "synthetic",
            "initialization": "random",
            "batch_norm_calibration": "one_synthetic_batch_momentum_1_no_optimizer",
            "seed": int(seed),
            "model": "yolo26m",
            "nc": 5,
            "names": INATRC_NAMES,
            "note": "Untrained synthetic teacher for software diagnostics; prohibited for full/research runs.",
        }
        yolo.ckpt = {"inatrc_diagnostics_only": True, "inatrc_diagnostics": metadata}
        yolo.save(str(destination))
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    check_teacher(destination)
    return destination
