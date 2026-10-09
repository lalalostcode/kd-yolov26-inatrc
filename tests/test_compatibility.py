"""Small synthetic checks of the pinned official YOLO26 and Native KD interfaces."""

import hashlib
import importlib.metadata

import pytest
import torch

from inatrc_kd.kd import configure_kd
from inatrc_kd.preflight import (
    INATRC_NAMES,
    _validate_runtime_requirements,
    _validate_python_torch_pair,
    check_teacher,
    environment_report,
    inspect_model,
    make_synthetic_teacher,
    native_gradient_probe,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def diagnostic_teacher(tmp_path_factory):
    return make_synthetic_teacher(tmp_path_factory.mktemp("teacher") / "diagnostic_yolo26m.pt")


@pytest.mark.parametrize("scale,channels", [("m", [256, 512, 512]), ("s", [128, 256, 512]), ("n", [64, 128, 256])])
def test_yolo26_graph_loss_backward(scale, channels):
    result = inspect_model(scale=scale, imgsz=160, device="cpu")
    assert result["reg_max"] == 1
    assert result["end2end"] is True
    assert result["nc"] == 5
    assert result["strides"] == [8, 16, 32]
    assert [shape[1] for shape in result["features"]] == channels
    assert [shape[-2:] for shape in result["features"]] == [[20, 20], [10, 10], [5, 5]]
    assert set(result["train_output_shapes"]) == {"one2many", "one2one"}
    assert result["student_gradient"]["nonzero"]
    assert result["student_gradient"]["finite"]
    assert result["hooks_removed"]


@pytest.mark.parametrize("student", ["s", "n"])
def test_official_native_gradient_and_hook_lifecycle(student):
    result = native_gradient_probe(student=student, imgsz=160, device="cpu")
    assert result["projectors_registered"]
    assert result["projectors_in_optimizer"]
    assert result["teacher_frozen"] and result["teacher_eval"]
    assert result["teacher_grads_absent"] and result["teacher_state_unchanged"]
    assert result["hook_count_stable"]
    assert result["hooks_after_cleanup"] == 0
    assert all(item["nonzero"] and item["finite"] for item in result["projector_gradients"])
    assert all(item["nonzero"] and item["finite"] for item in result["student_gradients"])
    assert all(loss > 0 for loss in result["kd_losses"])


def test_synthetic_teacher_provenance_and_full_gate(diagnostic_teacher):
    teacher = check_teacher(diagnostic_teacher)
    assert teacher["scale"] == "m" and teacher["nc"] == 5
    assert teacher["names"] == INATRC_NAMES
    assert teacher["diagnostics_only"]
    assert teacher["sha256"] == hashlib.sha256(diagnostic_teacher.read_bytes()).hexdigest()
    config = {
        "method": "native", "stage": "student", "mode": "smoke",
        "dataset": {"names": INATRC_NAMES},
        "kd": {"teacher_checkpoint": str(diagnostic_teacher), "dis": 6.0},
    }
    assert configure_kd(config) == {"distill_model": str(diagnostic_teacher), "dis": 6.0}
    config["mode"] = "full"
    with pytest.raises(ValueError, match="diagnostics-only"):
        configure_kd(config)
    config["mode"] = "smoke"
    config["dataset"]["names"] = list(reversed(INATRC_NAMES))
    with pytest.raises(ValueError, match="class IDs/names differ"):
        configure_kd(config)


def test_missing_teacher_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        check_teacher(tmp_path / "missing.pt")


@pytest.mark.parametrize("method", ["crosskd", "csakd"])
def test_custom_methods_never_run_a_baseline(method):
    from inatrc_kd.config import load_config
    with pytest.raises(ValueError, match="requires.*teacher_checkpoint"):
        configure_kd(load_config(method=method))
    assert configure_kd({"method": "none"}) == {}


def test_active_cpu_environment():
    result = environment_report(device="cpu")
    assert result["torch_file"]
    assert result["tensor_probe_passed"] and result["torchvision_nms_passed"]
    assert result["numpy_torch_bridge_passed"]
    assert result["resolved_device"] == "cpu"
    assert result["dependency_check_passed"]
    assert result["runtime_dependency_checks"]


@pytest.mark.parametrize("python,torch_version,vision_version", [
    ((3, 11), "2.4.1+cu121", "0.19.1+cu121"),
    ((3, 12), "2.6.0+cu124", "0.21.0+cu124"),
    ((3, 13), "2.7.1+cu126", "0.22.1+cu126"),
    ((3, 13), "2.8.0+cu128", "0.23.0+cu128"),
    ((3, 13), "2.10.0+cpu", "0.25.0+cpu"),
    ((3, 13), "2.13.0+cu130", "0.28.0+cu130"),
])
def test_python_and_stable_cuda_cpu_pair_support(python, torch_version, vision_version):
    assert _validate_python_torch_pair(torch_version, vision_version, python) == (
        tuple(int(n) for n in torch_version.split(".")[:2]),
        tuple(int(n) for n in vision_version.split(".")[:2]),
    )


@pytest.mark.parametrize("python,torch_version,vision_version,message", [
    ((3, 10), "2.10.0", "0.25.0", "Supported Python"),
    ((3, 14), "2.10.0", "0.25.0", "Supported Python"),
    ((3, 13), "2.4.1", "0.19.1", "Python 3.13 requires"),
    ((3, 13), "2.5.1", "0.20.1", "Python 3.13 requires"),
    ((3, 13), "2.6.0", "0.21.0", "Python 3.13 requires"),
    ((3, 13), "2.8.0", "0.22.0", "Unsupported Torch/torchvision pair"),
    ((3, 13), "2.10.0.dev20251001+cu128", "0.25.0", "stable Torch"),
    ((3, 13), "2.10.0", "0.25.0rc1", "stable Torch"),
])
def test_python_torch_pair_rejects_incompatible_or_unpinned_runtime(python, torch_version, vision_version, message):
    with pytest.raises(RuntimeError, match=message):
        _validate_python_torch_pair(torch_version, vision_version, python)


def test_numpy_torch_abi_failure_stops_preflight(monkeypatch):
    def incompatible(array):
        raise RuntimeError("Numpy is not available")

    monkeypatch.setattr(torch, "from_numpy", incompatible)
    with pytest.raises(RuntimeError, match="Torch/torchvision/NumPy operation check failed"):
        environment_report(device="cpu")


@pytest.mark.parametrize("active_version", ["1.0", None])
def test_runtime_requires_dist_rejects_mismatch_or_missing(monkeypatch, active_version):
    monkeypatch.setattr(importlib.metadata, "requires", lambda package: ["inatrc-probe>=2"])

    def version(package):
        if active_version is None:
            raise importlib.metadata.PackageNotFoundError(package)
        return active_version

    monkeypatch.setattr(importlib.metadata, "version", version)
    with pytest.raises(RuntimeError, match="Requires-Dist.*not satisfied"):
        _validate_runtime_requirements(("torch",))


def test_runtime_requires_dist_ignores_optional_extra(monkeypatch):
    monkeypatch.setattr(
        importlib.metadata, "requires",
        lambda package: ["inatrc-probe>=2", "inatrc-optional>=99; extra == 'gpu'"],
    )

    def version(package):
        assert package == "inatrc-probe"
        return "2.1"

    monkeypatch.setattr(importlib.metadata, "version", version)
    assert _validate_runtime_requirements(("torch",)) == [
        {"package": "torch", "requirement": "inatrc-probe>=2", "active_version": "2.1"}
    ]


def test_requested_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Requested GPU device"):
        environment_report(device="0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available in this test environment")
def test_native_cuda_gradient_probe():
    assert environment_report(device="0", require_gpu=True)["torchvision_nms_passed"]
    result = native_gradient_probe(student="n", imgsz=160, device="0", amp=True)
    assert result["amp_enabled"] and result["teacher_state_unchanged"]


def test_requested_diagnostic_batch_size():
    result = inspect_model(scale="n", imgsz=160, device="cpu", batch_size=3, amp=False)
    assert result["batch"] == 3
    assert all(shape[0] == 3 for shape in result["features"])
    assert result["amp_enabled"] is False
