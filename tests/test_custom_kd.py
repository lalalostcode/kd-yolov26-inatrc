"""Paper equations, KD-only routing, frozen teacher and lifecycle regression."""
import math
from copy import deepcopy

import pytest
import torch
from ultralytics.nn.distill_model import DistillationModel

from inatrc_kd.config import load_config
from inatrc_kd.kd import configure_kd
from inatrc_kd.kd.crosskd import quality_focal_loss, aligned_giou, decoded_xyxy, CrossKDModel
from inatrc_kd.kd.csakd import csakd_losses, CSAKDModel
from inatrc_kd.kd.probe import custom_gradient_probe, hook_count
from inatrc_kd.preflight import _model, _batch, calibrate_diagnostic_bn


def test_qfl_gamma_one_matches_paper_numeric_case_and_detaches_target():
    logits = torch.zeros(1, 1, requires_grad=True)
    target = torch.tensor([[math.log(3)]], requires_grad=True)
    value = quality_focal_loss(logits, target)
    assert float(value.detach()) == pytest.approx(math.log(2) / 4)
    value.sum().backward()
    assert logits.grad.abs().sum() > 0
    assert target.grad is None
    assert float(quality_focal_loss(target.detach(), target).sum()) == 0


def test_giou_equal_overlap_and_disjoint_geometries():
    source = torch.tensor([[0., 0., 2., 2.]]).expand(3, 4)
    target = torch.tensor([[0., 0., 2., 2.], [1., 0., 3., 2.], [4., 0., 6., 2.]])
    assert aligned_giou(source, target).tolist() == pytest.approx([1, 1 / 3, -1 / 3])


def test_direct_distance_decode_uses_live_stride_and_anchor_center():
    features = [torch.zeros(1, 2, 1, 1)]
    decoded = decoded_xyxy(torch.ones(1, 4, 1), features, torch.tensor([8.]))
    assert decoded.flatten().tolist() == [-4., -4., 12., 12.]


def test_csakd_paper_per_sample_aawm_and_literal_l2_norm():
    student = torch.ones(2, 2, 2, 3)
    teacher = torch.stack((torch.ones(2, 2, 3), torch.full((2, 2, 3), 2.)))
    cross = torch.zeros_like(student)
    agka, supervision = csakd_losses(student, teacher, cross, "paper")
    guided = 1 + 1 / math.sqrt(2)
    expected_agka = math.sqrt(12) * (abs(guided - 1) + abs(guided - 2)) / 2
    expected_cross = math.sqrt(12) * ((1 - torch.sigmoid(torch.tensor(math.sqrt(2)))) +
                                    (1 - torch.sigmoid(torch.tensor(2 * math.sqrt(2))))) / 2
    assert float(agka) == pytest.approx(expected_agka)
    assert float(supervision) == pytest.approx(float(expected_cross))


def test_author_variant_is_distinct_and_matches_normalized_code_case():
    student = torch.ones(2, 2, 2, 3)
    teacher = student * 2
    cross = torch.zeros_like(student)
    agka, supervision = csakd_losses(student, teacher, cross, "author")
    assert float(agka) == pytest.approx(0, abs=1e-7)
    assert float(supervision) == pytest.approx(0.5 * (1 - torch.sigmoid(torch.tensor(2 * math.sqrt(2)))))
    assert csakd_losses(student, teacher, cross, "paper")[0] > 0


@pytest.mark.parametrize("variant", ["paper", "author"])
def test_csakd_non_square_attention_has_cross_gradient_and_no_teacher_gradient(variant):
    torch.manual_seed(42)
    source = torch.rand(2, 3, 4, 7, requires_grad=True)
    teacher = torch.rand(2, 3, 4, 7, requires_grad=True)
    cross = torch.rand(2, 3, 4, 7, requires_grad=True)
    agka, supervision = csakd_losses(source, teacher, cross, variant)
    (agka + supervision).backward()
    assert source.grad.isfinite().all() and source.grad.abs().sum() > 0
    assert cross.grad.isfinite().all() and cross.grad.abs().sum() > 0
    assert teacher.grad is None


@pytest.mark.parametrize("method", ["crosskd", "csakd"])
@pytest.mark.parametrize("student", ["s", "n"])
def test_custom_kd_only_gradient_and_two_optimizer_steps(method, student, tmp_path):
    from inatrc_kd.io import write_json
    result = custom_gradient_probe(method, student)
    assert all(result[key] for key in ("adapter_in_optimizer", "teacher_frozen", "teacher_eval",
                                      "teacher_grads_absent", "teacher_state_unchanged", "hooks_stable", "caches_empty"))
    assert result["hooks_after_cleanup"] == 0
    assert all(value > 0 and math.isfinite(value) for value in result["kd_losses"])
    write_json(tmp_path / f"{method}-{student}-probe.json", result)


@pytest.mark.parametrize("method,cls", [("crosskd", CrossKDModel), ("csakd", CSAKDModel)])
def test_custom_serialization_and_hook_re_registration_are_stable(method, cls):
    device = torch.device("cpu")
    wrapper = cls(calibrate_diagnostic_bn(_model("m", 160, device)), _model("n", 160, device),
                  load_config()["kd"][method])
    try:
        initial = hook_count(wrapper)
        wrapper._register_feature_hooks()
        assert hook_count(wrapper) == initial
        wrapper.train()
        wrapper(_batch(160, device))
        copied = deepcopy(wrapper).eval()
        try:
            assert isinstance(copied, DistillationModel)
            assert hook_count(copied) == initial
            copied.teacher_model = None  # reproduce EMA's teacher-free validation path
            with torch.no_grad():
                loss, items = copied(_batch(160, device))
            assert loss.isfinite().all() and float(items["dis_loss"]) == 0
        finally:
            copied._remove_feature_hooks()
    finally:
        wrapper._remove_feature_hooks()
    assert hook_count(wrapper) == 0


@pytest.mark.parametrize("method", ["crosskd", "csakd"])
def test_invalid_or_missing_custom_teacher_never_becomes_baseline(method):
    config = load_config(method=method)
    config["kd"]["teacher_checkpoint"] = "missing-teacher.pt"
    with pytest.raises(FileNotFoundError):
        configure_kd(config)
    if method == "crosskd":
        config["kd"][method]["branch"] = "one2one"
    else:
        config["kd"][method]["variant"] = "generic-attention"
    with pytest.raises(ValueError):
        configure_kd(config)


@pytest.mark.parametrize("method,cls", [("crosskd", CrossKDModel), ("csakd", CSAKDModel)])
def test_cpu_autocast_kd_fp32_alignment_and_exception_cache_cleanup(method, cls, monkeypatch):
    device = torch.device("cpu")
    wrapper = cls(calibrate_diagnostic_bn(_model("m", 160, device)), _model("n", 160, device),
                  load_config()["kd"][method]).train()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, items = wrapper(_batch(160, device))
        assert loss.isfinite().all() and items["dis_loss"].dtype == torch.float32
        loss[-1].backward()

        def fail():
            raise RuntimeError("injected KD failure")

        monkeypatch.setattr(wrapper, "kd_loss", fail)
        with pytest.raises(RuntimeError, match="injected KD failure"):
            wrapper(_batch(160, device))
        assert not wrapper._teacher_feats and not wrapper._student_feats
    finally:
        wrapper._remove_feature_hooks()
    assert hook_count(wrapper) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/AMP unavailable in local CPU environment")
@pytest.mark.parametrize("method", ["crosskd", "csakd"])
def test_custom_cuda_amp(method):
    assert custom_gradient_probe(method, "n", device="0", amp=True)["amp_enabled"]
