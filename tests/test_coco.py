"""COCO area metrics must preserve empty/invalid groups and native YOLO metrics."""
from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

import inatrc_kd.coco as coco_module
from inatrc_kd.coco import AreaAuditValidator, evaluate_coco_area, remap_coco_predictions
from inatrc_kd.pipeline import SafeDetectionValidator


def _fixture(sizes=(20, 50, 100), *, annotations=True, predictions=True):
    images, ground_truth_rows, detection_rows, counts = [], [], [], []
    for index, size in enumerate(sizes):
        filename = f"road-camera-{index}.jpg"
        images.append({"id": index, "file_name": filename, "height": 640, "width": 640})
        bbox = [10.0, 15.0, float(size), float(size)]
        if annotations:
            ground_truth_rows.append({"id": index + 1, "image_id": index, "category_id": 1,
                                      "bbox": bbox, "area": float(size * size), "iscrowd": 0})
        if predictions:
            detection_rows.append({"image_id": f"road-camera-{index}", "file_name": filename,
                                   "category_id": 1, "bbox": bbox, "score": 0.95})
        counts.append(int(predictions))
    ground_truth = {"images": images, "annotations": ground_truth_rows,
                    "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "motorcycle"}]}
    return ground_truth, detection_rows, counts


def test_perfect_predictions_original_area_and_empty_class():
    ground_truth, predictions, counts = _fixture()
    original = deepcopy((ground_truth, predictions, counts))
    result = evaluate_coco_area(ground_truth, predictions, counts)
    for key in ("AP", "AP50", "AP_S", "AP_M", "AP_L"):
        assert result[key] == pytest.approx(1.0)
    assert result["protocol"]["max_detections"] == [1, 10, 100]
    assert result["protocol"]["area_ranges_pixels_squared"] == [[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]]
    assert result["protocol"]["iou_thresholds"] == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    assert all(result["per_class"][1][key] is None for key in ("AP", "AP50", "AP_S", "AP_M", "AP_L"))
    assert original == (ground_truth, predictions, counts)


def test_empty_predictions_are_zero_for_ground_truth_groups():
    result = evaluate_coco_area(*_fixture(predictions=False))
    assert all(result[key] == 0.0 for key in ("AP", "AP50", "AP_S", "AP_M", "AP_L"))
    assert result["prediction_count"] == 0
    assert result["ground_truth_count"] == 3


@pytest.mark.parametrize("predictions", [False, True])
def test_no_ground_truth_has_null_metrics(predictions):
    result = evaluate_coco_area(*_fixture(annotations=False, predictions=predictions))
    assert all(result[key] is None for key in ("AP", "AP50", "AP_S", "AP_M", "AP_L"))
    assert all(row["AP"] is None for row in result["per_class"])


def test_empty_area_groups_have_null_metrics():
    result = evaluate_coco_area(*_fixture(sizes=(20,)))
    assert result["AP_S"] == pytest.approx(1)
    assert result["AP_M"] is None
    assert result["AP_L"] is None


@pytest.mark.parametrize("size,valid_groups", [(32, ("AP_S", "AP_M")), (96, ("AP_M", "AP_L"))])
def test_standard_coco_inclusive_area_boundaries(size, valid_groups):
    result = evaluate_coco_area(*_fixture(sizes=(size,)))
    for key in ("AP_S", "AP_M", "AP_L"):
        if key in valid_groups:
            assert result[key] == pytest.approx(1)
        else:
            assert result[key] is None


def test_numeric_and_nonnumeric_filename_ids_are_remapped_by_audited_order():
    ground_truth, predictions, counts = _fixture(sizes=(20, 50))
    ground_truth["images"][1]["file_name"] = "00042.jpg"
    predictions[1].update(file_name="00042.jpg", image_id=42)
    result = remap_coco_predictions(ground_truth, predictions, counts)
    assert [row["image_id"] for row in result] == [0, 1]
    assert [row["image_id"] for row in predictions] == ["road-camera-0", 42]
    assert evaluate_coco_area(ground_truth, predictions, counts)["AP"] == pytest.approx(1)


@pytest.mark.parametrize("counts", [[1], [0, 0, 0], [1, 1, 2], [-1, 2, 2], [True, 1, 1]])
def test_prediction_count_mismatches_fail(counts):
    ground_truth, predictions, _ = _fixture()
    with pytest.raises(ValueError, match="count|integer"):
        evaluate_coco_area(ground_truth, predictions, counts)


def test_wrong_prediction_order_fails_instead_of_silent_remap():
    ground_truth, predictions, counts = _fixture()
    predictions[0], predictions[1] = predictions[1], predictions[0]
    with pytest.raises(ValueError, match="source does not match"):
        evaluate_coco_area(ground_truth, predictions, counts)


def test_mixed_prediction_image_chunk_fails():
    ground_truth, predictions, _ = _fixture()
    with pytest.raises(ValueError, match="different images"):
        evaluate_coco_area(ground_truth, predictions, [2, 1, 0])


@pytest.mark.parametrize("target,key,value,match", [
    ("annotations", "area", -1, "area"),
    ("annotations", "bbox", [10, 10, 0, 20], "positive"),
    ("annotations", "image_id", 999, "unknown"),
    ("annotations", "category_id", 999, "unknown"),
    ("annotations", "bbox", [10, 10, float("nan"), 20], "finite"),
    ("predictions", "category_id", 999, "unknown"),
    ("predictions", "score", float("nan"), "finite"),
    ("predictions", "score", 2, "between"),
])
def test_invalid_coco_input_fails(target, key, value, match):
    ground_truth, predictions, counts = _fixture()
    rows = predictions if target == "predictions" else ground_truth[target]
    rows[0][key] = value
    with pytest.raises(ValueError, match=match):
        evaluate_coco_area(ground_truth, predictions, counts)


@pytest.mark.parametrize("key", ["images", "categories", "annotations"])
def test_duplicate_coco_ids_fail(key):
    ground_truth, predictions, counts = _fixture()
    ground_truth[key].append(deepcopy(ground_truth[key][0]))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_coco_area(ground_truth, predictions, counts)


@pytest.mark.parametrize("installed", [None, "1.6.6", "1.7.0"])
def test_requires_exact_pinned_coco_dependency(monkeypatch, installed):
    def probe(_):
        if installed is None:
            raise PackageNotFoundError
        return installed
    monkeypatch.setattr(coco_module, "version", probe)
    with pytest.raises(RuntimeError, match="requires faster-coco-eval==1.6.7"):
        evaluate_coco_area(*_fixture())


def _validator(ground_truth, predictions, counts):
    validator = AreaAuditValidator.__new__(AreaAuditValidator)
    validator.training = False
    validator.args = SimpleNamespace(save_json=True)
    validator.gdict = ground_truth
    validator.jdict = predictions
    validator.pred_counts = counts
    validator.is_custom_json = True
    validator.metrics = SimpleNamespace()
    return validator


def test_validator_keeps_native_statistics_and_evaluates_once(monkeypatch):
    validator = _validator(*_fixture())
    stats = {"fitness": 0.731, "metrics/mAP50-95(B)": 0.64, "metrics/mAP50(B)": 0.77}
    before = deepcopy(stats)
    assert validator.eval_json(stats) is stats
    assert stats == before
    assert validator.metrics.coco_area["AP"] == pytest.approx(1)
    monkeypatch.setattr(coco_module, "evaluate_coco_area", lambda *_: pytest.fail("COCO was evaluated twice"))
    assert validator.eval_json(stats) is stats


def test_validator_get_stats_evaluates_empty_detections(monkeypatch):
    validator = _validator(*_fixture(predictions=False))
    native_stats = {"fitness": 0.0, "metrics/mAP50-95(B)": 0.0, "metrics/mAP_small(B)": 0.0}
    monkeypatch.setattr(SafeDetectionValidator, "get_stats", lambda _: native_stats)
    result = validator.get_stats()
    assert result["fitness"] == 0.0
    assert "metrics/mAP_small(B)" not in result
    assert validator.metrics.coco_area["AP"] == 0.0


def test_validator_training_does_not_run_coco(monkeypatch):
    validator = _validator(*_fixture())
    validator.training = True
    monkeypatch.setattr(coco_module, "evaluate_coco_area", lambda *_: pytest.fail("COCO must be final-only"))
    stats = {"fitness": 0.25}
    assert validator.eval_json(stats) is stats


def test_validator_invalid_export_fails_clearly():
    ground_truth, predictions, _ = _fixture()
    validator = _validator(ground_truth, predictions, [0, 0, 0])
    with pytest.raises(RuntimeError, match="COCO area evaluation failed.*prediction counts"):
        validator.eval_json({"fitness": 0.0})


def test_validator_exports_original_names_and_dimensions(monkeypatch):
    validator = _validator({"images": []}, [], [])
    validator.build_gdict = True
    def export(self, _predictions, _batch):
        self.gdict["images"].extend([{"id": 4}, {"id": 9}])
    monkeypatch.setattr(SafeDetectionValidator, "update_metrics", export)
    validator.update_metrics([], {"im_file": ["/input/road north.jpg", "/input/0010.png"],
                                  "ori_shape": [(1080, 1920), (480, 640)]})
    assert validator.gdict["images"] == [
        {"id": 4, "file_name": "road north.jpg", "height": 1080, "width": 1920},
        {"id": 9, "file_name": "0010.png", "height": 480, "width": 640},
    ]


def test_validator_export_missing_image_fails(monkeypatch):
    validator = _validator({"images": []}, [], [])
    validator.build_gdict = True
    monkeypatch.setattr(SafeDetectionValidator, "update_metrics", lambda *_: None)
    with pytest.raises(RuntimeError, match="lost validation images"):
        validator.update_metrics([], {"im_file": ["/input/road.jpg"], "ori_shape": [(480, 640)]})


@pytest.mark.smoke
def test_actual_ultralytics_final_validation_returns_empty_prediction_report(tmp_path):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from inatrc_kd.data import create_synthetic_dataset, write_runtime_yaml

    dataset = create_synthetic_dataset(tmp_path / "source")
    data_yaml = write_runtime_yaml(dataset, tmp_path / "runtime.yaml")
    model = YOLO("yolo26n.yaml")
    model.model = DetectionModel("yolo26n.yaml", nc=5, verbose=False)
    result = model.val(validator=AreaAuditValidator, data=str(data_yaml), device="cpu", imgsz=160,
                       batch=2, workers=0, plots=False, save_json=True, nms=None, conf=1.0,
                       quantize=None, project=str(tmp_path / "validation"), name="empty")
    assert result.coco_area["prediction_count"] == 0
    assert result.coco_area["ground_truth_count"] > 0
    assert result.coco_area["AP"] == 0.0
    assert result.box.map == 0.0
    assert len(result.coco_area["per_class"]) == 5
    assert all(row["class_name"] is not None for row in result.coco_area["per_class"])
    assert not list(dataset.rglob("*.cache"))
