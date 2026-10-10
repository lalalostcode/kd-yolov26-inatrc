"""COCO AP berdasarkan luas bbox asli; metrik utama Ultralytics tetap terpisah."""
from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import math
from numbers import Integral
from pathlib import Path

import numpy as np

from .pipeline import SafeDetectionValidator


COCO_EVAL_VERSION = "1.6.7"
COCO_AREA_RANGES = [[0, 1e10], [0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e10]]
COCO_AREA_LABELS = ["all", "small", "medium", "large"]
COCO_METRIC_KEYS = ("AP", "AP50", "AP_S", "AP_M", "AP_L")


def _coco_integer(value, context):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"COCO {context} must be an integer: {value!r}")
    return int(value)


def _coco_bbox(value, context):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"COCO {context} requires four bbox coordinates")
    if any(isinstance(number, bool) or not isinstance(number, (int, float)) for number in value):
        raise ValueError(f"COCO {context} bbox must contain finite numbers")
    bbox = [float(number) for number in value]
    if not all(math.isfinite(number) for number in bbox) or min(bbox[2:]) <= 0:
        raise ValueError(f"COCO {context} bbox must be finite with positive width/height")
    return bbox


def remap_coco_predictions(ground_truth, predictions, pred_counts):
    """Cocokkan urutan ekspor per gambar, termasuk nama file yang bukan angka."""
    images = ground_truth["images"]
    if len(pred_counts) != len(images):
        raise ValueError("COCO prediction-count length differs from the ground-truth image count")
    counts = [_coco_integer(count, "prediction count") for count in pred_counts]
    if any(count < 0 for count in counts) or sum(counts) != len(predictions):
        raise ValueError("COCO prediction counts do not match the exported detections")

    remapped, offset, seen_sources = [], 0, set()
    for image, count in zip(images, counts, strict=True):
        chunk = predictions[offset:offset + count]
        offset += count
        sources = {(row.get("file_name"), row.get("image_id")) for row in chunk}
        if len(sources) > 1:
            raise ValueError(f"COCO detections from different images were mixed for image {image['id']}")
        if sources:
            source = next(iter(sources))
            if source in seen_sources:
                raise ValueError("COCO detections repeat the same source image in different image groups")
            seen_sources.add(source)
            if image.get("file_name"):
                filename = Path(image["file_name"]).name
                stem = Path(filename).stem
                expected_id = int(stem) if stem.isnumeric() else stem
                if source != (filename, expected_id):
                    raise ValueError(f"COCO prediction source does not match ground-truth image {filename}")
        remapped.extend({**row, "image_id": image["id"]} for row in chunk)
    return remapped


def _validate_coco_inputs(ground_truth, predictions, pred_counts):
    if not isinstance(ground_truth, dict):
        raise ValueError("COCO ground truth must be a dictionary")
    for key in ("images", "annotations", "categories"):
        if not isinstance(ground_truth.get(key), list):
            raise ValueError(f"COCO ground truth requires a {key} list")
    if not isinstance(predictions, list):
        raise ValueError("COCO predictions must be a list")
    if not isinstance(pred_counts, (list, tuple)):
        raise ValueError("COCO prediction counts must be a list or tuple")

    ground_truth = deepcopy(ground_truth)
    image_ids, category_ids = set(), set()
    for key, target in (("images", image_ids), ("categories", category_ids)):
        for item in ground_truth[key]:
            item["id"] = _coco_integer(item["id"], f"{key} ID")
            if item["id"] in target:
                raise ValueError(f"COCO contains duplicate {key} IDs")
            target.add(item["id"])
    if not category_ids:
        raise ValueError("COCO ground truth requires at least one category")
    annotation_ids = set()
    for row in ground_truth["annotations"]:
        row["id"] = _coco_integer(row["id"], "annotation ID")
        row["image_id"] = _coco_integer(row["image_id"], "annotation image ID")
        row["category_id"] = _coco_integer(row["category_id"], "annotation category ID")
        if row["id"] in annotation_ids:
            raise ValueError("COCO contains duplicate annotation IDs")
        annotation_ids.add(row["id"])
        if row["image_id"] not in image_ids or row["category_id"] not in category_ids:
            raise ValueError("COCO annotation refers to an unknown image/category")
        row["bbox"] = _coco_bbox(row["bbox"], "annotation")
        area = float(row["area"])
        if not math.isfinite(area) or not math.isclose(area, row["bbox"][2] * row["bbox"][3], rel_tol=1e-6):
            raise ValueError("COCO annotation area must equal the original-pixel bbox area")
        row["area"] = area
        row["iscrowd"] = _coco_integer(row.get("iscrowd", 0), "iscrowd")
        if row["iscrowd"] not in (0, 1):
            raise ValueError("COCO iscrowd must be zero or one")

    remapped = remap_coco_predictions(ground_truth, predictions, pred_counts)
    for row in remapped:
        row["category_id"] = _coco_integer(row["category_id"], "prediction category ID")
        if row["category_id"] not in category_ids:
            raise ValueError("COCO prediction refers to an unknown category")
        row["bbox"] = _coco_bbox(row["bbox"], "prediction")
        row["score"] = float(row["score"])
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("COCO prediction score must be finite and between zero and one")
    return ground_truth, remapped


def _coco_mean_or_null(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values) & (values >= 0)]
    return float(valid.mean()) if valid.size else None


def evaluate_coco_area(ground_truth, predictions, pred_counts):
    """Hitung AP COCO terpisah; kelompok tanpa ground truth mempunyai nilai null."""
    try:
        installed = version("faster-coco-eval")
    except PackageNotFoundError:
        raise RuntimeError(f"COCO evaluation requires faster-coco-eval=={COCO_EVAL_VERSION}") from None
    if installed != COCO_EVAL_VERSION:
        raise RuntimeError(f"COCO evaluation requires faster-coco-eval=={COCO_EVAL_VERSION}; found {installed}")
    from faster_coco_eval import COCO, COCOeval_faster

    ground_truth, predictions = _validate_coco_inputs(ground_truth, predictions, pred_counts)
    annotation_api = COCO(ground_truth, use_deepcopy=True)
    if predictions:
        prediction_api = annotation_api.loadRes(predictions)
    else:
        # loadRes([]) pada versi pin mengakses elemen pertama; COCO kosong tetap sah.
        prediction_api = COCO({"images": deepcopy(ground_truth["images"]),
                               "categories": deepcopy(ground_truth["categories"]), "annotations": []})
    evaluator = COCOeval_faster(annotation_api, prediction_api, iouType="bbox")
    evaluator.params.imgIds = [image["id"] for image in ground_truth["images"]]
    evaluator.params.catIds = sorted(category["id"] for category in ground_truth["categories"])
    evaluator.params.iouThrs = np.linspace(0.5, 0.95, 10)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.params.areaRng = deepcopy(COCO_AREA_RANGES)
    evaluator.params.areaRngLbl = list(COCO_AREA_LABELS)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # precision: IoU, recall, kelas, kelompok luas, maxDets; -1 berarti tidak valid.
    precision = np.asarray(evaluator.eval["precision"])
    slices = (precision[:, :, :, 0, -1], precision[0, :, :, 0, -1],
              precision[:, :, :, 1, -1], precision[:, :, :, 2, -1], precision[:, :, :, 3, -1])
    report = {key: _coco_mean_or_null(values) for key, values in zip(COCO_METRIC_KEYS, slices, strict=True)}
    report["per_class"] = []
    names = {category["id"]: category.get("name") for category in ground_truth["categories"]}
    for position, category_id in enumerate(evaluator.params.catIds):
        class_slices = (precision[:, :, position, 0, -1], precision[0, :, position, 0, -1],
                        precision[:, :, position, 1, -1], precision[:, :, position, 2, -1],
                        precision[:, :, position, 3, -1])
        report["per_class"].append({"category_id": int(category_id), "class_name": names[category_id],
                                   **{key: _coco_mean_or_null(values) for key, values in
                                      zip(COCO_METRIC_KEYS, class_slices, strict=True)}})
    report.update(image_count=len(ground_truth["images"]), ground_truth_count=len(ground_truth["annotations"]),
                  prediction_count=len(predictions))
    report["protocol"] = {"evaluator": f"faster-coco-eval=={COCO_EVAL_VERSION}", "iou_type": "bbox",
                          "iou_thresholds": [round(float(value), 2) for value in evaluator.params.iouThrs],
                          "max_detections": [1, 10, 100], "area_ranges_pixels_squared": deepcopy(COCO_AREA_RANGES),
                          "area_labels": list(COCO_AREA_LABELS), "primary_metric": "Ultralytics mAP50_95 unchanged"}
    return report


class AreaAuditValidator(SafeDetectionValidator):
    """Validator final dengan laporan COCO tambahan, tanpa mengganti fitness native."""

    def init_metrics(self, model):
        self.gdict = None
        super().init_metrics(model)
        if self.gdict is not None:
            for class_id, category in enumerate(self.gdict["categories"]):
                category["name"] = self.names[class_id]
        self._coco_area_evaluated = False
        self.metrics.coco_area = None

    def update_metrics(self, preds, batch):
        image_offset = len(self.gdict["images"]) if self.build_gdict else 0
        super().update_metrics(preds, batch)
        if self.build_gdict:
            added = self.gdict["images"][image_offset:]
            if len(added) != len(batch["im_file"]):
                raise RuntimeError("COCO ground-truth export lost validation images")
            for image, filename, shape in zip(added, batch["im_file"], batch["ori_shape"], strict=True):
                image.update(file_name=Path(filename).name, height=int(shape[0]), width=int(shape[1]))

    def get_stats(self):
        stats = super().get_stats()
        # Placeholder nol upstream bukan laporan COCO; hindari dua versi metrik luas.
        for size in ("small", "medium", "large"):
            stats.pop(f"metrics/mAP_{size}(B)", None)
        if not self.training and self.args.save_json:
            self.eval_json(stats)  # Tetap jalan ketika seluruh prediksi kosong.
        return stats

    def eval_json(self, stats):
        if self.training or not self.args.save_json or getattr(self, "_coco_area_evaluated", False):
            return stats
        if not self.gdict or not self.is_custom_json:
            raise RuntimeError("COCO area audit requires the custom InaTRC validation export")
        try:
            self.metrics.coco_area = evaluate_coco_area(self.gdict, self.jdict, self.pred_counts)
        except Exception as error:
            raise RuntimeError(f"COCO area evaluation failed: {error}") from error
        self._coco_area_evaluated = True
        return stats
