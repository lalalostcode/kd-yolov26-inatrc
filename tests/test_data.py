"""Regression tests for original data preservation and audit semantics."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from inatrc_kd.data import (
    CLASS_NAMES,
    DatasetAuditError,
    audit_dataset,
    create_synthetic_dataset,
    write_runtime_yaml,
    verify_dataset_unchanged,
)


@pytest.fixture
def dataset(tmp_path):
    return create_synthetic_dataset(tmp_path / "synthetic", seed=42)


@pytest.fixture(autouse=True)
def isolate_library_settings(monkeypatch, tmp_path):
    # Imports may create settings/font caches; keep these inside test outputs.
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


def _source_snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _first_label(root):
    return sorted((root / "train" / "labels").glob("*.txt"))[0]


def test_synthetic_layout_fingerprint_and_source_preservation(dataset, tmp_path):
    before = _source_snapshot(dataset)
    report = audit_dataset(dataset, tmp_path / "reports", expected_images=20)
    assert _source_snapshot(dataset) == before
    assert report["total_images"] == report["total_boxes"] == 20
    assert [row["images"] for row in report["splits"]] == [10, 5, 5]
    assert report["class_names"] == list(CLASS_NAMES)
    assert all(row["boxes"] > 0 for row in report["class_distribution"])
    assert report["cross_split_exact_duplicates"] == []
    assert report["near_duplicate_or_sequence_leakage_checked"] is False
    assert report["fingerprint"] == audit_dataset(dataset, expected_images=20)["fingerprint"]
    saved = json.loads((tmp_path / "reports" / "dataset_audit.json").read_text(encoding="utf-8"))
    assert saved == report
    assert (tmp_path / "reports" / "cross_split_exact_duplicates.csv").read_text().strip() == "sha256,occurrences"


def test_fingerprint_preserves_baseline_byte_serialization(dataset):
    digest = hashlib.sha256()
    for split in ("train", "val", "test"):
        for image in sorted((dataset / split / "images").iterdir(), key=lambda p: p.stem):
            label = dataset / split / "labels" / f"{image.stem}.txt"
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            digest.update(f"{split}/{image.name}\n{image_hash}\n{label.name}\n".encode())
            digest.update(label.read_bytes())
    assert audit_dataset(dataset, expected_images=20)["fingerprint"] == digest.hexdigest()
    previous = digest.hexdigest()
    _first_label(dataset).write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    assert audit_dataset(dataset, expected_images=20)["fingerprint"] != previous


def test_synthetic_seed_reproducibility(dataset, tmp_path):
    same = create_synthetic_dataset(tmp_path / "same", seed=42)
    other = create_synthetic_dataset(tmp_path / "other", seed=123)
    assert _source_snapshot(same) == _source_snapshot(dataset)
    assert _source_snapshot(other) != _source_snapshot(dataset)
    with pytest.raises(DatasetAuditError, match="new or empty"):
        create_synthetic_dataset(dataset)


def test_empty_and_metadata_labels_allowed(dataset):
    _first_label(dataset).write_text("", encoding="utf-8")
    (dataset / "train" / "labels" / "classes.txt").write_text("metadata", encoding="utf-8")
    (dataset / "train" / "labels" / "_classes.txt").write_text("metadata", encoding="utf-8")
    report = audit_dataset(dataset, expected_images=20)
    assert report["splits"][0]["empty_labels"] == 1
    assert report["total_boxes"] == 19
    assert report["splits"][0]["labels"] == 10


@pytest.mark.parametrize("annotation, message", [
    ("0 0.1 0.1 0.2 0.2 0.3 0.3", "Polygon/segmentation"),
    ("0 0.5 0.5 0.5", "5 columns"),
    ("text 0.5 0.5 0.5 0.5", "nonnumeric"),
    ("0 nan 0.5 0.5 0.5", "non-finite"),
    ("0 inf 0.5 0.5 0.5", "non-finite"),
    ("5 0.5 0.5 0.5 0.5", "invalid class ID"),
    ("-1 0.5 0.5 0.5 0.5", "invalid class ID"),
    ("0.5 0.5 0.5 0.5 0.5", "invalid class ID"),
    ("0 0.5 0.5 0 0.5", "invalid normalized"),
    ("0 0.99 0.5 0.5 0.5", "invalid normalized"),
    ("0 0.5 0.5 0.5 0.5\n0.0 0.5 0.5 0.5 0.5", "duplicate label row"),
    ("0 0.5 0.5 0.5 0.5\n\n1 0.5 0.5 0.5 0.5", "5 columns"),
])
def test_invalid_annotation_fails_with_filename(dataset, annotation, message):
    label = _first_label(dataset)
    label.write_text(annotation, encoding="utf-8")
    with pytest.raises(DatasetAuditError, match=message) as captured:
        audit_dataset(dataset, expected_images=20)
    assert label.name in str(captured.value)


def test_baseline_bbox_edge_tolerance(dataset):
    _first_label(dataset).write_text("0 0.24995 0.5 0.5 0.5", encoding="utf-8")
    assert audit_dataset(dataset, expected_images=20)["total_boxes"] == 20
    _first_label(dataset).write_text("0 0.2498 0.5 0.5 0.5", encoding="utf-8")
    with pytest.raises(DatasetAuditError, match="invalid normalized"):
        audit_dataset(dataset, expected_images=20)


def test_expected_count_is_configurable(dataset):
    with pytest.raises(DatasetAuditError, match="Expected 3250 images, found 20"):
        audit_dataset(dataset)
    assert audit_dataset(dataset, expected_images=None)["total_images"] == 20


@pytest.mark.parametrize("change, message", [
    ("missing", "missing labels"), ("orphan", "orphan labels"),
    ("stem", "Duplicate image stem"), ("nested_images", "Nested dataset"),
    ("nested_labels", "Nested dataset"), ("unsupported", "Unsupported image"),
])
def test_layout_errors_are_not_silently_skipped(dataset, change, message):
    label = _first_label(dataset)
    if change == "missing":
        label.unlink()
    elif change == "orphan":
        (label.parent / "orphan.txt").write_text("", encoding="utf-8")
    elif change == "stem":
        image = dataset / "train" / "images" / f"{label.stem}.png"
        shutil.copyfile(image, image.with_suffix(".jpg"))
    elif change.startswith("nested"):
        (dataset / "train" / change.removeprefix("nested_") / "nested").mkdir()
    else:
        (dataset / "train" / "images" / "cached.npy").write_bytes(b"not an image")
    with pytest.raises(DatasetAuditError, match=message):
        audit_dataset(dataset, expected_images=20)


def test_exact_cross_split_duplicates_are_reported_without_mutation(dataset):
    source = sorted((dataset / "train" / "images").iterdir())[0]
    destination = sorted((dataset / "val" / "images").iterdir())[0]
    shutil.copyfile(source, destination)
    before = _source_snapshot(dataset)
    report = audit_dataset(dataset, expected_images=20)
    assert len(report["cross_split_exact_duplicates"]) == 1
    assert report["warnings"]
    assert {entry["split"] for entry in report["cross_split_exact_duplicates"][0]["occurrences"]} == {"train", "val"}
    assert _source_snapshot(dataset) == before


def test_corrupt_image_rejected_without_repair(dataset):
    image = sorted((dataset / "train" / "images").iterdir())[0]
    image.write_bytes(b"corrupt image")
    before = _source_snapshot(dataset)
    with pytest.raises(DatasetAuditError, match="Corrupt or unreadable image"):
        audit_dataset(dataset, expected_images=20)
    assert _source_snapshot(dataset) == before


def test_jpeg_missing_eof_rejected_without_repair(dataset):
    from PIL import Image

    image = sorted((dataset / "train" / "images").iterdir())[0]
    jpeg = image.with_suffix(".jpg")
    with Image.open(image) as loaded:
        loaded.save(jpeg)
    image.unlink()
    jpeg.write_bytes(jpeg.read_bytes()[:-2])
    before = _source_snapshot(dataset)
    with pytest.raises(DatasetAuditError, match="JPEG EOF|Corrupt or unreadable"):
        audit_dataset(dataset, expected_images=20)
    assert _source_snapshot(dataset) == before


def test_outputs_and_runtime_yaml_stay_outside_source(dataset, tmp_path):
    import yaml

    before = _source_snapshot(dataset)
    with pytest.raises(DatasetAuditError, match="outside the source"):
        audit_dataset(dataset, dataset / "reports", expected_images=20)
    with pytest.raises(DatasetAuditError, match="outside the source"):
        write_runtime_yaml(dataset, dataset / "data.yaml")
    runtime = write_runtime_yaml(dataset, tmp_path / "outputs" / "data.yaml")
    config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    assert config["nc"] == 5
    assert config["names"] == list(CLASS_NAMES)
    assert config["val"] == str(dataset / "val" / "images")
    assert _source_snapshot(dataset) == before


def test_safe_loader_redirects_cache_and_preserves_sources(dataset, tmp_path):
    from ultralytics.cfg import get_cfg
    from inatrc_kd.data import SafeYOLODataset

    before = _source_snapshot(dataset)
    arguments = dict(
        img_path=str(dataset / "train" / "images"), imgsz=64, batch_size=2,
        data={"path": str(dataset), "nc": 5, "names": dict(enumerate(CLASS_NAMES))},
        augment=False, cache=False, hyp=get_cfg(), task="detect", audit_cache_dir=tmp_path / "cache",
    )
    loaded = SafeYOLODataset(**arguments)
    assert len(loaded) == 10
    assert sum(len(label["cls"]) for label in loaded.labels) == 10
    assert list((tmp_path / "cache").glob("*.cache"))
    assert len(SafeYOLODataset(**arguments)) == 10  # exercise cached reload as well
    assert _source_snapshot(dataset) == before
    assert not list(dataset.rglob("*.cache"))
    assert not list(dataset.rglob("*.npy"))


def test_safe_loader_rejects_silent_image_exclusion(dataset, tmp_path, monkeypatch):
    from ultralytics.data.dataset import YOLODataset
    from inatrc_kd.data import SafeYOLODataset

    def corrupt_scan(self, cache_path, cache_hash):
        return {"labels": [], "results": (10, 0, 0, 1, 10), "msgs": ["corrupt"]}, False

    monkeypatch.setattr(YOLODataset, "_load_or_scan_cache", corrupt_scan)
    probe = SafeYOLODataset.__new__(SafeYOLODataset)
    probe.im_files = [str(path) for path in sorted((dataset / "train" / "images").iterdir())]
    probe.label_files = [str(dataset / "train" / "labels" / f"{Path(name).stem}.txt") for name in probe.im_files]
    probe.audit_cache_dir = tmp_path / "cache"
    probe.audited_files = None
    with pytest.raises(DatasetAuditError, match="scan changed audited images/boxes"):
        probe._load_or_scan_cache(dataset / "train" / "labels.cache", "hash")


def test_safe_loader_rejects_disk_cache(dataset, tmp_path):
    from inatrc_kd.data import SafeYOLODataset

    with pytest.raises(DatasetAuditError, match="Disk image caching"):
        SafeYOLODataset(data={"path": str(dataset)}, audit_cache_dir=tmp_path / "cache", cache="disk")


def test_end_verification_reads_bytes_without_repeating_decode(dataset, monkeypatch):
    import inatrc_kd.data as module
    report = audit_dataset(dataset, expected_images=20)
    def forbidden(*args, **kwargs):
        raise AssertionError("Identical bytes do not need repeated decode/annotation parsing")
    monkeypatch.setattr(module, "_verify_image", forbidden)
    monkeypatch.setattr(module, "_label_rows", forbidden)
    verify_dataset_unchanged(report)


@pytest.mark.parametrize("kind", ["image", "label", "pair-name", "missing", "added"])
def test_end_verification_rejects_changed_bytes_and_inventory(dataset, kind):
    import os
    report = audit_dataset(dataset, expected_images=20)
    image = sorted((dataset / "train" / "images").iterdir())[0]
    label = dataset / "train" / "labels" / f"{image.stem}.txt"
    if kind in ("image", "label"):
        path = image if kind == "image" else label
        previous = path.stat()
        content = bytearray(path.read_bytes())
        content[-2] ^= 1
        path.write_bytes(content)
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    elif kind == "pair-name":
        image.rename(image.with_name("renamed.png"))
        label.rename(label.with_name("renamed.txt"))
    elif kind == "missing":
        image.unlink()
    else:
        shutil.copyfile(image, image.with_name("added.png"))
        shutil.copyfile(label, label.with_name("added.txt"))
    with pytest.raises(DatasetAuditError, match="changed"):
        verify_dataset_unchanged(report)


def test_loader_skips_decode_only_for_matching_fresh_audit(dataset, tmp_path, monkeypatch):
    import inatrc_kd.data as module
    from ultralytics.cfg import get_cfg
    report = audit_dataset(dataset, expected_images=20)
    monkeypatch.setattr(module, "_verify_image", lambda path: pytest.fail("Repeated decode"))
    from inatrc_kd.data import SafeYOLODataset
    arguments = dict(img_path=str(dataset / "train" / "images"), imgsz=64, batch_size=2,
                     data={"path": str(dataset), "nc": 5, "names": dict(enumerate(CLASS_NAMES)),
                           "audit_file_sha256": report["file_sha256"]},
                     hyp=get_cfg(), augment=False, cache=False, audit_cache_dir=tmp_path / "cache")
    assert len(SafeYOLODataset(**arguments)) == 10
    assert len(SafeYOLODataset(**arguments)) == 10
    _first_label(dataset).write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    with pytest.raises(DatasetAuditError, match="bytes changed since audit"):
        SafeYOLODataset(**arguments)
    assert not list(dataset.rglob("*.cache"))


def test_stale_manifest_cannot_trigger_source_image_repair(dataset, tmp_path, monkeypatch):
    from unittest.mock import Mock
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from inatrc_kd.data import SafeYOLODataset
    report = audit_dataset(dataset, expected_images=20)
    image = sorted((dataset / "train" / "images").iterdir())[0]
    image.write_bytes(b"corrupt bytes")
    upstream = Mock(side_effect=AssertionError("Do not scan/repair altered source"))
    monkeypatch.setattr(YOLODataset, "_load_or_scan_cache", upstream)
    with pytest.raises(DatasetAuditError, match="bytes changed since audit"):
        SafeYOLODataset(img_path=str(dataset / "train" / "images"), imgsz=64, batch_size=2,
                        data={"path": str(dataset), "nc": 5, "names": dict(enumerate(CLASS_NAMES)),
                              "audit_file_sha256": report["file_sha256"]},
                        hyp=get_cfg(), augment=False, cache=False, audit_cache_dir=tmp_path / "cache")
    upstream.assert_not_called()
    assert image.read_bytes() == b"corrupt bytes"


def test_runtime_manifest_requires_correct_dataset_root(dataset, tmp_path):
    report = audit_dataset(dataset, expected_images=20)
    report["source_path"] = str(tmp_path / "different-source")
    with pytest.raises(DatasetAuditError, match="different dataset root"):
        write_runtime_yaml(dataset, tmp_path / "outputs" / "data.yaml", audit=report)
