"""Audit gambar/label tanpa mengubah sumber; loader menyimpan cache di output.

Data sintetis hanya untuk smoke. Ultralytics dimuat saat loader diperlukan.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

CLASS_NAMES = (
    "Kelas-1 (Non-truk)",
    "Kelas-2 (Truk 2 sumbu)",
    "Kelas-3 (Truk 3 sumbu)",
    "Kelas-4 (Truk 4 sumbu)",
    "Kelas-5 (Truk >=5 sumbu)",
)
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
LABEL_METADATA = frozenset({"classes.txt", "_classes.txt"})
BBOX_TOLERANCE = 1e-4

if TYPE_CHECKING:
    from ultralytics.data.dataset import YOLODataset as SafeYOLODataset


class DatasetAuditError(ValueError):
    """The dataset cannot be consumed without changing or excluding data."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outside_source(path: Path, source: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == source or source in resolved.parents:
        raise DatasetAuditError(f"Output must be outside the source dataset: {resolved}")
    return resolved


def _files(directory: Path, *, labels: bool) -> list[Path]:
    if not directory.is_dir():
        raise DatasetAuditError(f"Missing required dataset directory: {directory}")
    result = []
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            raise DatasetAuditError(f"Nested dataset directories are unsupported: {path}")
        if not path.is_file():
            raise DatasetAuditError(f"Unsupported dataset entry: {path}")
        if labels and path.name.lower() in LABEL_METADATA:
            continue
        supported = path.suffix.lower() == ".txt" if labels else path.suffix.lower() in IMAGE_EXTENSIONS
        if not supported:
            raise DatasetAuditError(f"Unsupported {'label' if labels else 'image'} file: {path}")
        result.append(path)
    return result


def _by_stem(paths: Sequence[Path], kind: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        if path.stem in result:
            raise DatasetAuditError(f"Duplicate {kind} stem: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def _verify_image(path: Path) -> None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.width < 10 or image.height < 10:
                raise DatasetAuditError(f"Image dimensions must be at least 10 pixels: {path}")
            image_format = image.format
            image.verify()
        # verify() memeriksa struktur file; load() memastikan piksel bisa dibaca.
        with Image.open(path) as image:
            image.load()
        if image_format == "JPEG":
            with path.open("rb") as stream:
                stream.seek(-2, 2)
                if stream.read() != b"\xff\xd9":
                    raise DatasetAuditError(
                        f"JPEG EOF marker is missing; automatic JPEG repair is forbidden: {path}"
                    )
    except DatasetAuditError:
        raise
    except Exception as exc:
        raise DatasetAuditError(f"Corrupt or unreadable image {path}: {exc}") from exc


def _label_rows(path: Path, *, class_count: int = 5) -> tuple[bytes, list[tuple[float, ...]]]:
    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8-sig").strip().splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetAuditError(f"Unreadable UTF-8 label {path}: {exc}") from exc
    rows = []
    seen: set[tuple[float, ...]] = set()
    for line_number, line in enumerate(lines, 1):
        location = f"{path}:{line_number}"
        parts = line.split()
        if len(parts) != 5:
            format_note = " Polygon/segmentation conversion requires explicit approval." if len(parts) > 5 else ""
            raise DatasetAuditError(
                f"{location}: expected YOLO bbox with 5 columns, found {len(parts)}.{format_note}"
            )
        try:
            row = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise DatasetAuditError(f"{location}: nonnumeric label value") from exc
        if not all(math.isfinite(value) for value in row):
            raise DatasetAuditError(f"{location}: non-finite label value")
        cls, x, y, width, height = row
        # Format YOLO: class_id, pusat x/y, lebar/tinggi; koordinat relatif 0..1.
        if not cls.is_integer() or not 0 <= cls < class_count:
            raise DatasetAuditError(f"{location}: invalid class ID {cls}; expected 0..{class_count - 1}")
        if not (
            0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1
            and x - width / 2 >= -BBOX_TOLERANCE
            and y - height / 2 >= -BBOX_TOLERANCE
            and x + width / 2 <= 1 + BBOX_TOLERANCE
            and y + height / 2 <= 1 + BBOX_TOLERANCE
        ):
            raise DatasetAuditError(f"{location}: invalid normalized bounding box {row[1:]}")
        if row in seen:
            raise DatasetAuditError(f"{location}: duplicate label row; automatic removal is forbidden")
        seen.add(row)
        rows.append(row)
    return raw, rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(
    root: str | Path,
    output_dir: str | Path | None = None,
    expected_images: int | None = 3250,
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Validate all original splits without editing, repairing, or dropping data.

    The fingerprint reproduces the baseline notebook's byte serialization,
    with streaming image hashes. ``expected_images=None`` disables only the
    version-specific image-count assertion. Exact duplicates are reported as
    warnings; this audit cannot establish temporal/perceptual independence.
    """
    root = Path(root).expanduser().resolve()
    if len(class_names) != 5 or any(not isinstance(name, str) or not name for name in class_names):
        raise DatasetAuditError("InaTRC requires exactly five nonempty class names")
    if expected_images is not None and (
        isinstance(expected_images, bool) or not isinstance(expected_images, int) or expected_images <= 0
    ):
        raise DatasetAuditError("expected_images must be a positive integer or None")
    destination = _outside_source(Path(output_dir), root) if output_dir is not None else None
    split_rows, class_rows, box_rows = [], [], []
    occurrences: dict[str, list[dict[str, str]]] = defaultdict(list)
    fingerprint = hashlib.sha256()

    for split in SPLITS:
        images = _files(root / split / "images", labels=False)
        labels = _files(root / split / "labels", labels=True)
        image_map = _by_stem(images, "image")
        label_map = _by_stem(labels, "label")
        missing = sorted(image_map.keys() - label_map.keys())
        orphan = sorted(label_map.keys() - image_map.keys())
        if missing or orphan:
            raise DatasetAuditError(f"{split}: missing labels={missing}; orphan labels={orphan}")
        if not images:
            raise DatasetAuditError(f"Dataset split contains no images: {root / split}")
        counts = [0] * 5
        empty = 0
        for stem, image in sorted(image_map.items()):
            label = label_map[stem]
            _verify_image(image)
            image_hash = _sha256_file(image)
            label_bytes, rows = _label_rows(label)
            # Fingerprint berubah jika nama, isi gambar, atau anotasi berubah.
            fingerprint.update(f"{split}/{image.name}\n{image_hash}\n{label.name}\n".encode())
            fingerprint.update(label_bytes)
            occurrences[image_hash].append({"split": split, "image": image.name})
            empty += int(not rows)
            for cls, _, _, width, height in rows:
                class_id = int(cls)
                counts[class_id] += 1
                box_rows.append({
                    "split": split, "class_id": class_id, "class_name": class_names[class_id],
                    "normalized_area": width * height,
                    "normalized_width": width, "normalized_height": height,
                })
        split_rows.append({"split": split, "images": len(images), "labels": len(labels),
                           "boxes": sum(counts), "empty_labels": empty})
        for class_id, count in enumerate(counts):
            class_rows.append({"split": split, "class_id": class_id,
                               "class_name": class_names[class_id], "boxes": count})

    total_images = sum(row["images"] for row in split_rows)
    if expected_images is not None and total_images != expected_images:
        raise DatasetAuditError(f"Expected {expected_images} images, found {total_images} at {root}")
    duplicates = [{"sha256": digest, "occurrences": entries}
                  for digest, entries in sorted(occurrences.items())
                  if len({entry["split"] for entry in entries}) > 1]
    warning_messages = []
    if duplicates:
        warning_messages.append(
            f"Found {len(duplicates)} exact cross-split duplicate groups; investigate before the final study."
        )
    report = {
        "source_path": str(root), "class_names": list(class_names), "expected_images": expected_images,
        "total_images": total_images, "total_boxes": len(box_rows),
        "sha256": fingerprint.hexdigest(), "fingerprint": fingerprint.hexdigest(),
        "splits": split_rows, "class_distribution": class_rows, "bbox_distribution": box_rows,
        "cross_split_exact_duplicates": duplicates, "exact_cross_split_duplicate_groups": len(duplicates),
        "near_duplicate_or_sequence_leakage_checked": False,
        "note": "SHA-256 image bytes+label bytes+relative split/name; exact duplicate check only",
        "warnings": warning_messages,
    }
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        _write_csv(destination / "dataset_audit.csv", split_rows,
                   ("split", "images", "labels", "boxes", "empty_labels"))
        _write_csv(destination / "class_distribution.csv", class_rows,
                   ("split", "class_id", "class_name", "boxes"))
        _write_csv(destination / "bbox_distribution.csv", box_rows,
                   ("split", "class_id", "class_name", "normalized_area", "normalized_width", "normalized_height"))
        duplicate_csv = [{**row, "occurrences": json.dumps(row["occurrences"], ensure_ascii=False)}
                         for row in duplicates]
        _write_csv(destination / "cross_split_exact_duplicates.csv", duplicate_csv, ("sha256", "occurrences"))
        (destination / "dataset_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        (destination / "dataset_fingerprint.json").write_text(
            json.dumps({key: report[key] for key in (
                "sha256", "source_path", "note", "near_duplicate_or_sequence_leakage_checked",
                "exact_cross_split_duplicate_groups",
            )}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def write_runtime_yaml(root: str | Path, out_path: str | Path) -> Path:
    """Write a five-class data YAML outside the original dataset."""
    import yaml

    root = Path(root).expanduser().resolve()
    destination = _outside_source(Path(out_path), root)
    for split in SPLITS:
        if not (root / split / "images").is_dir() or not (root / split / "labels").is_dir():
            raise DatasetAuditError(f"Missing dataset split layout at {root / split}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {"path": str(root), **{split: str(root / split / "images") for split in SPLITS},
            "nc": 5, "names": list(CLASS_NAMES)}
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination


def create_synthetic_dataset(root: str | Path, seed: int = 42) -> Path:
    """Create 20 unique 64px PNGs, five classes, train10/val5/test5.

    The destination must be new or empty to prevent overwriting real data.
    This fixture is deliberately small and has no research metric meaning.
    """
    from PIL import Image, ImageDraw

    root = Path(root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise DatasetAuditError(f"Synthetic fixture destination must be new or empty: {root}")
    rng = random.Random(seed)
    index = 0
    for split, count in (("train", 10), ("val", 5), ("test", 5)):
        image_dir, label_dir = root / split / "images", root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for offset in range(count):
            image = Image.new("RGB", (64, 64), tuple(rng.randrange(24, 80) for _ in range(3)))
            draw = ImageDraw.Draw(image)
            draw.rectangle((16, 16, 47, 47), fill=tuple(rng.randrange(128, 256) for _ in range(3)))
            # Piksel penanda memastikan setiap gambar sintetis memiliki isi unik.
            image.putpixel((0, 0), (index, seed % 256, 255 - index))
            name = f"synthetic_{index:03d}"
            image.save(image_dir / f"{name}.png")
            (label_dir / f"{name}.txt").write_text(f"{offset % 5} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            index += 1
    return root


def __getattr__(name: str) -> Any:
    """Load the actual Ultralytics subclass only for training/evaluation."""
    if name != "SafeYOLODataset":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import numpy as np
    from ultralytics.data.dataset import YOLODataset

    if not hasattr(YOLODataset, "_load_or_scan_cache"):
        raise RuntimeError("Unsupported Ultralytics: YOLODataset._load_or_scan_cache is required")

    class SafeYOLODataset(YOLODataset):
        """Keep label caches in run output and reject any silent scan changes."""

        def __init__(self, *args: Any, audit_cache_dir: str | Path, **kwargs: Any) -> None:
            data = kwargs.get("data", {})
            if not data.get("path"):
                raise DatasetAuditError("SafeYOLODataset requires an explicit data['path']")
            self.audit_cache_dir = _outside_source(Path(audit_cache_dir), Path(data["path"]).resolve())
            if kwargs.get("cache") == "disk":
                raise DatasetAuditError("Disk image caching would write into the source dataset; use cache=False")
            if kwargs.get("task", "detect") != "detect":
                raise DatasetAuditError("Stage A SafeYOLODataset supports bbox detection only")
            self.audit_cache_dir.mkdir(parents=True, exist_ok=True)
            super().__init__(*args, **kwargs)

        def _load_or_scan_cache(self, cache_path: Path, cache_hash: str) -> tuple[dict, bool]:
            expected: dict[str, list[tuple[float, ...]]] = {}
            content_digest = hashlib.sha256(str(Path(cache_path).resolve()).encode())
            for image_name, label_name in zip(self.im_files, self.label_files, strict=True):
                image_path, label_path = Path(image_name), Path(label_name)
                _verify_image(image_path)
                raw, rows = _label_rows(label_path)
                expected[image_name] = rows
                content_digest.update(image_name.encode())
                content_digest.update(_sha256_file(image_path).encode())
                content_digest.update(raw)
            # Nama cache berdasarkan isi file; cache lama tidak lolos saat isi berubah.
            redirected = self.audit_cache_dir / f"{content_digest.hexdigest()}.cache"
            cache, exists = super()._load_or_scan_cache(redirected, cache_hash)
            found, missing, empty, corrupt, total = cache["results"]
            actual_labels = cache["labels"]
            actual_images = [label["im_file"] for label in actual_labels]
            expected_boxes = sum(map(len, expected.values()))
            actual_boxes = sum(len(label["cls"]) for label in actual_labels)
            if (missing or corrupt or total != len(expected) or found != len(expected)
                    or len(actual_images) != len(expected) or set(actual_images) != set(expected)
                    or expected_boxes != actual_boxes):
                raise DatasetAuditError(
                    "Ultralytics scan changed audited images/boxes: "
                    f"expected images={len(expected)}, boxes={expected_boxes}; "
                    f"loaded images={len(actual_images)}, boxes={actual_boxes}, "
                    f"missing={missing}, corrupt={corrupt}. Details: {cache.get('msgs', [])}"
                )
            if empty != sum(not rows for rows in expected.values()):
                raise DatasetAuditError("Ultralytics scan changed the audited empty-label count")
            for label in actual_labels:
                wanted = np.asarray(expected[label["im_file"]], dtype=np.float32).reshape(-1, 5)
                actual = np.concatenate((label["cls"], label["bboxes"]), axis=1)
                if not np.array_equal(actual, wanted) or len(label.get("segments", [])):
                    raise DatasetAuditError(f"Ultralytics scan changed label values: {label['im_file']}")
            return cache, exists

    SafeYOLODataset.__qualname__ = "SafeYOLODataset"
    globals()[name] = SafeYOLODataset
    return SafeYOLODataset
