"""Helper yang ditampilkan langsung di notebook; runtime tidak mengimpor file ini."""

import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile


def _setup_redact(text):
    """Jangan tulis credential ke layar atau log instalasi."""
    value = str(text)
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        value = value.replace(key, "[REDACTED]")
    return value


def _setup_run_command(command, log_path):
    """Tampilkan progres pip sambil menyimpan log yang dapat diunduh."""
    chunks = []
    with Path(log_path).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in process.stdout:
            safe_line = _setup_redact(line)
            print(safe_line, end="", flush=True)
            log.write(safe_line)
            log.flush()
            chunks.append(safe_line)
        returncode = process.wait()
    if returncode:
        tail = "".join(chunks)[-2000:]
        raise RuntimeError(f"pip gagal (exit {returncode}). {tail}\nLog: {log_path}")
    return "".join(chunks)


def _setup_package_snapshot(package):
    distribution = importlib.metadata.distribution(package)
    return {
        "version": distribution.version,
        "location": str(Path(distribution.locate_file("")).resolve()),
    }


def validate_loaded_packages(packages):
    """Package yang diganti pip harus dimuat ulang melalui restart kernel."""
    module_names = {
        "pillow": "PIL", "pyyaml": "yaml", "faster-coco-eval": "faster_coco_eval",
        "opencv-python": "cv2", "opencv-python-headless": "cv2",
        "ultralytics-thop": "thop",
    }
    for package in packages:
        normalized = re.sub(r"[-_.]+", "-", package).lower()
        module_name = module_names.get(normalized, normalized.replace("-", "_"))
        module = sys.modules.get(module_name)
        if module is None:
            continue
        installed = _setup_package_snapshot(package)
        loaded_version = getattr(module, "__version__", None)
        module_file = getattr(module, "__file__", None)
        # cv2 tidak menampilkan nomor build wheel, tetapi versi core harus sama.
        expected_version, active_version = installed["version"], str(loaded_version)
        if normalized in {"opencv-python", "opencv-python-headless"}:
            expected_core = re.match(r"^\d+\.\d+\.\d+", expected_version)
            active_core = re.match(r"^\d+\.\d+\.\d+", active_version)
            expected_version = expected_core.group() if expected_core else expected_version
            active_version = active_core.group() if active_core else active_version
        stale_version = loaded_version is not None and active_version != expected_version
        stale_location = module_file is not None and not Path(module_file).resolve().is_relative_to(
            Path(installed["location"]).resolve()
        )
        if stale_version or stale_location:
            raise RuntimeError(
                f"{package} yang sudah dimuat berbeda dari instalasi aktif. "
                "Restart Session (Kaggle) / Restart runtime (Colab), lalu Run All kembali."
            )


def _setup_pinned_requirements(requirements):
    pins = {}
    for requirement in requirements:
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)", requirement)
        if match is None:
            raise ValueError(f"Dependency notebook harus berupa nama==versi: {requirement!r}")
        name, version = match.groups()
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in {"torch", "torchvision"}:
            raise ValueError("Torch/torchvision berasal dari runtime; jangan masukkan ke dependency notebook.")
        if normalized in pins and pins[normalized][1] != version:
            raise ValueError(f"Pin dependency ganda berbeda: {name}")
        pins[normalized] = (name, version)
    return pins


def _setup_validate_requirements(packages):
    """Periksa dependency aplikasi aktif; tool lain bawaan Kaggle tidak dilatih."""
    from packaging.requirements import Requirement

    checked = []
    for package in packages:
        for declaration in importlib.metadata.requires(package) or []:
            requirement = Requirement(declaration)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(f"Dependency {package} belum terpenuhi: {requirement.name}.") from None
            if not requirement.specifier.contains(installed, prereleases=True):
                raise RuntimeError(f"Dependency {package} tidak sesuai: {declaration}; aktif {installed}.")
            checked.append({"package": package, "requirement": declaration, "active_version": installed})
    return checked


def setup_dependencies(requirements, output_root):
    """Resolve dahulu, lindungi Torch CUDA, lalu install paket aplikasi saja."""
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Notebook mendukung Python kernel 3.11, 3.12, dan 3.13.")
    pins = _setup_pinned_requirements(requirements)
    logs = Path(output_root).expanduser().resolve() / "setup_logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "pip_setup.log"
    report_path = logs / "pip_plan.json"
    constraints = logs / "protected_torch.txt"
    started = time.perf_counter()
    print("Instalasi dependency: memeriksa rencana pip...", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n=== Instalasi dependency ===\n")
    try:
        before = {name: _setup_package_snapshot(name) for name in ("torch", "torchvision")}
        validate_loaded_packages(["torch", "torchvision", *[name for name, _ in pins.values()]])
        constraints.write_text(
            "".join(f"{name}=={info['version']}\n" for name, info in before.items()), encoding="utf-8"
        )
        # Report hanya memuat rencana; belum ada package yang diubah di sini.
        command = [
            sys.executable, "-m", "pip", "install", "--dry-run", "--report", str(report_path),
            "--constraint", str(constraints), "--only-binary=faster-coco-eval",
            "--disable-pip-version-check", *[f"{name}=={version}" for name, version in pins.values()],
        ]
        _setup_run_command(command, log_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # Simpan versi report yang sudah disanitasi, bukan environment/credential.
        report_path.write_text(_setup_redact(json.dumps(report, indent=2)), encoding="utf-8")
        install_pins = []
        for item in report.get("install", []):
            metadata = item["metadata"]
            name, version = metadata["name"], metadata["version"]
            normalized = re.sub(r"[-_.]+", "-", name).lower()
            if normalized in before:
                raise RuntimeError(
                    f"STOP: pip hendak mengganti {name}. Torch/torchvision bawaan tidak boleh diubah."
                )
            _setup_pinned_requirements([f"{name}=={version}"])
            install_pins.append(f"{name}=={version}")
        if install_pins:
            print("Instalasi dependency: memasang paket aplikasi yang telah di-resolve...", flush=True)
            _setup_run_command([
                sys.executable, "-m", "pip", "install", "--no-deps",
                "--only-binary=faster-coco-eval", "--disable-pip-version-check", *install_pins,
            ], log_path)
        else:
            print("Dependency sudah sesuai; instalasi dilewati.", flush=True)
        after = {name: _setup_package_snapshot(name) for name in before}
        if after != before:
            raise RuntimeError("STOP: versi/lokasi Torch berubah. Pulihkan runtime GPU sebelum melanjutkan.")
        installed = {name: _setup_package_snapshot(name) for name, _ in pins.values()}
        for name, version in pins.values():
            if installed[name]["version"] != version:
                raise RuntimeError(f"Versi {name} tidak sesuai pin {version}.")
        validate_loaded_packages(["torch", "torchvision", *installed])
        # --no-deps tetap harus menghasilkan dependency graph aplikasi yang konsisten.
        checked = _setup_validate_requirements(["torch", "torchvision", *installed])
        elapsed = round(time.perf_counter() - started, 3)
        print(f"Instalasi dependency selesai: {elapsed:.1f} detik. Log: {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"Selesai: {elapsed:.3f} detik\n")
        return {
            "python_executable": sys.executable, "torch_before": before, "torch_after": after,
            "packages": installed, "duration_seconds": elapsed,
            "dependency_checks": checked,
            "log_path": str(log_path), "report_path": str(report_path),
        }
    except Exception as exc:
        safe_error = _setup_redact(f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter() - started
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"GAGAL setelah {elapsed:.3f} detik: {safe_error}\n")
        raise RuntimeError(
            f"Tahap Instalasi Dependency gagal setelah {elapsed:.1f} detik. "
            f"{safe_error}\nLog: {log_path}"
        ) from None


def detect_platform():
    """Gunakan direktori platform; smoke lokal boleh memakai CPU."""
    if Path("/kaggle/working").is_dir():
        return {
            "name": "kaggle", "work_root": "/kaggle/working", "input_root": "/kaggle/input",
            "output_root": "/kaggle/working/outputs",
        }
    if Path("/content").is_dir() or "google.colab" in sys.modules:
        return {
            "name": "colab", "work_root": "/content", "input_root": "/content/inputs",
            "output_root": "/content/outputs",
        }
    return {
        "name": "local", "work_root": str(Path.cwd()), "input_root": None,
        "output_root": str(Path.cwd() / "outputs"),
    }


def get_wandb_secret(mode, platform):
    """Ambil key hanya untuk online; nilai key tidak pernah dikembalikan/dicetak."""
    if mode not in {"disabled", "offline", "online"}:
        raise ValueError("WANDB_MODE harus disabled, offline, atau online.")
    if mode != "online":
        os.environ.pop("WANDB_API_KEY", None)
        return None
    if os.environ.get("WANDB_API_KEY"):
        return None
    name = platform["name"] if isinstance(platform, dict) else platform
    try:
        if name == "kaggle":
            from kaggle_secrets import UserSecretsClient
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        elif name == "colab":
            from google.colab import userdata
            key = userdata.get("WANDB_API_KEY")
        else:
            raise ValueError("WANDB_API_KEY belum diatur pada environment lokal.")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Secret kosong.")
        os.environ["WANDB_API_KEY"] = key
    except Exception:
        raise RuntimeError(
            "W&B online memerlukan WANDB_API_KEY dari environment atau Secrets platform. "
            "Aktifkan akses secret, atau pilih WANDB_MODE='disabled'/'offline'."
        ) from None
    return None


def _standalone_is_dataset_root(path):
    path = Path(path)
    return all((path / split / kind).is_dir() for split in ("train", "val", "test") for kind in ("images", "labels"))


def discover_dataset_root(root, input_root, mode):
    """Pilih hanya satu dataset dengan ketiga split; jangan menebak jika ambigu."""
    if mode not in {"smoke", "full"}:
        raise ValueError("MODE harus smoke atau full.")
    if root:
        selected = Path(root).expanduser().resolve()
        if not _standalone_is_dataset_root(selected):
            raise RuntimeError(f"DATASET_ROOT harus berisi train/val/test dengan images dan labels: {selected}")
        return str(selected)
    candidates = set()
    if input_root and Path(input_root).is_dir():
        inputs = Path(input_root).expanduser().resolve()
        for directory, subdirs, _ in os.walk(inputs):
            # Lewati gambar individual; penemuan hanya membutuhkan direktori split.
            subdirs[:] = sorted(name for name in subdirs if name not in {"images", "labels", ".git", ".cache"})
            candidate = Path(directory)
            if _standalone_is_dataset_root(candidate):
                candidates.add(candidate.resolve())
                subdirs[:] = []
    if len(candidates) > 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise RuntimeError(f"Lebih dari satu dataset ditemukan. Isi DATASET_ROOT secara eksplisit: {paths}")
    if candidates:
        return str(next(iter(candidates)))
    if mode == "full":
        raise RuntimeError("Dataset InaTRC belum ditemukan. Tambahkan Kaggle Input atau isi DATASET_ROOT.")
    return None


def create_results_zip(run_root, output_root=None):
    """Bundle hanya hasil run ini; data sintetis/secret/teacher tidak ikut."""
    run = Path(run_root).expanduser().resolve()
    if not run.is_dir():
        raise RuntimeError(f"Direktori hasil tidak tersedia: {run}")
    target = Path(output_root).expanduser().resolve() if output_root else run.parent
    target.mkdir(parents=True, exist_ok=True)
    archive_path = target / f"{run.name}-results-{uuid.uuid4().hex[:8]}.zip"
    approved_names = {
        "best.pt", "last.pt", "results.csv", "epoch_history.csv", "checkpoint_manifest.csv",
        "metrics.json", "metrics.csv", "per_class.csv", "coco_metrics.json", "coco_metrics.csv",
        "resolved_config.json", "resolved_config.yaml", "dataset_audit.json", "dataset_fingerprint.json",
        "environment.json", "metadata.json", "checkpoint_audit.json", "summary.json", "summary.csv",
        "source_provenance.json", "training_status.json", "initial_weights.json",
        "args.yaml", "complexity.json", "run_summary.json",
        "training_summary.json", "coco_area.json", "initial_provenance.json", "teacher_provenance.json",
        "data_runtime.yaml", "predictions.json",
        "dataset_audit.csv", "class_distribution.csv", "bbox_distribution.csv", "cross_split_exact_duplicates.csv",
    }
    skipped_parts = {"wandb", "synthetic_data", "synthetic", "dataset", "teacher", "diagnostic_teacher", "cache", ".cache"}
    selected = []
    for path in sorted(run.rglob("*")):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(run):
            continue
        relative = path.relative_to(run)
        if any(part in skipped_parts for part in relative.parts):
            continue
        # Kurva dan matriks validasi ikut sebagai hasil akhir.
        is_plot = path.suffix.lower() in {".png", ".jpg", ".svg", ".pdf"} and (
            any(part in {"val", "validation", "best-val", "evaluation"} for part in relative.parts)
            or path.stem.endswith("curve")
            or path.stem.startswith(("confusion_matrix", "results"))
        )
        if path.name not in approved_names and not is_plot:
            continue
        if path.suffix.lower() in {".json", ".yaml", ".csv"}:
            content = path.read_text(encoding="utf-8")
            if "WANDB_API_KEY" in content or os.environ.get("WANDB_API_KEY", "\0") in content:
                raise RuntimeError(f"Bundle dibatalkan: credential ditemukan dalam {relative}.")
        selected.append((path, relative))
    if not selected:
        raise RuntimeError("Tidak ada artifact hasil yang dikenali untuk dibundel.")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in selected:
            archive.write(path, arcname=str(Path(run.name) / relative))
    print(f"Bundle hasil: {archive_path}", flush=True)
    return archive_path
