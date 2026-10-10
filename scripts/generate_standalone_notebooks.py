"""Tanam kode yang diuji menjadi notebook biasa; generator hanya digunakan lokal."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from pprint import pformat
import textwrap
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "notebooks" / "standalone"
PRESETS = (
    {"name": "01_teacher_yolo26m", "title": "Teacher YOLO26m", "stage": "teacher", "student": "n", "method": "none", "mode": "full"},
    {"name": "02_baseline_yolo26s", "title": "Baseline YOLO26s", "stage": "student", "student": "s", "method": "none", "mode": "full"},
    {"name": "03_baseline_yolo26n", "title": "Baseline YOLO26n", "stage": "student", "student": "n", "method": "none", "mode": "full"},
    {"name": "04_native_yolo26s", "title": "Native KD YOLO26s", "stage": "student", "student": "s", "method": "native", "mode": "full"},
    {"name": "05_native_yolo26n", "title": "Native KD YOLO26n", "stage": "student", "student": "n", "method": "native", "mode": "full"},
    {"name": "06_crosskd_yolo26s", "title": "CrossKD YOLO26s", "stage": "student", "student": "s", "method": "crosskd", "mode": "smoke"},
    {"name": "07_crosskd_yolo26n", "title": "CrossKD YOLO26n", "stage": "student", "student": "n", "method": "crosskd", "mode": "smoke"},
    {"name": "08_csakd_yolo26s", "title": "CSAKD YOLO26s", "stage": "student", "student": "s", "method": "csakd", "mode": "smoke"},
    {"name": "09_csakd_yolo26n", "title": "CSAKD YOLO26n", "stage": "student", "student": "n", "method": "csakd", "mode": "smoke"},
)


def _own_import(node):
    return isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("inatrc_kd"))


def _module(relative, *, names=None, exclude=()):
    """Salin definisi beserta komentar, sambil membuang import internal repo."""
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    tree, lines = ast.parse(source), source.splitlines(keepends=True)
    ignored = set()
    for node in ast.walk(tree):
        if _own_import(node) or isinstance(node, ast.ImportFrom) and node.module == "__future__":
            ignored.update(range(node.lineno, node.end_lineno + 1))
    blocks = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in exclude or isinstance(node, ast.If):
            continue
        if names is not None and not isinstance(node, (ast.Import, ast.ImportFrom)) and name not in names:
            continue
        if _own_import(node) or isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        start = min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])])
        block = "".join(line for number, line in enumerate(lines[start - 1:node.end_lineno], start) if number not in ignored)
        blocks.append(block.rstrip())
    return "\n\n".join(blocks) + "\n"


def _loader():
    source = (REPO_ROOT / "src/inatrc_kd/data.py").read_text(encoding="utf-8")
    node = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef) and node.name == "SafeYOLODataset")
    return "from ultralytics.data.dataset import YOLODataset\n\n" + textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])) + "\n"


def _literal(name, value):
    return name + " = " + pformat(value, width=100, sort_dicts=False) + "\n"


def _requirements():
    """Ambil pin aplikasi dari lock; Torch CUDA diwarisi dari kernel."""
    packages = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    names = ("ultralytics", "wandb", "faster-coco-eval", "pandas", "pillow", "pyyaml", "matplotlib",
             "ultralytics-thop", "packaging", "opencv-python")
    pins = []
    for name in names:
        versions = {row["version"] for row in packages if row["name"] == name}
        if len(versions) != 1:
            raise ValueError(f"Expected one lock version for {name}: {versions}")
        pins.append(f"{name}=={versions.pop()}")
    numpy = sorted({row["version"] for row in packages if row["name"] == "numpy"})
    if len(numpy) != 2:
        raise ValueError("Review NumPy pins for Python 3.11 versus 3.12/3.13")
    return pins, numpy


def _configuration(preset):
    sections = {name: yaml.safe_load((REPO_ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
                for name in ("training", "dataset", "kd", "tracking")}
    training = sections.pop("training")
    evaluation = training["evaluation"]
    evaluation.update(conf=0.001, save_json=False)
    kd = sections["kd"]
    kd = {key: value for key, value in kd.items() if key not in {"crosskd", "csakd"} or key == preset["method"]}
    code = "from copy import deepcopy\n\n"
    for name, value in (("FULL_PROTOCOL", training["full"]), ("SMOKE_PROTOCOL", training["smoke"]),
                        ("EVALUATION_PROTOCOL", evaluation), ("DATASET_PROTOCOL", sections["dataset"]),
                        ("KD_PROTOCOL", kd), ("TRACKING_PROTOCOL", sections["tracking"])):
        code += _literal(name, value) + "\n"
    code += _module("src/inatrc_kd/config.py", names={"validate_output_location", "validate_config"})
    code += textwrap.dedent('''

    def make_config():
        """Bangun parameter efektif dari pilihan di cell pertama."""
        training = deepcopy(FULL_PROTOCOL)
        evaluation = deepcopy(EVALUATION_PROTOCOL)
        evaluation.update(coco_area=bool(EVALUATE_COCO_AREA), save_json=bool(EVALUATE_COCO_AREA))
        if MODE == "smoke":
            training.update(SMOKE_PROTOCOL)
            evaluation["pilot_latency"] = False
        training["seed"] = SEED
        dataset = deepcopy(DATASET_PROTOCOL)
        dataset["root"] = discover_dataset_root(DATASET_ROOT, INPUT_ROOT, MODE)
        dataset["expected_images"] = EXPECTED_IMAGES
        kd = deepcopy(KD_PROTOCOL)
        kd["teacher_checkpoint"] = str(Path(TEACHER_CKPT).expanduser().resolve()) if TEACHER_CKPT else None
        tracking = deepcopy(TRACKING_PROTOCOL)
        tracking.update(mode=WANDB_MODE, project=WANDB_PROJECT, entity=WANDB_ENTITY)
        scale = "m" if STAGE == "teacher" else STUDENT
        model = f"yolo26{scale}.{'yaml' if MODE == 'smoke' else 'pt'}"
        if MODEL_WEIGHTS:
            checkpoint = Path(MODEL_WEIGHTS).expanduser().resolve()
            if MODE != "full" or not checkpoint.is_file() or checkpoint.suffix != ".pt":
                raise ValueError("MODEL_WEIGHTS harus checkpoint pretrained COCO lokal untuk mode full.")
            model = str(checkpoint)
        config = {"stage": STAGE, "student": STUDENT, "method": METHOD, "mode": MODE,
                  "seed": SEED, "device": str(DEVICE), "model": model, "output_root": str(OUTPUT_ROOT),
                  "training": training, "evaluation": evaluation, "dataset": dataset, "kd": kd,
                  "tracking": tracking, "create_results_zip": bool(CREATE_RESULTS_ZIP)}
        validate_config(config)
        if STAGE == "teacher" and SEED != 42:
            raise ValueError("Teacher penelitian tetap seed 42; seed student boleh diubah.")
        if METHOD != "none" and MODE == "full" and not TEACHER_CKPT:
            raise ValueError("KD full memerlukan TEACHER_CKPT: best.pt teacher YOLO26m InaTRC lima kelas.")
        return config
    ''')
    return code


def _kd(preset):
    method = preset["method"]
    if method == "none":
        return 'def configure_kd(config):\n    """Teacher dan baseline hanya memakai loss deteksi."""\n    return {}\n'
    code = _module("src/inatrc_kd/kd/native.py")
    if method in {"crosskd", "csakd"}:
        code += "\n" + _module("src/inatrc_kd/kd/common.py")
        code += "\n" + _module(f"src/inatrc_kd/kd/{method}.py")
        selected = "CrossKDModel" if method == "crosskd" else "CSAKDModel"
        code += textwrap.dedent(f'''

        class CustomKDTrainer(SafeDetectionTrainer):
            """Daftarkan adapter sebelum optimizer/EMA; bersihkan hook saat selesai."""
            def set_model_attributes(self):
                super().set_model_attributes()
                if self.args.compile or self.args.resume:
                    raise ValueError("Custom KD pilot does not support compile or resume")
                config = yaml.safe_load((Path(self.save_dir) / "resolved_config.yaml").read_text(encoding="utf-8"))
                if config["method"] != {method!r}:
                    raise ValueError("Metode tidak sesuai dengan preset notebook.")
                self.model = {selected}(self.args.distill_model, self.model, config["kd"][{method!r}])

            def train(self):
                try:
                    return super().train()
                finally:
                    model = unwrap_model(self.model)
                    if hasattr(model, "_remove_feature_hooks"):
                        model._remove_feature_hooks()
                        model._student_feats.clear()
                        model._teacher_feats.clear()
        ''')
    function = {"native": "native_train_kwargs", "crosskd": "configure_crosskd", "csakd": "configure_csakd"}[method]
    code += f'\ndef configure_kd(config):\n    if config["stage"] != "student" or config["method"] != {method!r}:\n        raise ValueError("Preset KD notebook tidak sesuai.")\n    return {function}(config)\n'
    return code


def _training(preset):
    code = _module("src/inatrc_kd/train.py")
    code = code.replace("source_provenance(REPO_ROOT)", "source_provenance()")
    start = code.index('    provenance_file = os.getenv("INATRC_SOURCE_PROVENANCE")')
    end = code.index("    try:\n", start)
    code = code[:start] + code[end:]
    # Environment sudah dicek oleh cell GPU; probe acak tetap tersedia di CLI lokal.
    code = code.replace('environment = environment_report(config["device"], require_gpu=config["device"] != "cpu")',
                        'environment = ENVIRONMENT\n        if environment["requested_device"] != config["device"]:\n            raise ValueError("Device berubah setelah preflight; jalankan ulang cell Pemeriksaan GPU.")')
    start = code.index('        metadata["architecture_probe"] = inspect_model(')
    end = code.index('        model = YOLO(config["model"])', start)
    code = code[:start] + code[end:]
    if preset["method"] == "none":
        start = code.index('        # Teacher acak hanya alat diagnosis smoke')
        end = code.index("        kd_args = configure_kd(config)", start)
        code = code[:start] + code[end:]
    if preset["method"] not in {"crosskd", "csakd"}:
        start = code.index('        if config["method"] in ("crosskd", "csakd"):')
        end = code.index("        model.train(trainer=trainer", start)
        code = code[:start] + code[end:]
    else:
        code = code.replace('"guide": "docs/stage_b_design.md"', '"guide": "Algoritme KD di notebook ini; konfigurasi pilot"')
    code = code.replace('        if kd_args:\n            metadata["teacher_checkpoint_sha256"] = sha256(kd_args["distill_model"])',
                        '        if kd_args:\n            metadata["teacher_checkpoint_sha256"] = sha256(kd_args["distill_model"])\n'
                        '            if TEACHER_SHA256 and metadata["teacher_checkpoint_sha256"] != TEACHER_SHA256:\n'
                        '                raise ValueError("Hash teacher berbeda dari TEACHER_SHA256 yang ditentukan.")')
    return code


def _evaluation():
    code = _module("src/inatrc_kd/coco.py")
    evaluation = _module("src/inatrc_kd/evaluate.py")
    evaluation = evaluation.replace("validator=SafeDetectionValidator", 'validator=AreaAuditValidator if evaluation["coco_area"] else SafeDetectionValidator').replace("save_json=False", 'save_json=evaluation["coco_area"]')
    evaluation = evaluation.replace('    metrics["per_class"] = per_class',
        '    metrics["per_class"] = per_class\n'
        '    if evaluation["coco_area"]:\n'
        '        if getattr(result, "coco_area", None) is None:\n'
        '            raise RuntimeError("COCO area evaluator did not return final metrics")\n'
        '        metrics["coco_area"] = result.coco_area\n'
        '        write_json(run_dir / "coco_area.json", result.coco_area)\n'
        '        pd.DataFrame([{key: result.coco_area[key] for key in COCO_METRIC_KEYS}]).to_csv(run_dir / "coco_metrics.csv", index=False)')
    evaluation = evaluation.replace('    write_json(run_dir / "metrics.json", metrics)',
        '    pd.DataFrame([{key: metrics[key] for key in ("precision", "recall", "mAP50", "mAP50_95")}]).to_csv(run_dir / "metrics.csv", index=False)\n'
        '    write_json(run_dir / "metrics.json", metrics)')
    return code + "\n" + evaluation


def _cell(kind, cell_id, source):
    result = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": source.rstrip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def render_notebook(preset):
    """Notebook hasil hanya memerlukan file .ipynb dan dependency publik."""
    cells = []
    def md(cell_id, source):
        cells.append(_cell("markdown", cell_id, source))
    def code(cell_id, source):
        ast.parse(source)
        cells.append(_cell("code", cell_id, source))

    md("intro", f'''# {preset["title"]} — InaTRC

Notebook mandiri untuk Kaggle/Colab. Semua fungsi dan algoritme terlihat di bawah.
Run All menjalankan **satu eksperimen**. Default notebook ini: **{preset["mode"]}**, seed 42.
Teacher/baseline/Native full memerlukan dataset asli; semua KD full memerlukan teacher YOLO26m InaTRC yang sama.
Smoke memakai data sintetis dan hasilnya **bukan hasil penelitian**. CrossKD/CSAKD berstatus **pilot**.

Protokol bersama: 640 px, batch 16, AdamW, maksimal 100 epoch, patience 25. Batch tidak diturunkan otomatis.
Metrik utama: mAP50–95 Ultralytics pada **val**; split test hanya diaudit.
Internet diperlukan saat dependency atau pretrained COCO belum tersedia.
''')
    md("config-heading", "## Konfigurasi Eksperimen\nIsi path dataset/checkpoint. Pilih `smoke` untuk diagnosis 1 epoch sebelum training penelitian.")
    code("config", f'''# Stage/model/metode sesuai judul; gunakan notebook lain untuk eksperimen lain.
STAGE = {preset["stage"]!r}
STUDENT = {preset["student"]!r}  # teacher selalu memakai YOLO26m
METHOD = {preset["method"]!r}
MODE = {preset["mode"]!r}  # "smoke" atau "full"
SEED = 42
DEVICE = "0"  # GPU pertama; "cpu" hanya untuk diagnosis lokal

DATASET_ROOT = ""  # Kaggle: otomatis mencari satu dataset train/val/test
EXPECTED_IMAGES = 3250  # ubah eksplisit jika memakai revisi dataset berbeda
TEACHER_CKPT = ""  # KD full: /kaggle/input/.../best.pt atau path Colab
TEACHER_SHA256 = ""  # opsional: kunci hash teacher yang sama di semua KD full
MODEL_WEIGHTS = None  # opsional: pretrained COCO .pt lokal sesuai ukuran model
INPUT_ROOT = None  # otomatis: /kaggle/input atau /content/inputs
OUTPUT_ROOT = None  # otomatis: /kaggle/working/outputs atau /content/outputs

WANDB_MODE = "disabled"  # "disabled", "offline", atau "online"
WANDB_PROJECT = "skripsi-inatrc-yolo26"
WANDB_ENTITY = None  # username/team; WANDB_API_KEY di Secrets, jangan ditulis di sini

# Fitur tambahan: samakan pilihan COCO di semua eksperimen yang dibandingkan.
EVALUATE_COCO_AREA = False  # True: AP-S/AP-M/AP-L tambahan, lebih banyak ekspor/evaluasi
CREATE_RESULTS_ZIP = False  # True: bundel unduhan; Kaggle Save Version sudah menyimpan folder

# Opsional Colab Drive: uncomment, lalu isi DATASET_ROOT / TEACHER_CKPT di atas.
# from google.colab import drive
# drive.mount("/content/drive")
''')
    md("setup-heading", "## Instalasi Dependency\nPip menggunakan interpreter kernel. Rencana instalasi diperiksa terlebih dahulu; Torch/torchvision bawaan dipertahankan. Bila paket sudah di-import dengan versi berbeda, restart session/kernel lalu Run All.")
    code("runtime-definitions", _module("scripts/standalone_runtime.py"))
    pins, numpy = _requirements()
    code("setup", _literal("APP_REQUIREMENTS", pins) + f'''
APP_REQUIREMENTS.append("numpy==" + ({numpy[0]!r} if sys.version_info[:2] < (3, 12) else {numpy[1]!r}))
if not EVALUATE_COCO_AREA:
    APP_REQUIREMENTS = [pin for pin in APP_REQUIREMENTS if not pin.startswith("faster-coco-eval==")]
PLATFORM_INFO = detect_platform()
INPUT_ROOT = str(Path(INPUT_ROOT or PLATFORM_INFO["input_root"] or Path.cwd() / "inputs").expanduser().resolve())
OUTPUT_ROOT = str(Path(OUTPUT_ROOT or PLATFORM_INFO["output_root"]).expanduser().resolve())
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
print("Platform:", PLATFORM_INFO["name"], "| Output:", OUTPUT_ROOT, flush=True)
SETUP_REPORT = setup_dependencies(APP_REQUIREMENTS, OUTPUT_ROOT)
get_wandb_secret(WANDB_MODE, PLATFORM_INFO)
''')
    code("imports", '''import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO, settings
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds, unwrap_model

# Matplotlib menulis cache di output, terpisah dari dataset sumber.
os.environ.setdefault("MPLCONFIGDIR", str(Path(OUTPUT_ROOT) / "matplotlib-cache"))
''')
    md("environment-heading", "## Pemeriksaan GPU\nCek versi aktif, operasi CUDA, torchvision NMS, dan pertukaran NumPy–Torch. Gagal di sini menghentikan training.")
    code("environment-definitions", _module("src/inatrc_kd/preflight.py", names={"INATRC_NAMES", "_device", "_validate_runtime_requirements", "_validate_python_torch_pair", "environment_report"}).replace("uv environment", "pip/kernel environment")
         + "\n" + _module("src/inatrc_kd/io.py", names={"sha256", "write_json"}))
    code("environment", '''validate_loaded_packages([requirement.split("==", 1)[0] for requirement in APP_REQUIREMENTS])
checked = None if SETUP_REPORT is None else [row for row in SETUP_REPORT["dependency_checks"] if row["package"] in ("torch", "torchvision")]
ENVIRONMENT = environment_report(str(DEVICE), require_gpu=str(DEVICE) != "cpu", dependency_checks=checked)
write_json(Path(OUTPUT_ROOT) / "setup_logs" / "environment.json", ENVIRONMENT)
print("Pemeriksaan GPU/ABI lolos:", ENVIRONMENT["resolved_device"], ENVIRONMENT["torch_version"], flush=True)
''')
    md("data-heading", "## Penemuan dan Audit Dataset\nAudit membaca ketiga split, pasangan gambar/label, isi label, gambar rusak, dan duplikasi. Fingerprint memakai isi file. Kebocoran temporal/perseptual belum diperiksa. Cache loader hanya ditulis ke output.")
    code("data-definitions", _module("src/inatrc_kd/data.py", exclude={"__getattr__"}) + "\n" + _loader() + "\n" + _module("src/inatrc_kd/pipeline.py"))
    code("configuration-definitions", _configuration(preset))
    code("configuration", '''CONFIG = make_config()
print("Dataset:", CONFIG["dataset"]["root"] or "sintetis (smoke)", flush=True)
print("Protokol:", CONFIG["model"], MODE, "seed", SEED, "batch", CONFIG["training"]["batch"], flush=True)
# Audit lengkap sekali; loader dan pemeriksaan akhir mencocokkan seluruh byte dengan audit itu.
''')
    md("model-heading", "## Model dan Teacher\nFull dimulai dari pretrained COCO sesuai ukuran. Hash checkpoint dan state student awal dicatat. Teacher KD harus YOLO26m lima kelas dengan urutan kelas yang sama; teacher sintetis hanya diperbolehkan pada smoke.")
    names = {"_state_hash"}
    if preset["method"] != "none":
        names.update({"ordered_names", "check_teacher", "calibrate_diagnostic_bn", "make_synthetic_teacher"})
    code("model-definitions", _module("src/inatrc_kd/preflight.py", names=names))
    md("kd-heading", "## Algoritme KD\n" + {"none": "Loss deteksi Ultralytics untuk teacher/baseline.", "native": "API resmi Ultralytics: distill_model dan dis=6.0. Mekanisme KD tetap milik Ultralytics.", "crosskd": "Adaptasi pilot CrossKD: suffix head teacher one-to-many, adapter sebelum optimizer, teacher frozen. Sumber: [CrossKD penulis](https://github.com/jbwang1997/CrossKD).", "csakd": "Adaptasi pilot CSAKD: varian paper, pasangan backbone/neck dari graph. Sumber: [repository penulis](https://github.com/KefanZhan/YOLOv8-KD)."}[preset["method"]])
    code("kd-definitions", _kd(preset))
    md("logging-heading", "## Logging per Epoch\nSelalu tampil dan tersimpan lokal: loss, KD, metrik val, LR, fitness, durasi, is_best, dan best_epoch. Kejadian penyimpanan best.pt menjadi acuan; fitness seri mengikuti checkpoint terbaru. Final validation tidak menambah epoch.")
    code("tracking-definitions", _module("src/inatrc_kd/tracking.py"))
    md("evaluation-heading", "## Evaluasi best.pt\nMuat ulang YOLO(best.pt), val FP32: nms=None, conf=0.001, IoU=0.7, max_det=300, augmentasi off. Parameter/FLOPs dari checkpoint lima kelas sebelum fusion; latency, VRAM, dan durasi tetap dicatat. COCO area opsional (EVALUATE_COCO_AREA=True) memakai bbox pixel asli/maxDets=[1,10,100]; definisi metrik utama tetap sama.")
    code("evaluation-definitions", _evaluation())
    md("training-heading", "## Training\nSatu run memakai folder unik. OOM/nonfinite menghentikan run. best.pt dan last.pt disimpan bila tersedia; finalisasi Ultralytics dapat menghapus optimizer dari checkpoint.")
    provenance = {"format": "inline-notebook", "preset": preset["name"], "code_sha256": "CODE_SHA256_PLACEHOLDER",
                  "code_hash_scheme": "sha256_joined_code_cells_with_hash_placeholder", "ultralytics": "8.4.155"}
    code("training-definitions", _literal("NOTEBOOK_PROVENANCE", provenance) + '\ndef source_provenance():\n    return deepcopy(NOTEBOOK_PROVENANCE)\n\n' + _training(preset))
    code("training", '''# Satu pemanggilan = satu eksperimen, tanpa loop seed/model.
SUMMARY = run_experiment(CONFIG)
''')
    md("results-heading", "## Ringkasan dan Unduh Hasil\nUnduh best.pt dan CSV epoch. Kaggle Save Version menyimpan folder output. ZIP opsional lewat CREATE_RESULTS_ZIP=True. Gunakan teacher best.pt yang sama sebagai Input seluruh KD full. Colab: unduh hasil sebelum session berakhir.")
    code("results", '''RUN_DIR = Path(SUMMARY["run_dir"])
print(f"Best epoch {SUMMARY['best_epoch']} | best.pt: {SUMMARY['best_pt']}", flush=True)
print("mAP50-95:", SUMMARY["metrics"]["mAP50_95"], flush=True)
if EVALUATE_COCO_AREA:
    print("COCO area:", SUMMARY["metrics"]["coco_area"], flush=True)
RESULT_ZIP = create_results_zip(RUN_DIR, OUTPUT_ROOT) if CREATE_RESULTS_ZIP else None
try:
    from IPython.display import display, FileLink
    for result_path in (RUN_DIR / "weights" / "best.pt", RUN_DIR / "epoch_history.csv", RUN_DIR / "checkpoint_manifest.csv", RESULT_ZIP):
        if result_path is not None:
            display(FileLink(str(result_path)))
except ImportError:
    print("Folder hasil:", RUN_DIR, flush=True)

# Opsional Colab: uncomment untuk langsung mengunduh ZIP.
# from google.colab import files
# if RESULT_ZIP is not None:
#     files.download(str(RESULT_ZIP))
''')
    digest = hashlib.sha256("\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code").encode()).hexdigest()
    for cell in cells:
        cell["source"] = [line.replace("CODE_SHA256_PLACEHOLDER", digest) for line in cell["source"]]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "inatrc_standalone": {**preset, "code_sha256": digest, "code_hash_scheme": provenance["code_hash_scheme"]}}, "cells": cells}


def _serialized(preset):
    return json.dumps(render_notebook(preset), ensure_ascii=False, indent=1) + "\n"


def generate_notebooks(output_dir=OUTPUT_DIR):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for preset in PRESETS:
        path = output / f"{preset['name']}.ipynb"
        path.write_text(_serialized(preset), encoding="utf-8")
        paths.append(path)
    return paths


def check_notebooks(output_dir=OUTPUT_DIR):
    output = Path(output_dir)
    return [f"{preset['name']}.ipynb" for preset in PRESETS
            if not (output / f"{preset['name']}.ipynb").is_file()
            or (output / f"{preset['name']}.ipynb").read_text(encoding="utf-8") != _serialized(preset)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        changed = check_notebooks()
        if changed:
            raise SystemExit("Regenerate: " + ", ".join(changed))
        print("Sembilan notebook mandiri konsisten.")
    else:
        for path in generate_notebooks():
            print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
