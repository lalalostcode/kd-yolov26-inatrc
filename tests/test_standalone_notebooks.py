"""Notebook mandiri harus terbaca, konsisten, dan benar-benar berjalan tanpa repo."""

import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

import nbformat
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks" / "standalone"
EXPECTED = (
    ("01_teacher_yolo26m", "teacher", "n", "none", "full"),
    ("02_baseline_yolo26s", "student", "s", "none", "full"),
    ("03_baseline_yolo26n", "student", "n", "none", "full"),
    ("04_native_yolo26s", "student", "s", "native", "full"),
    ("05_native_yolo26n", "student", "n", "native", "full"),
    ("06_crosskd_yolo26s", "student", "s", "crosskd", "smoke"),
    ("07_crosskd_yolo26n", "student", "n", "crosskd", "smoke"),
    ("08_csakd_yolo26s", "student", "s", "csakd", "smoke"),
    ("09_csakd_yolo26n", "student", "n", "csakd", "smoke"),
)
HEADINGS = (
    "Konfigurasi Eksperimen", "Instalasi Dependency", "Pemeriksaan GPU",
    "Penemuan dan Audit Dataset", "Model dan Teacher", "Algoritme KD",
    "Logging per Epoch", "Training", "Evaluasi best.pt", "Ringkasan dan Unduh Hasil",
)
_ISOLATED_INITIAL_STATES = {}
_ISOLATED_TEACHER_STATES = {}


@pytest.fixture(scope="module")
def standalone_generator():
    spec = importlib.util.spec_from_file_location(
        "standalone_notebook_generator", REPO_ROOT / "scripts" / "generate_standalone_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=EXPECTED, ids=[row[0] for row in EXPECTED])
def standalone_notebook(request):
    expected = request.param
    notebook = nbformat.read(NOTEBOOK_DIR / f"{expected[0]}.ipynb", as_version=4)
    return expected, notebook


def _code_cells(notebook):
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _source(cell):
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _cell(notebook, cell_id):
    return next(cell for cell in _code_cells(notebook) if cell["id"] == cell_id)


def _literal_config(notebook):
    # Jangan menjalankan setup saat hanya membaca pilihan pengguna.
    namespace = {}
    exec(compile(_source(_cell(notebook, "config")), "standalone-config", "exec"), namespace)
    return namespace


def _tree(notebook):
    return ast.parse("\n".join(_source(cell) for cell in _code_cells(notebook)))


def _definition_namespace(notebook):
    """Baca definisi langsung dari notebook, tanpa setup, audit, atau training."""
    namespace = {"__name__": "standalone_acceptance"}
    for cell in _code_cells(notebook):
        if cell["id"] in {"config", "imports"} or cell["id"].endswith("-definitions"):
            exec(compile(_source(cell), "standalone:" + cell["id"], "exec"), namespace)
            if cell["id"] == "config":
                namespace["OUTPUT_ROOT"] = str(REPO_ROOT / "outputs" / "test-runtime" / "standalone-definitions")
                namespace["INPUT_ROOT"] = str(REPO_ROOT / "outputs" / "test-runtime")
    return namespace


def test_exact_standalone_presets_and_generator_parity(standalone_generator):
    assert isinstance(standalone_generator.PRESETS, tuple)
    assert [
        (preset["name"], preset["stage"], preset["student"], preset["method"], preset["mode"])
        for preset in standalone_generator.PRESETS
    ] == list(EXPECTED)
    assert sorted(path.stem for path in NOTEBOOK_DIR.glob("*.ipynb")) == [row[0] for row in EXPECTED]
    assert standalone_generator.check_notebooks() == []


def test_standalone_notebook_schema_syntax_defaults_and_bahasa(standalone_notebook):
    expected, notebook = standalone_notebook
    nbformat.validate(notebook)
    for cell in _code_cells(notebook):
        compile(_source(cell), expected[0] + ":" + cell["id"], "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
    config = _literal_config(notebook)
    assert (config["STAGE"], config["STUDENT"], config["METHOD"], config["MODE"]) == expected[1:]
    assert (config["SEED"], config["WANDB_MODE"]) == (42, "disabled")
    assert config["BENCHMARK_TEST"] is False
    assert config["EVALUATE_COCO_AREA"] is False and config["CREATE_RESULTS_ZIP"] is False
    assert config.get("TEACHER_CKPT") in (None, "")
    markdown = "\n".join(_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    for heading in HEADINGS:
        assert heading in markdown
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))


def test_standalone_code_fingerprint_can_be_recomputed(standalone_notebook):
    _, notebook = standalone_notebook
    provenance = notebook["metadata"]["inatrc_standalone"]
    assert provenance["code_hash_scheme"] == "sha256_joined_code_cells_with_hash_placeholder"
    recorded = provenance["code_sha256"]
    code = "\n".join(_source(cell).replace(recorded, "CODE_SHA256_PLACEHOLDER") for cell in _code_cells(notebook))
    assert hashlib.sha256(code.encode("utf-8")).hexdigest() == recorded
    definitions = ast.parse(_source(_cell(notebook, "training-definitions")))
    assignment = next(node for node in definitions.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "NOTEBOOK_PROVENANCE" for target in node.targets))
    embedded = ast.literal_eval(assignment.value)
    assert embedded["code_sha256"] == recorded and embedded["code_hash_scheme"] == provenance["code_hash_scheme"]


def test_standalone_code_is_visible_without_repository_bootstrap(standalone_notebook):
    _, notebook = standalone_notebook
    nodes = list(ast.walk(_tree(notebook)))
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "Notebook tidak boleh memakai relative import modul repo."
            assert not (node.module or "").startswith("inatrc_kd")
        if isinstance(node, ast.Import):
            assert all(not name.name.startswith("inatrc_kd") for name in node.names)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"exec", "eval", "__import__"}, "Source harus terlihat sebagai Python biasa."
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {"git", "uv"}, "Setup notebook harus memakai pip tanpa mengambil source repo."
        if isinstance(node, ast.Attribute):
            assert not (isinstance(node.value, ast.Name) and node.value.id == "base64")
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"insert", "append"}
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys" and node.func.value.attr == "path"
        for node in nodes
    )
    # YAML konfigurasi ditanam sebagai data; tidak perlu configs/*.yaml di runtime.
    constants = [node.value for node in nodes if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any(value.startswith("configs/") for value in constants)
    assert not any(value.endswith(".py") for value in constants)


def test_standalone_run_all_calls_one_experiment_and_only_selected_kd(standalone_notebook):
    expected, notebook = standalone_notebook
    trees = [ast.parse(_source(cell)) for cell in _code_cells(notebook)]
    calls = [(tree, node) for tree in trees for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_experiment"]
    assert len(calls) == 1
    tree, call = calls[0]
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "CONFIG"
    assert not any(call in list(ast.walk(node)) for node in ast.walk(tree)
                   if isinstance(node, (ast.For, ast.AsyncFor, ast.While)))
    classes = {node.name for node in ast.walk(_tree(notebook)) if isinstance(node, ast.ClassDef)}
    cross_classes = {name for name in classes if name.startswith("CrossKD")}
    csa_classes = {name for name in classes if name.startswith("CSAKD")}
    assert bool(cross_classes) == (expected[3] == "crosskd")
    assert bool(csa_classes) == (expected[3] == "csakd")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "evaluate" and isinstance(node.func.value, ast.Name)
                   and node.func.value.id.lower().startswith("test")
                   for tree in trees for node in ast.walk(tree))


def _file_snapshot(directory):
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


def test_standalone_generator_deterministic_and_check_never_mutates(standalone_generator, tmp_path):
    output = tmp_path / "notebooks"
    paths = standalone_generator.generate_notebooks(output_dir=output)
    assert [path.stem for path in paths] == [row[0] for row in EXPECTED]
    rendered = {path.name: path.read_bytes() for path in paths}
    standalone_generator.generate_notebooks(output_dir=output)
    assert {path.name: path.read_bytes() for path in paths} == rendered
    assert standalone_generator.check_notebooks(output_dir=output) == []
    # Sebuah file milik pengguna tidak boleh dihapus oleh generator atau --check.
    notes = output / "notes.txt"
    notes.write_text("Keep user notes.", encoding="utf-8")
    paths[0].unlink()
    paths[1].write_text("{}\n", encoding="utf-8")
    before = _file_snapshot(output)
    assert standalone_generator.check_notebooks(output_dir=output) == [paths[0].name, paths[1].name]
    assert _file_snapshot(output) == before
    standalone_generator.generate_notebooks(output_dir=output)
    assert notes.read_text(encoding="utf-8") == "Keep user notes."


def test_standalone_generator_check_cli_is_read_only_from_other_directory(tmp_path):
    before = _file_snapshot(NOTEBOOK_DIR)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_standalone_notebooks.py"), "--check"],
        cwd=tmp_path, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _file_snapshot(NOTEBOOK_DIR) == before


def test_standalone_embedded_training_and_evaluation_protocol_is_fair(standalone_notebook):
    _, notebook = standalone_notebook
    namespace = _definition_namespace(notebook)
    full = namespace["FULL_PROTOCOL"]
    expected = {
        "imgsz": 640, "batch": 16, "epochs": 100, "patience": 25,
        "optimizer": "AdamW", "lr0": 0.001, "weight_decay": 0.0005,
        "warmup_epochs": 3.0, "amp": True, "deterministic": True,
    }
    assert {key: full[key] for key in expected} == expected
    assert full == yaml.safe_load((REPO_ROOT / "configs" / "training.yaml").read_text(encoding="utf-8"))["full"]
    evaluation = namespace["EVALUATION_PROTOCOL"]
    assert evaluation["split"] == "val" and evaluation["nms"] is None
    assert evaluation["conf"] == 0.001 and evaluation["iou"] == 0.7 and evaluation["max_det"] == 300
    assert evaluation["augment"] is False and evaluation["quantize"] is None
    assert namespace["SMOKE_PROTOCOL"]["nbs"] == 2
    assert namespace["DATASET_PROTOCOL"]["expected_images"] == 3250
    assert namespace["DATASET_PROTOCOL"]["names"] == list(namespace["CLASS_NAMES"])


def test_standalone_full_factory_requires_real_dataset_and_explicit_kd_teacher(standalone_notebook, tmp_path):
    expected, notebook = standalone_notebook
    namespace = _definition_namespace(notebook)
    namespace.update(MODE="full", DEVICE="cpu", INPUT_ROOT=str(tmp_path / "inputs"),
                     OUTPUT_ROOT=str(tmp_path / "outputs"), DATASET_ROOT="", TEACHER_CKPT="")
    with pytest.raises((ValueError, RuntimeError), match="[Dd]ataset|InaTRC"):
        namespace["make_config"]()
    dataset = tmp_path / "inputs" / "inatrc"
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (dataset / split / kind).mkdir(parents=True)
    namespace["DATASET_ROOT"] = str(dataset)
    if expected[3] != "none":
        with pytest.raises(ValueError, match="teacher|TEACHER"):
            namespace["make_config"]()
        namespace["TEACHER_CKPT"] = str(tmp_path / "shared-inatrc-teacher" / "best.pt")
    config = namespace["make_config"]()
    scale = "m" if expected[1] == "teacher" else expected[2]
    assert config["model"] == f"yolo26{scale}.pt"
    assert config["dataset"]["root"] == str(dataset.resolve())
    assert config["mode"] == "full" and config["training"]["epochs"] == 100
    assert config["evaluation"]["split"] == "val"
    if expected[3] == "none":
        assert config["kd"]["teacher_checkpoint"] is None
    else:
        assert config["kd"]["teacher_checkpoint"].endswith("best.pt")
    namespace["MODE"] = "smoke"
    namespace["DATASET_ROOT"] = ""
    namespace["INPUT_ROOT"] = str(tmp_path / "empty")
    namespace["TEACHER_CKPT"] = ""
    smoke = namespace["make_config"]()
    assert smoke["model"] == f"yolo26{scale}.yaml" and smoke["kd"]["teacher_checkpoint"] is None


def test_standalone_epoch_logger_tracks_tied_best_and_final_validation_once(tmp_path):
    notebook = nbformat.read(NOTEBOOK_DIR / "03_baseline_yolo26n.ipynb", as_version=4)
    namespace = _definition_namespace(notebook)
    recorder_type = namespace["EpochRecorder"]
    tracker = Mock()
    config = {"training": {"epochs": 10}, "method": "none",
              "tracking": {"upload_checkpoint_each_epoch": False}}
    recorder = recorder_type(config, tmp_path, tracker)
    for epoch, fitness in enumerate((0.25, 0.2, 0.25)):
        trainer = SimpleNamespace(
            epoch=epoch, fitness=fitness, best_fitness=0.25, wdir=tmp_path / "weights",
            device=SimpleNamespace(type="cpu"), tloss=None,
            label_loss_items=Mock(return_value={"train/box_loss": 1.0, "train/cls_loss": 2.0}),
            metrics={"metrics/mAP50-95(B)": fitness}, lr={"lr/pg0": 0.001}, validator=None,
        )
        recorder.on_epoch_start(trainer)
        recorder.on_save(trainer)
        recorder.on_fit_epoch_end(trainer)
    recorder.on_fit_epoch_end(trainer)
    assert tracker.log.call_count == 3
    assert recorder.best_epoch == 3
    assert [row["checkpoint/is_best"] for row in recorder.rows] == [True, False, True]
    assert [row["checkpoint/best_epoch"] for row in recorder.rows] == [1, 1, 3]
    assert len(_csv_rows(tmp_path / "epoch_history.csv")) == 3
    assert len(_csv_rows(tmp_path / "checkpoint_manifest.csv")) == 3
    tracker.epoch_artifact.assert_not_called()


def test_optional_exports_keep_primary_evaluator_and_metric_definition(tmp_path, monkeypatch):
    import numpy as np
    import torch
    notebook = nbformat.read(NOTEBOOK_DIR / "03_baseline_yolo26n.ipynb", as_version=4)
    namespace = _definition_namespace(notebook)
    native = torch.nn.Linear(2, 2)
    native.model = [SimpleNamespace(nc=5)]
    native.names = dict(enumerate(namespace["CLASS_NAMES"]))
    box = SimpleNamespace(mp=0.6, mr=0.7, map50=0.8, map=0.51,
                          ap_class_index=np.arange(5), all_ap=np.full((5, 10), 0.51),
                          p=np.full(5, 0.6), r=np.full(5, 0.7), ap50=np.full(5, 0.8))
    result = SimpleNamespace(box=box, save_dir=tmp_path / "val", speed={"inference": 1.0},
                             coco_area={key: 0.51 for key in namespace["COCO_METRIC_KEYS"]})
    val = Mock(return_value=result)
    monkeypatch.setitem(namespace, "YOLO", lambda path: SimpleNamespace(model=native, val=val))
    monkeypatch.setitem(namespace, "get_flops", lambda model, imgsz: 1.0)
    namespace.update(MODE="smoke", DEVICE="cpu", INPUT_ROOT=str(tmp_path / "inputs"),
                     OUTPUT_ROOT=str(tmp_path / "outputs"), DATASET_ROOT="")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test fixture; YOLO loading mocked")
    metrics, options = [], []
    for enabled in (False, True):
        namespace["EVALUATE_COCO_AREA"] = enabled
        config = namespace["make_config"]()
        run = tmp_path / str(enabled)
        run.mkdir()
        metrics.append(namespace["evaluate_checkpoint"](checkpoint, "unused.yaml", run, config))
        options.append(val.call_args.kwargs)
        assert (run / "coco_area.json").exists() == enabled
        assert ("coco_area" in metrics[-1]) == enabled
    assert options[0]["validator"] is namespace["SafeDetectionValidator"]
    assert options[1]["validator"] is namespace["AreaAuditValidator"]
    for key in ("nms", "conf", "iou", "max_det", "augment", "quantize", "split"):
        assert options[0][key] == options[1][key]
    for key in ("precision", "recall", "mAP50", "mAP50_95", "per_class", "complexity"):
        assert metrics[0][key] == metrics[1][key]


def test_teacher_seed_is_fixed_but_student_seed_can_change(tmp_path):
    for filename, stage in (("01_teacher_yolo26m", "teacher"), ("03_baseline_yolo26n", "student")):
        namespace = _definition_namespace(nbformat.read(NOTEBOOK_DIR / f"{filename}.ipynb", as_version=4))
        namespace.update(MODE="smoke", SEED=43, DEVICE="cpu", INPUT_ROOT=str(tmp_path / "inputs"),
                         OUTPUT_ROOT=str(tmp_path / "outputs"), DATASET_ROOT="")
        if stage == "teacher":
            with pytest.raises(ValueError, match="Teacher.*42"):
                namespace["make_config"]()
        else:
            assert namespace["make_config"]()["training"]["seed"] == 43


def test_coco8_config_bypasses_only_inatrc_requirements(standalone_notebook, tmp_path):
    expected, notebook = standalone_notebook
    namespace = _definition_namespace(notebook)
    namespace.update(BENCHMARK_TEST=True, MODE="full", DEVICE="cpu", SEED=42,
                     PLATFORM_INFO={"name": "local"}, OUTPUT_ROOT=str(tmp_path),
                     DATASET_ROOT="/missing/InaTRC", EXPECTED_IMAGES=1,
                     EVALUATE_COCO_AREA=True, CREATE_RESULTS_ZIP=True,
                     TEACHER_CKPT="teacher-coco8.pt" if expected[3] != "none" else "")
    namespace["discover_dataset_root"] = Mock(side_effect=AssertionError("InaTRC discovery forbidden"))
    namespace["validate_config"] = Mock(side_effect=AssertionError("InaTRC class/split validation forbidden"))
    config = namespace["make_config"]()
    scale = "m" if expected[1] == "teacher" else expected[2]
    assert config["model"] == f"yolo26{scale}.pt" and config["benchmark_test"]
    assert config["dataset"]["yaml"] == "coco8.yaml" and len(config["dataset"]["names"]) == 80
    assert config["dataset"]["names"][0] == "person" and config["dataset"]["names"][79] == "toothbrush"
    assert {key: config["training"][key] for key in ("epochs", "imgsz", "batch", "workers", "seed", "nbs")} == {
        "epochs": 1, "imgsz": 320, "batch": 2, "workers": 0, "seed": 42, "nbs": 2,
    }
    assert config["training"]["amp"] is False and config["tracking"]["mode"] == "disabled"
    assert not config["evaluation"]["coco_area"] and not config["create_results_zip"]
    namespace["SEED"] = 43
    with pytest.raises(ValueError, match="seed 42"):
        namespace["make_config"]()
    namespace.update(SEED=42, PLATFORM_INFO={"name": "kaggle"})
    with pytest.raises(ValueError, match="GPU"):
        namespace["make_config"]()


@pytest.mark.parametrize("filename", ["04_native_yolo26s", "06_crosskd_yolo26s", "08_csakd_yolo26s"])
def test_coco8_kd_requires_trained_teacher_not_raw_pretrained(filename, tmp_path):
    notebook = nbformat.read(NOTEBOOK_DIR / f"{filename}.ipynb", as_version=4)
    namespace = _definition_namespace(notebook)
    namespace.update(BENCHMARK_TEST=True, DEVICE="cpu", PLATFORM_INFO={"name": "local"}, OUTPUT_ROOT=str(tmp_path))
    with pytest.raises(ValueError, match="TEACHER_CKPT"):
        namespace["make_config"]()
    namespace["TEACHER_CKPT"] = "teacher-coco8.pt"
    config = namespace["make_config"]()
    report = {"path": "teacher-coco8.pt", "names": namespace["ordered_names"](config["dataset"]["names"]),
              "diagnostics_only": False, "benchmark_metadata": None}
    teacher_check = Mock(return_value=report)
    namespace["check_teacher"] = teacher_check
    with pytest.raises(ValueError, match="trained COCO8 teacher"):
        namespace["configure_kd"](config)
    teacher_check.assert_called_with(config["kd"]["teacher_checkpoint"], expected_nc=80)
    report["benchmark_metadata"] = {"status": "PASS", "stage": "teacher", "dataset": "coco8.yaml",
                                    "seed": 42, "completed_epochs": 1, "optimizer_updates": 2}
    assert namespace["configure_kd"](config)["distill_model"] == "teacher-coco8.pt"


def test_coco8_preserves_default_five_class_teacher_guard(tmp_path):
    from inatrc_kd.preflight import check_teacher, make_synthetic_teacher
    teacher = make_synthetic_teacher(tmp_path / "five-class-teacher.pt")
    assert check_teacher(teacher)["nc"] == 5
    with pytest.raises(ValueError, match="80 classes"):
        check_teacher(teacher, expected_nc=80)


def test_coco8_run_all_selects_one_branch_and_uses_real_kd(standalone_notebook):
    expected, notebook = standalone_notebook
    statement = ast.parse(_source(_cell(notebook, "training"))).body[0]
    assert isinstance(statement, ast.If) and statement.test.id == "BENCHMARK_TEST"
    assert statement.body[0].value.func.id == "run_benchmark"
    assert statement.orelse[0].value.func.id == "run_experiment"
    namespace = _definition_namespace(notebook)
    assert namespace["BenchmarkTrainer"].__bases__ == (namespace[
        "CustomKDTrainer" if expected[3] in {"crosskd", "csakd"} else "SafeDetectionTrainer"],)
    assert namespace["BenchmarkTrainer"].build_dataset is namespace["DetectionTrainer"].build_dataset
    assert namespace["BenchmarkTrainer"].get_validator is namespace["DetectionTrainer"].get_validator


@pytest.mark.parametrize("expected", EXPECTED[3:], ids=[row[0] for row in EXPECTED[3:]])
def test_inline_kd_only_gradient_teacher_and_optimizer(expected):
    import torch
    from ultralytics.nn.distill_model import DistillationModel
    from inatrc_kd.preflight import _model, _batch, _gradient_summary

    notebook = nbformat.read(NOTEBOOK_DIR / f"{expected[0]}.ipynb", as_version=4)
    namespace = _definition_namespace(notebook)
    selected = torch.device("cpu")
    torch.manual_seed(42)
    student = _model(expected[2], 160, selected)
    teacher = namespace["calibrate_diagnostic_bn"](_model("m", 160, selected), 160)
    if expected[3] == "native":
        wrapper = DistillationModel(teacher_model=teacher, student_model=student).train()
    else:
        cls = namespace["CrossKDModel" if expected[3] == "crosskd" else "CSAKDModel"]
        wrapper = cls(teacher, student, namespace["KD_PROTOCOL"][expected[3]]).train()
    try:
        before = namespace["_state_hash"](wrapper.teacher_model)
        optimizer = torch.optim.AdamW(parameter for parameter in wrapper.parameters() if parameter.requires_grad)
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert all(id(parameter) in optimized for parameter in wrapper.projector.parameters())
        hooks = lambda: sum(len(module._forward_hooks) + len(module._forward_pre_hooks) for module in wrapper.modules())
        initial_hooks = hooks()
        for _ in range(2):
            wrapper.zero_grad(set_to_none=True)
            loss, items = wrapper(_batch(160, selected))
            assert bool(loss.isfinite().all()) and float(items["dis_loss"]) > 0
            loss[-1].backward()  # Loss KD saja; loss deteksi tidak menutupi gradient yang putus.
            for parameters in (wrapper.student_model.parameters(), wrapper.projector.parameters()):
                gradient = _gradient_summary(parameters)
                assert gradient["finite"] and gradient["nonzero"]
            assert all(not parameter.requires_grad and parameter.grad is None for parameter in wrapper.teacher_model.parameters())
            assert not wrapper.teacher_model.training
            assert namespace["_state_hash"](wrapper.teacher_model) == before
            assert hooks() == initial_hooks
            if expected[3] != "native":
                assert not wrapper._student_feats and not wrapper._teacher_feats
    finally:
        wrapper._remove_feature_hooks()
        wrapper._student_feats.clear()
        wrapper._teacher_feats.clear()
        wrapper.zero_grad(set_to_none=True)
    assert hooks() == 0


# Harness memilih smoke CPU/benchmark; seluruh definisi training/KD berasal dari notebook yang disalin.
# exec diperlukan oleh test harness untuk menjalankan cell, tidak ditanam dalam notebook.
ISOLATED_HARNESS = r'''
import builtins
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

root = Path.cwd()
os.environ["YOLO_CONFIG_DIR"] = str(root / "yolo-settings")
os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
os.environ["WANDB_MODE"] = "disabled"
os.environ.pop("WANDB_API_KEY", None)
original_import = builtins.__import__
def independent_import(name, *args, **kwargs):
    if name == "inatrc_kd" or name.startswith("inatrc_kd."):
        raise AssertionError("A standalone notebook imported repository code: " + name)
    if name.startswith("faster_coco_eval") and os.environ.get("TEST_OPTIONAL_COCO", "0") != "1":
        raise AssertionError("COCO disabled must work without faster_coco_eval.")
    return original_import(name, *args, **kwargs)
builtins.__import__ = independent_import
# IPython menaruh kelas cell pada modul __main__; samakan agar torch.save dapat pickle.
namespace = globals()
stage_profile = {}
process_started = time.perf_counter()
notebook = json.loads((root / "experiment.ipynb").read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    source = source if isinstance(source, str) else "".join(source)
    if cell["id"] == "setup":
        def setup_dependencies(*args, **kwargs):
            expected = {"ultralytics": "8.4.155", "wandb": "0.28.1"}
            if os.environ.get("TEST_OPTIONAL_COCO", "0") == "1":
                expected["faster-coco-eval"] = "1.6.7"
            for name, version in expected.items():
                assert importlib.metadata.version(name) == version
            print("TEST: pip installation skipped; existing pinned dependencies checked.", flush=True)
        namespace["setup_dependencies"] = setup_dependencies
    print("TEST CELL " + cell["id"], flush=True)
    exec(compile(source, "experiment.ipynb:" + cell["id"], "exec"), namespace)
    if cell["id"] == "training-definitions":
        def measured(function, name):
            def call(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    row = stage_profile.setdefault(name, {"calls": 0, "seconds": 0.0})
                    row["calls"] += 1
                    row["seconds"] += time.perf_counter() - started
            return call
        for name in ("environment_report", "audit_dataset", "verify_dataset_unchanged", "inspect_model", "evaluate_checkpoint", "_verify_image", "_sha256_file", "make_synthetic_teacher", "create_results_zip"):
            if name in namespace:
                namespace[name] = measured(namespace[name], name)
    if cell["id"] == "config":
        benchmark = os.environ.get("TEST_COCO8_BENCHMARK", "0") == "1"
        namespace.update(MODE="smoke", DEVICE="cpu", WANDB_MODE="disabled", SEED=42,
                         BENCHMARK_TEST=benchmark, DATASET_ROOT="",
                         TEACHER_CKPT=os.environ.get("TEST_BENCHMARK_TEACHER", "") if benchmark else "",
                         MODEL_WEIGHTS=os.environ.get("TEST_BENCHMARK_WEIGHTS") if benchmark else None,
                         INPUT_ROOT=str(root / "inputs"), OUTPUT_ROOT=str(root / "outputs"))
        namespace["EVALUATE_COCO_AREA"] = os.environ.get("TEST_OPTIONAL_COCO", "0") == "1"
        namespace["CREATE_RESULTS_ZIP"] = os.environ.get("TEST_OPTIONAL_ZIP", "0") == "1"
        (root / "inputs").mkdir(exist_ok=True)
summary_path = root / "outputs" / ("benchmark_latest.json" if benchmark else "latest_run.json")
assert summary_path.is_file(), "Notebook did not complete one experiment."
summary = json.loads(summary_path.read_text(encoding="utf-8"))
run = Path(summary["run_dir"])
metadata = summary if benchmark else json.loads((run / "metadata.json").read_text(encoding="utf-8"))
profile = {"stages": stage_profile, "process_seconds": time.perf_counter() - process_started,
           "training_seconds": metadata.get("training_seconds", metadata.get("duration_seconds")), "metrics": summary["metrics"],
           "initial_student_state_sha256": metadata["initial_student_state_sha256"],
           "files": sorted(str(path.relative_to(run)) for path in run.rglob("*") if path.is_file())}
(root / "outputs" / "test_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
print("ISOLATED_SUMMARY=" + json.dumps(summary), flush=True)
'''


def _csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.mark.smoke
@pytest.mark.parametrize("expected", EXPECTED, ids=[row[0] for row in EXPECTED])
def test_standalone_smoke_runs_outside_checkout_without_repo_import(expected, tmp_path):
    # System temp dipakai agar cwd/isi input benar-benar berada di luar checkout.
    with tempfile.TemporaryDirectory(prefix="inatrc-standalone-") as location:
        root = Path(location).resolve()
        assert REPO_ROOT not in root.parents
        notebook = (NOTEBOOK_DIR / f"{expected[0]}.ipynb").read_bytes()
        (root / "experiment.ipynb").write_bytes(notebook)
        (root / "run_notebook.py").write_text(ISOLATED_HARNESS, encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("WANDB_API_KEY", None)
        environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", "run_notebook.py"], cwd=root, env=environment,
            text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=180,
        )
        log = tmp_path / f"{expected[0]}-isolated.log"
        log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        assert completed.returncode == 0, f"Standalone smoke failed; log: {log}\n{log.read_text(encoding='utf-8')[-10000:]}"
        summary = json.loads((root / "outputs" / "latest_run.json").read_text(encoding="utf-8"))
        (tmp_path / f"{expected[0]}-profile.json").write_bytes((root / "outputs" / "test_profile.json").read_bytes())
        assert summary["status"] == "completed" and summary["nonfinal"] is True
        assert summary["mode"] == "smoke" and summary["test_evaluated"] is False
        assert summary["best_epoch"] == 1
        run_dir = Path(summary["run_dir"])
        assert root in run_dir.parents
        mandatory = (
            "weights/best.pt", "results.csv", "epoch_history.csv", "checkpoint_manifest.csv",
            "resolved_config.yaml", "metadata.json", "metrics.json", "per_class.csv",
            "checkpoint_audit.json", "complexity.json", "run_summary.json", "training_summary.json",
        )
        for name in mandatory:
            assert (run_dir / name).is_file(), name
        assert len(_csv_rows(run_dir / "results.csv")) == 1
        history = _csv_rows(run_dir / "epoch_history.csv")
        assert len(history) == 1 and int(history[0]["epoch"]) == 1
        assert int(float(history[0]["checkpoint/best_epoch"])) == 1
        assert history[0]["checkpoint/is_best"] == "True"
        manifest = _csv_rows(run_dir / "checkpoint_manifest.csv")
        assert len(manifest) == 1 and manifest[0]["is_best_at_save"] == "True"
        assert "Best epoch 1" in completed.stdout
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        assert (config["stage"], config["student"], config["method"]) == expected[1:4]
        assert config["training"]["epochs"] == 1 and config["training"]["nbs"] == 2
        assert config["training"]["batch"] == 2 and config["training"]["imgsz"] == 160
        assert config["tracking"]["mode"] == "disabled"
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["optimizer_updates"] > 0
        assert len(metadata["initial_student_state_sha256"]) == 64
        evidence = {
            "preset": expected[0], "optimizer_updates": metadata["optimizer_updates"],
            "initial_student_state_sha256": metadata["initial_student_state_sha256"],
            "initial_teacher_state_sha256": metadata.get("initial_teacher_state_sha256"),
            "teacher_checkpoint_sha256": metadata.get("teacher_checkpoint_sha256"),
            "teacher_frozen": metadata.get("teacher_frozen"),
            "teacher_state_unchanged": metadata.get("teacher_state_unchanged"),
            "adapter_parameters": metadata.get("adapter_parameters"), "best_epoch": summary["best_epoch"],
        }
        (tmp_path / f"{expected[0]}-evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        if expected[1] == "student":
            previous = _ISOLATED_INITIAL_STATES.setdefault(expected[2], metadata["initial_student_state_sha256"])
            assert metadata["initial_student_state_sha256"] == previous, "Initialization differs across methods for the same student/seed."
        checkpoint = summary["metrics"]["checkpoint"]
        assert checkpoint["nc"] == 5 and checkpoint["teacher_present"] is False
        assert checkpoint["projector_present"] is False
        assert summary["metrics"]["split"] == "val" and len(summary["metrics"]["per_class"]) == 5
        assert summary["metrics"]["nms"] is None
        if expected[3] != "none":
            assert metadata["teacher_frozen"] and metadata["teacher_state_unchanged"]
            assert metadata["adapter_parameters"] > 0
            assert any("dis_loss" in name for name in history[0])
            previous = _ISOLATED_TEACHER_STATES.setdefault("shared_m", metadata["initial_teacher_state_sha256"])
            assert metadata["initial_teacher_state_sha256"] == previous, "Synthetic teacher differs across methods."
        assert not list((run_dir / "synthetic_data").rglob("*.cache"))
        audit = json.loads((run_dir / "reports" / "training_data" / "dataset_audit.json").read_text(encoding="utf-8"))
        assert audit["total_images"] == 20
        assert not list((run_dir / "weights").glob("epoch*.pt"))
        coco_paths = sorted(run_dir.glob("*coco*.json"))
        assert bool(coco_paths) == config["evaluation"]["coco_area"]
        bundles = list((root / "outputs").glob("*.zip"))
        assert len(bundles) == int(config["create_results_zip"])
        if bundles:
            with zipfile.ZipFile(bundles[0]) as bundle:
                assert any(name.endswith("weights/best.pt") for name in bundle.namelist())
                assert any(name.endswith("epoch_history.csv") for name in bundle.namelist())
                assert not any(".env" in name or name.endswith("settings.json") for name in bundle.namelist())
        profile = json.loads((tmp_path / f"{expected[0]}-profile.json").read_text(encoding="utf-8"))
        assert "inspect_model" not in profile["stages"] and "environment_report" not in profile["stages"]
        assert profile["stages"]["audit_dataset"]["calls"] == 1
        assert profile["stages"]["verify_dataset_unchanged"]["calls"] == 1
        assert profile["stages"]["_verify_image"]["calls"] == 20


def _execute_coco8_notebook(expected, root, log_dir, teacher=None):
    """Pemakaian ulang harness notebook yang ada dengan input COCO8/pretrained nyata."""
    assets = REPO_ROOT / "outputs" / "coco8-benchmark" / "assets"
    scale = "m" if expected[1] == "teacher" else expected[2]
    (root / "experiment.ipynb").write_bytes((NOTEBOOK_DIR / f"{expected[0]}.ipynb").read_bytes())
    (root / "run_notebook.py").write_text(ISOLATED_HARNESS, encoding="utf-8")
    (root / "inputs").mkdir()
    weights = root / "inputs" / f"yolo26{scale}.pt"
    shutil.copy2(assets / weights.name, weights)
    shutil.copytree(assets / "coco8", root / "outputs" / "benchmark_datasets" / "coco8")
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "WANDB_API_KEY", "TEST_OPTIONAL_COCO", "TEST_OPTIONAL_ZIP"):
        environment.pop(key, None)
    environment.update(PYTHONUTF8="1", TEST_COCO8_BENCHMARK="1", TEST_BENCHMARK_WEIGHTS=str(weights),
                       TEST_BENCHMARK_TEACHER=str(teacher or ""))
    completed = subprocess.run([sys.executable, "-I", "run_notebook.py"], cwd=root, env=environment,
                               text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=300)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{expected[0]}-coco8.log"
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    assert completed.returncode == 0, f"COCO8 CPU failed: {log}\n{log.read_text(encoding='utf-8')[-10000:]}"
    summary = json.loads((root / "outputs" / "benchmark_latest.json").read_text(encoding="utf-8"))
    (log_dir / f"{expected[0]}-coco8.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert summary["status"] == "PASS" and summary["runtime"] == "cpu" and summary["nonfinal"]
    assert summary["nc"] == 80 and (summary["train_images"], summary["val_images"]) == (4, 4)
    assert summary["completed_epochs"] == 1 and summary["best_epoch"] == 1
    assert summary["optimizer_updates"] == summary["gradient_steps"] == summary["loss_batches"] == 2
    assert summary["validation_reloaded"] and not summary["test_evaluated"]
    run = Path(summary["run_dir"])
    assert len(_csv_rows(run / "epoch_history.csv")) == len(_csv_rows(run / "results.csv")) == 1
    assert not list((root / "outputs").rglob("*coco_metrics*")) and not list((root / "outputs").rglob("*.zip"))
    assert not (run / "reports").exists() and not (run / "complexity.json").exists()
    profile = json.loads((root / "outputs" / "test_profile.json").read_text(encoding="utf-8"))
    for name in ("audit_dataset", "verify_dataset_unchanged", "_verify_image", "evaluate_checkpoint", "create_results_zip", "make_synthetic_teacher"):
        assert name not in profile["stages"], f"Benchmark invoked InaTRC/optional stage {name}"
    import torch
    from ultralytics import YOLO
    checkpoint = YOLO(summary["best_pt"])
    assert checkpoint.model.model[-1].nc == 80
    assert not hasattr(checkpoint.model, "teacher_model") and not hasattr(checkpoint.model, "projector")
    assert checkpoint.ckpt["coco8_benchmark"]["status"] == "PASS"
    if expected[3] != "none":
        assert summary["teacher_frozen"] and summary["teacher_state_unchanged"] and summary["kd_loss_batches"] == 2
        assert summary["teacher_checkpoint_sha256"] == hashlib.sha256(Path(teacher).read_bytes()).hexdigest()
    return summary


@pytest.fixture(scope="module")
def trained_coco8_teacher():
    assets = REPO_ROOT / "outputs" / "coco8-benchmark" / "assets"
    if not all((assets / name).exists() for name in ("yolo26m.pt", "yolo26s.pt", "yolo26n.pt", "coco8")):
        pytest.skip("COCO8 benchmark needs the official downloaded assets; tests never download them automatically")
    with tempfile.TemporaryDirectory(prefix="inatrc-coco8-teacher-") as location:
        root = Path(location)
        assert REPO_ROOT not in root.parents
        summary = _execute_coco8_notebook(EXPECTED[0], root, REPO_ROOT / "outputs/coco8-benchmark/smoke-results")
        yield summary


_COCO8_INITIAL_STATES = {}


@pytest.mark.smoke
@pytest.mark.parametrize("expected", EXPECTED, ids=[row[0] for row in EXPECTED])
def test_coco8_benchmark_cpu_outside_checkout(expected, trained_coco8_teacher):
    if expected[1] == "teacher":
        assert trained_coco8_teacher["status"] == "PASS"
        return
    with tempfile.TemporaryDirectory(prefix="inatrc-coco8-student-") as location:
        root = Path(location)
        assert REPO_ROOT not in root.parents
        teacher = trained_coco8_teacher["best_pt"] if expected[3] != "none" else None
        summary = _execute_coco8_notebook(expected, root, REPO_ROOT / "outputs/coco8-benchmark/smoke-results", teacher)
        previous = _COCO8_INITIAL_STATES.setdefault(expected[2], summary["initial_student_state_sha256"])
        assert previous == summary["initial_student_state_sha256"]
