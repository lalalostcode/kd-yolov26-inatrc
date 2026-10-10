"""Notebook mandiri harus terbaca, konsisten, dan benar-benar berjalan tanpa repo."""

import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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


@pytest.mark.parametrize("expected", EXPECTED[3:], ids=[row[0] for row in EXPECTED[3:]])
def test_inline_kd_only_gradient_teacher_and_optimizer(expected):
    import torch
    from ultralytics.nn.distill_model import DistillationModel

    notebook = nbformat.read(NOTEBOOK_DIR / f"{expected[0]}.ipynb", as_version=4)
    namespace = _definition_namespace(notebook)
    selected = torch.device("cpu")
    torch.manual_seed(42)
    student = namespace["_model"](expected[2], 160, selected)
    teacher = namespace["calibrate_diagnostic_bn"](namespace["_model"]("m", 160, selected), 160)
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
            loss, items = wrapper(namespace["_batch"](160, selected))
            assert bool(loss.isfinite().all()) and float(items["dis_loss"]) > 0
            loss[-1].backward()  # Loss KD saja; loss deteksi tidak menutupi gradient yang putus.
            for parameters in (wrapper.student_model.parameters(), wrapper.projector.parameters()):
                gradient = namespace["_gradient_summary"](parameters)
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


# Harness mengubah mode saja; seluruh definisi training/KD berasal dari notebook yang disalin.
# exec diperlukan oleh test harness untuk menjalankan cell, tidak ditanam dalam notebook.
ISOLATED_HARNESS = r'''
import builtins
import importlib.metadata
import json
import os
from pathlib import Path
import sys

root = Path.cwd()
os.environ["YOLO_CONFIG_DIR"] = str(root / "yolo-settings")
os.environ["MPLCONFIGDIR"] = str(root / "matplotlib")
os.environ["WANDB_MODE"] = "disabled"
os.environ.pop("WANDB_API_KEY", None)
original_import = builtins.__import__
def independent_import(name, *args, **kwargs):
    if name == "inatrc_kd" or name.startswith("inatrc_kd."):
        raise AssertionError("A standalone notebook imported repository code: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = independent_import
# IPython menaruh kelas cell pada modul __main__; samakan agar torch.save dapat pickle.
namespace = globals()
notebook = json.loads((root / "experiment.ipynb").read_text(encoding="utf-8"))
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    source = source if isinstance(source, str) else "".join(source)
    if cell["id"] == "setup":
        def setup_dependencies(*args, **kwargs):
            expected = {"ultralytics": "8.4.155", "wandb": "0.28.1", "faster-coco-eval": "1.6.7"}
            for name, version in expected.items():
                assert importlib.metadata.version(name) == version
            print("TEST: pip installation skipped; existing pinned dependencies checked.", flush=True)
        namespace["setup_dependencies"] = setup_dependencies
    print("TEST CELL " + cell["id"], flush=True)
    exec(compile(source, "experiment.ipynb:" + cell["id"], "exec"), namespace)
    if cell["id"] == "config":
        namespace.update(MODE="smoke", DEVICE="cpu", WANDB_MODE="disabled", SEED=42,
                         DATASET_ROOT="", TEACHER_CKPT="", MODEL_WEIGHTS=None,
                         INPUT_ROOT=str(root / "inputs"), OUTPUT_ROOT=str(root / "outputs"))
        (root / "inputs").mkdir(exist_ok=True)
summary_path = root / "outputs" / "latest_run.json"
assert summary_path.is_file(), "Notebook did not complete one experiment."
summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
        assert coco_paths, "Final COCO metrics should be available separately."
        bundles = list((root / "outputs").glob("*.zip"))
        assert len(bundles) == 1, "Notebook must provide one downloadable result bundle."
        with zipfile.ZipFile(bundles[0]) as bundle:
            assert any(name.endswith("weights/best.pt") for name in bundle.namelist())
            assert any(name.endswith("epoch_history.csv") for name in bundle.namelist())
            assert not any(".env" in name or name.endswith("settings.json") for name in bundle.namelist())
