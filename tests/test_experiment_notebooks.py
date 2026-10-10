"""Acceptance checks for fixed experiment notebooks; never run training."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import types

import nbformat
import pytest

from inatrc_kd.config import REPO_ROOT


EXPECTED = (
    ("01_teacher_yolo26m", "teacher", "n", "none", "yolo26m"),
    ("02_baseline_yolo26s", "student", "s", "none", "yolo26s"),
    ("03_baseline_yolo26n", "student", "n", "none", "yolo26n"),
    ("04_native_yolo26s", "student", "s", "native", "yolo26s"),
    ("05_native_yolo26n", "student", "n", "native", "yolo26n"),
    ("06_crosskd_yolo26s", "student", "s", "crosskd", "yolo26s"),
    ("07_crosskd_yolo26n", "student", "n", "crosskd", "yolo26n"),
    ("08_csakd_yolo26s", "student", "s", "csakd", "yolo26s"),
    ("09_csakd_yolo26n", "student", "n", "csakd", "yolo26n"),
)
TEMPLATE_PATH = REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb"
EXPERIMENT_DIR = REPO_ROOT / "notebooks" / "experiments"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "experiment_notebook_generator", REPO_ROOT / "scripts" / "generate_experiment_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def template():
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(params=EXPECTED, ids=[entry[0] for entry in EXPECTED])
def experiment(request):
    preset = request.param
    notebook = nbformat.read(EXPERIMENT_DIR / (preset[0] + ".ipynb"), as_version=4)
    return preset, notebook


def _code(notebook):
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def _source(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def _configuration(notebook):
    namespace = {}
    exec(compile(_code(notebook)[0], "experiment-configuration", "exec"), namespace)
    return namespace


def _assignment(sources, name):
    for source in sources:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets
            ):
                return node
    raise AssertionError(f"Missing notebook assignment: {name}")


def _run_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "command"
    )


def test_exact_nine_presets_and_generated_files(generator):
    assert isinstance(generator.PRESETS, tuple)
    assert [
        (preset["name"], preset["stage"], preset["student"], preset["method"])
        for preset in generator.PRESETS
    ] == [entry[:4] for entry in EXPECTED]
    assert sorted(path.stem for path in EXPERIMENT_DIR.glob("*.ipynb")) == [entry[0] for entry in EXPECTED]
    assert generator.check_notebooks() == []


def test_notebook_valid_unexecuted_and_fixed_defaults(experiment):
    expected, notebook = experiment
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, expected[0], "exec")
            assert cell.execution_count is None
            assert cell.outputs == []
    config = _configuration(notebook)
    assert (config["STAGE"], config["STUDENT"], config["METHOD"]) == expected[1:4]
    assert (config["MODE"], config["SEED"], config["WANDB_MODE"]) == ("smoke", 42, "disabled")
    assert (config["REPO_URL"], config["REPO_COMMIT"]) == ("", "")
    assert config["OUTPUT_ROOT"] == "/kaggle/working/outputs"
    assert notebook.metadata.inatrc_experiment == {
        "preset": expected[0], "stage": expected[1], "student": expected[2],
        "method": expected[3], "model": expected[4],
    }


@pytest.mark.parametrize("field", ["STAGE", "STUDENT", "METHOD"])
def test_accidental_preset_edit_is_rejected(experiment, field):
    _, notebook = experiment
    tree = ast.parse(_code(notebook)[0])
    assignment = _assignment(_code(notebook)[:1], field)
    replacement = {
        "STAGE": "student" if assignment.value.value == "teacher" else "teacher",
        "STUDENT": "s" if assignment.value.value == "n" else "n",
        "METHOD": "native" if assignment.value.value == "none" else "none",
    }[field]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == field for target in node.targets
        ):
            node.value = ast.Constant(value=replacement)
    with pytest.raises(AssertionError, match="Preset"):
        exec(compile(ast.fix_missing_locations(tree), "edited-preset", "exec"), {})


def test_run_all_delegates_exactly_one_experiment(experiment):
    _, notebook = experiment
    sources = _code(notebook)
    parsed = [ast.parse(source) for source in sources]
    calls = [(tree, node) for tree in parsed for node in ast.walk(tree) if _run_call(node)]
    assert len(calls) == 1
    tree, call = calls[0]
    # A training command inside a loop would turn Run All into several runs.
    assert not any(
        call in list(ast.walk(node)) for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    )
    assert not any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "train"
        for parsed_cell in parsed for node in ast.walk(parsed_cell)
    )
    config = _configuration(notebook)
    config["UV_RUN"] = ["python", "-m", "inatrc_kd"]
    assignment = _assignment(sources, "command")
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "one-experiment-command", "exec"), config)
    command = config["command"]
    assert command[:4] == ["python", "-m", "inatrc_kd", "run"]
    for flag, field in (("--stage", "STAGE"), ("--student", "STUDENT"), ("--method", "METHOD"),
                        ("--mode", "MODE"), ("--seed", "SEED"), ("--wandb-mode", "WANDB_MODE")):
        assert command[command.index(flag) + 1] == str(config[field])


def test_full_kd_requires_teacher_but_smoke_does_not(experiment):
    expected, notebook = experiment
    guards = [
        node for source in _code(notebook) for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and {"MODE", "METHOD", "TEACHER_CKPT"}.issubset(
            {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        )
    ]
    assert len(guards) == 1
    guard = compile(ast.Module(body=guards, type_ignores=[]), "full-kd-teacher-guard", "exec")
    namespace = {"MODE": "smoke", "METHOD": expected[3], "TEACHER_CKPT": ""}
    exec(guard, namespace)
    namespace["MODE"] = "full"
    if expected[3] != "none":
        with pytest.raises(ValueError, match="teacher"):
            exec(guard, namespace)
    else:
        exec(guard, namespace)
    namespace["TEACHER_CKPT"] = "/kaggle/input/shared-teacher/best.pt"
    exec(guard, namespace)


def test_shared_bootstrap_preserves_uv_and_commented_colab(experiment):
    _, notebook = experiment
    sources = _code(notebook)
    namespace = {"UV": ["uv"], "sys": types.SimpleNamespace(executable="/kernel/python")}
    for name in ("sync", "UV_RUN"):
        node = _assignment(sources, name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "shared-uv-contract", "exec"), namespace)
    sync = namespace["sync"]
    assert "--frozen" in sync
    assert "--no-dev" in sync
    assert sync[sync.index("--python") + 1] == "/kernel/python"
    excluded = [sync[index + 1] for index, item in enumerate(sync) if item == "--no-install-package"]
    assert excluded == ["torch", "torchvision"]
    assert namespace["UV_RUN"] == [
        "uv", "run", "--frozen", "--no-sync", "--python", "/kernel/python", "python", "-m", "inatrc_kd"
    ]
    source = "\n".join(sources)
    for marker in ("COLAB PATHS", "COLAB DRIVE", "COLAB SECRET"):
        assert marker + " BEGIN" in source
        assert marker + " END" in source
    assert "# from google.colab import userdata" in source
    assert "# from google.colab import drive" in source
    assert "from kaggle_secrets import UserSecretsClient" in source
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "google.colab"
        for cell in sources for node in ast.walk(ast.parse(cell))
    )


def test_render_keeps_template_unchanged_and_shared_cells_identical(generator, template, experiment):
    expected, notebook = experiment
    original = deepcopy(template)
    preset = next(preset for preset in generator.PRESETS if preset["name"] == expected[0])
    rendered = generator.render_notebook(template, preset)
    assert template == original
    assert rendered == json.loads((EXPERIMENT_DIR / (expected[0] + ".ipynb")).read_text(encoding="utf-8"))
    assert [cell.get("id") for cell in rendered["cells"]] == [cell.get("id") for cell in template["cells"]]
    template_codes = [cell for cell in template["cells"] if cell["cell_type"] == "code"]
    experiment_codes = [cell for cell in rendered["cells"] if cell["cell_type"] == "code"]
    assert [_source(cell) for cell in experiment_codes[1:]] == [_source(cell) for cell in template_codes[1:]]
    assert rendered["cells"][2:] == template["cells"][2:]


def _snapshot_files(root):
    return {path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def test_generator_is_deterministic_and_check_is_read_only(generator, tmp_path, monkeypatch, capsys):
    output = tmp_path / "experiments"
    paths = generator.generate_notebooks(TEMPLATE_PATH, output)
    assert [path.stem for path in paths] == [entry[0] for entry in EXPECTED]
    expected_bytes = {path.name: path.read_bytes() for path in paths}
    generator.generate_notebooks(TEMPLATE_PATH, output)
    assert {path.name: path.read_bytes() for path in paths} == expected_bytes
    unrelated = output / "notes.txt"
    unrelated.write_text("Preserve user notes.", encoding="utf-8")
    assert generator.check_notebooks(TEMPLATE_PATH, output) == []
    paths[0].unlink()
    paths[1].write_text("{}\n", encoding="utf-8")
    before = _snapshot_files(output)
    original_check = generator.check_notebooks
    monkeypatch.setattr(generator, "check_notebooks", lambda: original_check(TEMPLATE_PATH, output))
    assert generator.main(["--check"]) == 1
    assert paths[0].name in capsys.readouterr().out
    assert original_check(TEMPLATE_PATH, output) == [paths[0].name, paths[1].name]
    assert _snapshot_files(output) == before
    generator.generate_notebooks(TEMPLATE_PATH, output)
    before = _snapshot_files(output)
    assert generator.main(["--check"]) == 0
    assert "sesuai template" in capsys.readouterr().out
    assert _snapshot_files(output) == before
    assert unrelated.read_text(encoding="utf-8") == "Preserve user notes."


def test_render_clears_previous_execution_without_mutating_original(generator, template):
    previous = deepcopy(template)
    cell = next(cell for cell in previous["cells"] if cell["cell_type"] == "code")
    cell["execution_count"] = 7
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "old run\n"}]
    saved = deepcopy(previous)
    rendered = generator.render_notebook(previous, generator.PRESETS[0])
    assert previous == saved
    for cell in rendered["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
