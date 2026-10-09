"""CLI and static Kaggle runner acceptance checks; never execute a notebook."""

import ast
import sys
import types
import nbformat
import pytest

from inatrc_kd.cli import build_parser
from inatrc_kd.config import REPO_ROOT


@pytest.fixture(scope="module")
def notebook():
    path = REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb"
    result = nbformat.read(path, as_version=4)
    nbformat.validate(result)
    return result


def _code(notebook):
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def _assignment(cells, name):
    for source in cells:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
    raise AssertionError(f"Runner lacks configuration assignment {name}")


def _strings(node):
    return [child.value for child in ast.walk(node) if isinstance(child, ast.Constant) and isinstance(child.value, str)]


def _uncomment_block(source, marker):
    """Apply exactly the uncomment action described to notebook users."""
    inside = False
    lines = []
    for line in source.splitlines():
        if f"{marker} BEGIN" in line:
            inside = True
        elif f"{marker} END" in line:
            inside = False
        elif inside:
            indentation = line[:len(line) - len(line.lstrip())]
            assert line.lstrip().startswith("# ")
            line = indentation + line.lstrip()[2:]
        lines.append(line)
    return "\n".join(lines)


def _secret_setup(notebook, *, uncomment=False):
    source = next(cell for cell in _code(notebook) if cell.startswith('if WANDB_MODE == "online"'))
    if uncomment:
        source = _uncomment_block(source, "COLAB SECRET")
    # Execute only Secrets setup; dataset/preflight/training are excluded.
    tree = ast.parse(source)
    assert all(isinstance(node, ast.If) for node in tree.body[:2])
    return compile(ast.Module(body=tree.body[:2], type_ignores=[]), "runner-secrets", "exec")


def test_colab_commented_paths_override_config_and_share_setup(notebook):
    config = _code(notebook)[0]
    original = {}
    exec(compile(config, "runner-defaults", "exec"), original)
    assert (original["PLATFORM"], original["WORK_ROOT"], original["INPUT_ROOT"]) == (
        "kaggle", "/kaggle/working", "/kaggle/input"
    )
    colab = {}
    exec(compile(_uncomment_block(config, "COLAB PATHS"), "runner-colab", "exec"), colab)
    assert (colab["PLATFORM"], colab["WORK_ROOT"], colab["INPUT_ROOT"], colab["OUTPUT_ROOT"]) == (
        "colab", "/content", "/content/inputs", "/content/outputs"
    )
    setup = "\n".join(_code(notebook)[1:])
    assert 'REPO = Path(WORK_ROOT)' in setup
    assert 'INPUT = Path(INPUT_ROOT)' in setup
    assert 'PROVENANCE_PATH = Path(WORK_ROOT)' in setup
    assert 'binary_dir = Path(WORK_ROOT)' in setup
    assert 'str(Path(WORK_ROOT) / "uv-cache")' in setup
    assert "/kaggle/" not in setup
    # All optional blocks, including Drive mounting, compile when uncommented.
    for source in _code(notebook):
        for marker in ("COLAB PATHS", "COLAB DRIVE", "COLAB SECRET"):
            source = _uncomment_block(source, marker)
        compile(source, "runner-colab-all-blocks", "exec")


@pytest.mark.parametrize("platform", ["kaggle", "colab"])
def test_runner_reads_only_selected_secret_provider(notebook, monkeypatch, capsys, platform):
    calls = []

    def get_secret(name):
        calls.append(name)
        return "mock-secret-value"

    # Only the selected provider exists, so wrong-platform imports fail.
    if platform == "kaggle":
        module = types.ModuleType("kaggle_secrets")
        module.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=get_secret)
        monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    else:
        google = types.ModuleType("google")
        google.__path__ = []
        colab = types.ModuleType("google.colab")
        colab.userdata = types.SimpleNamespace(get=get_secret)
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {"WANDB_MODE": "online", "PLATFORM": platform, "CHILD_ENV": {}}
    exec(_secret_setup(notebook, uncomment=platform == "colab"), namespace)
    assert namespace["CHILD_ENV"] == {"WANDB_API_KEY": "mock-secret-value"}
    assert calls == ["WANDB_API_KEY"]
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("platform", ["kaggle", "colab"])
@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_runner_environment_key_priority_and_nononline_removal(notebook, platform, mode):
    namespace = {"WANDB_MODE": mode, "PLATFORM": platform,
                 "CHILD_ENV": {"WANDB_API_KEY": "mock-env-value", "OTHER": "kept"}}
    exec(_secret_setup(notebook), namespace)
    assert namespace["CHILD_ENV"] == ({"WANDB_API_KEY": "mock-env-value", "OTHER": "kept"}
                                      if mode == "online" else {"OTHER": "kept"})


def test_colab_secret_requires_uncomment_and_hides_provider_error(notebook, monkeypatch):
    namespace = {"WANDB_MODE": "online", "PLATFORM": "colab", "CHILD_ENV": {}}
    with pytest.raises(RuntimeError, match="COLAB SECRET"):
        exec(_secret_setup(notebook), namespace)

    def denied(name):
        raise ValueError("mock-sensitive-provider-error")

    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=denied)
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY") as captured:
        exec(_secret_setup(notebook, uncomment=True), namespace)
    assert "mock-sensitive-provider-error" not in str(captured.value)
    assert captured.value.__suppress_context__ is True


def test_cli_defaults_and_research_overrides_parse_without_training(tmp_path):
    parser = build_parser()
    defaults = parser.parse_args(["run"])
    assert (defaults.command, defaults.stage, defaults.student, defaults.method, defaults.seed, defaults.mode) == (
        "run", "student", "n", "none", 42, "smoke"
    )
    assert defaults.upload_checkpoint_each_epoch is False
    selected = parser.parse_args([
        "run", "--stage", "student", "--student", "s", "--method", "native",
        "--seed", "3407", "--mode", "full", "--dataset-root", str(tmp_path),
        "--teacher-ckpt", str(tmp_path / "best.pt"), "--device", "0", "--wandb-mode", "online",
        "--weights", str(tmp_path / "yolo26s.pt"),
    ])
    assert (selected.student, selected.method, selected.seed, selected.mode, selected.device) == (
        "s", "native", 3407, "full", "0"
    )
    assert selected.dataset_root == str(tmp_path)
    assert selected.teacher_ckpt == str(tmp_path / "best.pt")
    assert selected.weights == str(tmp_path / "yolo26s.pt")


@pytest.mark.parametrize("arguments", [
    ["run", "--student", "m"], ["run", "--method", "generic-mse"],
    ["run", "--mode", "multiseed"], ["run", "--seed", "invalid"],
    ["run", "--wandb-mode", "automatic"], ["evaluate"],
])
def test_cli_invalid_choices_stop_at_argument_parsing(arguments):
    with pytest.raises(SystemExit) as captured:
        build_parser().parse_args(arguments)
    assert captured.value.code == 2


def test_cli_preflight_and_count_override_are_explicit():
    parser = build_parser()
    checked = parser.parse_args(["preflight", "--require-gpu", "--models", "m", "s", "n", "--native-probe", "s"])
    assert checked.require_gpu is True
    assert checked.models == ["m", "s", "n"]
    assert checked.native_probe == "s"
    audit = parser.parse_args(["audit", "--dataset-root", "/kaggle/input/InaTRC", "--ignore-expected-count"])
    assert audit.ignore_expected_count is True
    assert parser.parse_args(["summarize"]).include_smoke is False


def test_runner_is_valid_unexecuted_notebook_with_small_defaults(notebook):
    cells = _code(notebook)
    values = {name: ast.literal_eval(_assignment(cells, name)) for name in (
        "STAGE", "STUDENT", "METHOD", "SEED", "MODE", "WANDB_MODE", "OUTPUT_ROOT",
    )}
    assert values == {
        "STAGE": "student", "STUDENT": "n", "METHOD": "none", "SEED": 42,
        "MODE": "smoke", "WANDB_MODE": "disabled", "OUTPUT_ROOT": "/kaggle/working/outputs",
    }
    for index, source in enumerate(cells):
        compile(source, f"kaggle-cell-{index}", "exec")
    assert all(cell.get("execution_count") is None and not cell.get("outputs")
               for cell in notebook.cells if cell.cell_type == "code")


def test_runner_preserves_kaggle_torch_and_uses_explicit_uv_environment(notebook):
    cells = _code(notebook)
    sync = _strings(_assignment(cells, "sync"))
    assert "--frozen" in sync
    assert "--no-dev" in sync
    assert sync.count("--no-install-package") == 2
    assert "torch" in sync and "torchvision" in sync
    invoke = _strings(_assignment(cells, "UV_RUN"))
    assert invoke[:3] == ["run", "--frozen", "--no-sync"]
    assert "--python" in invoke
    assert invoke[-3:] == ["python", "-m", "inatrc_kd"]
    source = "\n".join(cells)
    assert "--system-site-packages" in source
    assert "UV_PYTHON_DOWNLOADS" in source and '"never"' in source
    assert "KERNEL_TORCH" in source and 'active_torch["torch_file"]' in source
    assert 'active_torch["torch_version"]' in source
    assert 'active_torch["torchvision_file"]' in source
    assert 'active_torch["torchvision_version"]' in source
    assert 'active_torch["python"]' in source
    assert "uv==0.9.8" in source
    assert "pip install torch" not in source


@pytest.mark.parametrize("version,accepted", [
    ((3, 10), False), ((3, 11), True), ((3, 12), True), ((3, 13), True), ((3, 14), False),
])
def test_runner_and_metadata_agree_on_python_support(notebook, version, accepted):
    import tomllib
    from packaging.specifiers import SpecifierSet

    supported = SpecifierSet(tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["requires-python"])
    locked = SpecifierSet(tomllib.loads((REPO_ROOT / "uv.lock").read_text())["requires-python"])
    assert supported.contains(".".join(map(str, version))) is accepted
    assert locked.contains(".".join(map(str, version))) is accepted
    assertion = next(node for source in _code(notebook) for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.Assert) and any(isinstance(part, ast.Attribute)
                         and part.attr == "version_info" for part in ast.walk(node)))
    check = compile(ast.Module(body=[assertion], type_ignores=[]), "runner-python-guard", "exec")
    if accepted:
        exec(check, {"sys": types.SimpleNamespace(version_info=version)})
    else:
        with pytest.raises(AssertionError):
            exec(check, {"sys": types.SimpleNamespace(version_info=version)})


def test_runner_launches_exactly_one_package_experiment(notebook):
    cells = _code(notebook)
    command = _assignment(cells, "command")
    assert isinstance(command, ast.BinOp) and isinstance(command.left, ast.Name) and command.left.id == "UV_RUN"
    arguments = _strings(command)
    assert arguments[0] == "run"
    assert all(option in arguments for option in ("--stage", "--student", "--method", "--seed", "--mode"))
    assert "WANDB_API_KEY" not in arguments
    launches = []
    for source in cells:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run" and node.args
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "command"):
                launches.append(node)
        for loop in (node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))):
            assert not any(isinstance(node, ast.Name) and node.id == "command" for node in ast.walk(loop))
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == "train" for node in ast.walk(tree))
    assert len(launches) == 1


def test_runner_has_source_provenance_full_guards_and_secret_isolation(notebook):
    cells = _code(notebook)
    source = "\n".join(cells)
    assert "source_sha256" in source and "uv_lock_sha256" in source and "INATRC_SOURCE_PROVENANCE" in source
    assert "[0-9a-fA-F]{40}" in source
    assert "SNAPSHOT_ROOT" in source and "UV_CACHE_INPUT" in source and "--offline" in source
    assert 'MODE == "full" and data_root is None' in source
    assert 'MODE == "full" and METHOD != "none" and not TEACHER_CKPT' in source
    assert "UserSecretsClient" in source
    for cell in cells:
        for node in ast.walk(ast.parse(cell)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                assert "WANDB_API_KEY" not in _strings(node)
                assert not any(isinstance(part, ast.Name) and part.id == "CHILD_ENV" for part in ast.walk(node))


def test_real_dataset_audit_does_not_become_smoke_training_subset():
    # Verify control flow statically rather than running any real-data training.
    tree = ast.parse((REPO_ROOT / "src" / "inatrc_kd" / "train.py").read_text(encoding="utf-8"))
    smoke = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Compare)
                 and any(isinstance(part, ast.Constant) and part.value == "smoke" for part in ast.walk(node.test))
                 and any(isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                         and part.func.id == "create_synthetic_dataset" for statement in node.body for part in ast.walk(statement)))
    assignments = [node for statement in smoke.body for node in ast.walk(statement) if isinstance(node, ast.Assign)]
    selected = next(node for node in assignments if any(isinstance(target, ast.Name) and target.id == "data_root"
                                                       for target in node.targets))
    assert isinstance(selected.value, ast.Call) and selected.value.func.id == "create_synthetic_dataset"
    synthetic_audit = next(node for statement in smoke.body for node in ast.walk(statement)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "audit_dataset")
    assert next(keyword.value.value for keyword in synthetic_audit.keywords if keyword.arg == "expected_images") == 20
    assert any(isinstance(part, ast.Constant) and part.value == "random_yaml_diagnostics_only"
               for statement in smoke.body for part in ast.walk(statement))
