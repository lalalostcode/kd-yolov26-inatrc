"""Pip dan Secrets dimock; pengujian tidak memasang/mengganti dependency."""

import ast
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "standalone_runtime.py"
spec = importlib.util.spec_from_file_location("standalone_runtime_fixture", SOURCE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
VALIDATE_LOADED = runtime.validate_loaded_packages


@pytest.fixture
def pip_fixture(monkeypatch):
    """Metadata pip disimulasikan tanpa menyentuh environment pengujian."""
    versions = {
        "torch": "2.7.0+cu128", "torchvision": "0.22.0+cu128",
        "ultralytics": "8.4.100", "wandb": "0.28.1", "faster-coco-eval": "1.6.7",
    }
    plan = [{"metadata": {"name": "ultralytics", "version": "8.4.155"}}]
    calls = []

    def snapshot(name):
        return {"version": versions[name.lower()], "location": "/kernel/site-packages"}

    def command(arguments, log_path):
        calls.append(arguments)
        if "--dry-run" in arguments:
            report = arguments[arguments.index("--report") + 1]
            Path(report).write_text(json.dumps({"install": plan}), encoding="utf-8")
        else:
            for pin in arguments:
                if "==" in pin:
                    name, version = pin.split("==")
                    versions[name.lower()] = version
        return "pip simulated successfully\n"

    monkeypatch.setattr(runtime, "_setup_package_snapshot", snapshot)
    monkeypatch.setattr(runtime, "_setup_run_command", command)
    monkeypatch.setattr(runtime, "validate_loaded_packages", lambda packages: None)
    monkeypatch.setattr(runtime, "_setup_validate_requirements", lambda packages: [])
    return SimpleNamespace(versions=versions, plan=plan, calls=calls)


def test_runtime_top_level_is_standard_library_and_only_definitions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for statement in tree.body:
        assert isinstance(statement, (ast.Expr, ast.Import, ast.ImportFrom, ast.FunctionDef))
        if isinstance(statement, ast.Import):
            assert all(alias.name.split(".")[0] in sys.stdlib_module_names for alias in statement.names)
    forbidden = {"exec", "eval", "__import__"}
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in forbidden for node in ast.walk(tree))


def test_pip_resolves_then_installs_only_non_torch_pins(pip_fixture, tmp_path):
    pip_fixture.plan.append({"metadata": {"name": "some-transitive", "version": "2.3.1"}})
    report = runtime.setup_dependencies(["ultralytics==8.4.155", "wandb==0.28.1"], tmp_path)
    assert report["torch_before"] == report["torch_after"]
    assert len(pip_fixture.calls) == 2
    resolve, install = pip_fixture.calls
    assert resolve[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--dry-run" in resolve
    assert "--constraint" in resolve
    assert "--only-binary=faster-coco-eval" in resolve
    assert "--no-deps" in install
    assert "some-transitive==2.3.1" in install
    assert not any(pin.startswith(("torch==", "torchvision==")) for pin in install)
    protected = (tmp_path / "setup_logs" / "protected_torch.txt").read_text()
    assert protected == "torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\n"
    assert Path(report["log_path"]).is_file()
    assert Path(report["report_path"]).is_file()


@pytest.mark.parametrize("package", ["torch", "Torch", "torchvision"])
def test_pip_plan_rejects_torch_wheels(pip_fixture, tmp_path, package):
    pip_fixture.plan[:] = [{"metadata": {"name": package, "version": "2.8.0"}}]
    with pytest.raises(RuntimeError, match="bawaan tidak boleh diubah"):
        runtime.setup_dependencies(["ultralytics==8.4.155"], tmp_path)
    assert len(pip_fixture.calls) == 1


def test_existing_dependencies_skip_application_install(pip_fixture, tmp_path):
    pip_fixture.plan.clear()
    pip_fixture.versions["ultralytics"] = "8.4.155"
    runtime.setup_dependencies(["ultralytics==8.4.155"], tmp_path)
    assert len(pip_fixture.calls) == 1


def test_torch_snapshot_change_stops_setup(pip_fixture, monkeypatch, tmp_path):
    original = runtime._setup_run_command

    def changing_command(arguments, log_path):
        result = original(arguments, log_path)
        if "--no-deps" in arguments:
            pip_fixture.versions["torch"] = "2.8.0+cu128"
        return result

    monkeypatch.setattr(runtime, "_setup_run_command", changing_command)
    with pytest.raises(RuntimeError, match="Torch berubah"):
        runtime.setup_dependencies(["ultralytics==8.4.155"], tmp_path)


@pytest.mark.parametrize("declaration", ["torch==2.7.0", "torchvision==0.22.0", "numpy>=2", "https://example.invalid/a.whl", "--upgrade", "bad"])
def test_input_dependency_pins_reject_wheels_and_unpinned_values(declaration):
    with pytest.raises(ValueError):
        runtime._setup_pinned_requirements([declaration])


def test_duplicate_requirements_must_agree():
    assert len(runtime._setup_pinned_requirements(["PyYAML==6.0.3", "pyyaml==6.0.3"])) == 1
    with pytest.raises(ValueError, match="ganda"):
        runtime._setup_pinned_requirements(["PyYAML==6.0.3", "pyyaml==6.0.2"])


def test_setup_failure_secret_is_redacted(pip_fixture, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-private-test-key")

    def failing_command(*args):
        raise RuntimeError("Network failed for wandb-private-test-key")

    monkeypatch.setattr(runtime, "_setup_run_command", failing_command)
    with pytest.raises(RuntimeError) as error:
        runtime.setup_dependencies(["wandb==0.28.1"], tmp_path)
    log = (tmp_path / "setup_logs" / "pip_setup.log").read_text()
    assert "wandb-private-test-key" not in str(error.value) + log + capsys.readouterr().out
    assert "[REDACTED]" in log
    assert "Instalasi Dependency" in str(error.value)
    assert "pip_setup.log" in str(error.value)


def test_streaming_command_logs_output_and_redacts_secret(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "very-secret-key")
    captured = []

    def popen(arguments, **kwargs):
        captured.append((arguments, kwargs))
        return SimpleNamespace(stdout=io.StringIO("Download very-secret-key\nFinished\n"), wait=lambda: 0)

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    log = tmp_path / "pip.log"
    result = runtime._setup_run_command([sys.executable, "-m", "pip", "install"], log)
    assert "very-secret-key" not in result + log.read_text() + capsys.readouterr().out
    assert "[REDACTED]" in result
    assert captured[0][1]["stderr"] == runtime.subprocess.STDOUT


def test_streaming_command_failure_has_exit_code_and_log(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(
        stdout=io.StringIO("resolver conflict\n"), wait=lambda: 7,
    ))
    with pytest.raises(RuntimeError, match="exit 7.*resolver conflict"):
        runtime._setup_run_command(["not-run"], tmp_path / "pip.log")


@pytest.mark.parametrize("module_name,package", [("numpy", "numpy"), ("PIL", "Pillow"), ("yaml", "PyYAML"), ("thop", "ultralytics-thop")])
def test_stale_loaded_package_requires_restart(monkeypatch, module_name, package):
    monkeypatch.setitem(runtime.sys.modules, module_name, SimpleNamespace(__version__="1.0", __file__="/site/pkg/__init__.py"))
    monkeypatch.setattr(runtime, "_setup_package_snapshot", lambda name: {"version": "2.0", "location": "/site"})
    with pytest.raises(RuntimeError, match="Restart Session.*Restart runtime"):
        runtime.validate_loaded_packages([package])


def test_loaded_package_from_different_location_requires_restart(monkeypatch):
    monkeypatch.setitem(runtime.sys.modules, "numpy", SimpleNamespace(__version__="2.0", __file__="/old/numpy/__init__.py"))
    monkeypatch.setattr(runtime, "_setup_package_snapshot", lambda name: {"version": "2.0", "location": "/new"})
    with pytest.raises(RuntimeError, match="Restart Session"):
        runtime.validate_loaded_packages(["numpy"])


def test_cv2_build_number_difference_is_expected(monkeypatch):
    monkeypatch.setitem(runtime.sys.modules, "cv2", SimpleNamespace(__version__="4.13.0", __file__="/site/cv2/__init__.py"))
    monkeypatch.setattr(runtime, "_setup_package_snapshot", lambda name: {"version": "4.13.0.92", "location": "/site"})
    runtime.validate_loaded_packages(["opencv-python"])


@pytest.mark.parametrize("active_version", ["4.12.0", "3.13.0", "5.0.0"])
@pytest.mark.parametrize("package", ["opencv-python", "opencv-python-headless"])
def test_cv2_stale_core_version_requires_restart(monkeypatch, active_version, package):
    monkeypatch.setitem(runtime.sys.modules, "cv2", SimpleNamespace(__version__=active_version, __file__="/site/cv2/__init__.py"))
    monkeypatch.setattr(runtime, "_setup_package_snapshot", lambda name: {"version": "4.13.0.92", "location": "/site"})
    with pytest.raises(RuntimeError, match="Restart Session"):
        runtime.validate_loaded_packages([package])


def test_cv2_replaced_by_pip_requires_restart(pip_fixture, monkeypatch, tmp_path):
    pip_fixture.versions["opencv-python"] = "4.12.0.88"
    pip_fixture.plan[:] = [{"metadata": {"name": "opencv-python", "version": "4.13.0.92"}}]
    monkeypatch.setitem(runtime.sys.modules, "cv2", SimpleNamespace(
        __version__="4.12.0", __file__="/kernel/site-packages/cv2/__init__.py",
    ))
    monkeypatch.setattr(runtime, "validate_loaded_packages", lambda packages: VALIDATE_LOADED([
        name for name in packages if name == "opencv-python"
    ]))
    with pytest.raises(RuntimeError, match="Restart Session"):
        runtime.setup_dependencies(["opencv-python==4.13.0.92"], tmp_path)
    assert len(pip_fixture.calls) == 2
    assert pip_fixture.versions["opencv-python"] == "4.13.0.92"
    assert runtime.sys.modules["cv2"].__version__ == "4.12.0"


def test_loaded_numpy_matches_current_metadata(monkeypatch):
    monkeypatch.setitem(runtime.sys.modules, "numpy", SimpleNamespace(__version__="2.5.3", __file__="/site/numpy/__init__.py"))
    monkeypatch.setattr(runtime, "_setup_package_snapshot", lambda name: {"version": "2.5.3", "location": "/site"})
    runtime.validate_loaded_packages(["numpy"])


def test_numpy_replaced_by_pip_requires_restart_after_successful_install(pip_fixture, monkeypatch, tmp_path):
    pip_fixture.versions["numpy"] = "2.5.2"
    pip_fixture.plan[:] = [{"metadata": {"name": "numpy", "version": "2.5.3"}}]
    monkeypatch.setitem(runtime.sys.modules, "numpy", SimpleNamespace(
        __version__="2.5.2", __file__="/kernel/site-packages/numpy/__init__.py",
    ))
    # Jalankan validasi NumPy sungguhan dengan metadata pip simulasi.
    monkeypatch.setattr(runtime, "validate_loaded_packages", lambda packages: VALIDATE_LOADED([
        name for name in packages if name == "numpy"
    ]))
    with pytest.raises(RuntimeError, match="Restart Session"):
        runtime.setup_dependencies(["numpy==2.5.3"], tmp_path)
    assert len(pip_fixture.calls) == 2
    assert pip_fixture.versions["numpy"] == "2.5.3"
    assert runtime.sys.modules["numpy"].__version__ == "2.5.2"


def test_application_requirements_check_active_markers_and_versions(monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "requires", lambda name: [
        "numpy>=2,<3", "optional-tool; extra == 'dev'", "windows-only; sys_platform == 'never'",
    ])
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "2.5.3")
    checked = runtime._setup_validate_requirements(["ultralytics"])
    assert len(checked) == 1
    assert checked[0]["requirement"] == "numpy>=2,<3"


def test_application_requirements_reject_conflicts(monkeypatch):
    monkeypatch.setattr(runtime.importlib.metadata, "requires", lambda name: ["numpy<2"])
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "2.5.3")
    with pytest.raises(RuntimeError, match="tidak sesuai"):
        runtime._setup_validate_requirements(["ultralytics"])


@pytest.mark.parametrize("name", ["kaggle", "colab", "local"])
def test_detect_platform_paths(monkeypatch, tmp_path, name):
    monkeypatch.setattr(runtime.Path, "is_dir", lambda path: (
        str(path).replace("\\", "/") == "/kaggle/working" if name == "kaggle" else
        str(path).replace("\\", "/") == "/content" if name == "colab" else False
    ))
    monkeypatch.delitem(runtime.sys.modules, "google.colab", raising=False)
    result = runtime.detect_platform()
    assert result["name"] == name
    if name == "kaggle":
        assert result["input_root"] == "/kaggle/input"
        assert result["output_root"] == "/kaggle/working/outputs"
    elif name == "colab":
        assert result["output_root"] == "/content/outputs"
    else:
        assert result["input_root"] is None


@pytest.mark.parametrize("mode", ["disabled", "offline"])
def test_nononline_secrets_do_not_read_provider(monkeypatch, mode):
    monkeypatch.setenv("WANDB_API_KEY", "keep-out-of-training")
    monkeypatch.setitem(runtime.sys.modules, "kaggle_secrets", SimpleNamespace(
        UserSecretsClient=lambda: pytest.fail("Secrets provider must not be read"),
    ))
    assert runtime.get_wandb_secret(mode, "kaggle") is None
    assert "WANDB_API_KEY" not in runtime.os.environ


def test_online_environment_key_has_priority(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "already-present")
    monkeypatch.setitem(runtime.sys.modules, "kaggle_secrets", SimpleNamespace(
        UserSecretsClient=lambda: pytest.fail("Environment takes precedence"),
    ))
    assert runtime.get_wandb_secret("online", "kaggle") is None
    assert runtime.os.environ["WANDB_API_KEY"] == "already-present"


@pytest.mark.parametrize("platform", ["kaggle", "colab"])
def test_online_platform_secret_is_not_returned_or_printed(monkeypatch, capsys, platform):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    calls = []

    def secret(name):
        calls.append(name)
        return "test-private-platform-key"

    if platform == "kaggle":
        monkeypatch.setitem(runtime.sys.modules, "kaggle_secrets", SimpleNamespace(
            UserSecretsClient=lambda: SimpleNamespace(get_secret=secret),
        ))
    else:
        monkeypatch.setitem(runtime.sys.modules, "google.colab", SimpleNamespace(userdata=SimpleNamespace(get=secret)))
    assert runtime.get_wandb_secret("online", {"name": platform}) is None
    assert runtime.os.environ["WANDB_API_KEY"] == "test-private-platform-key"
    assert calls == ["WANDB_API_KEY"]
    assert "test-private-platform-key" not in capsys.readouterr().out


def test_secret_provider_error_is_sanitized(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def failed_secret(name):
        raise ValueError("provider contained private-unexpected-key")

    monkeypatch.setitem(runtime.sys.modules, "kaggle_secrets", SimpleNamespace(
        UserSecretsClient=lambda: SimpleNamespace(get_secret=failed_secret),
    ))
    with pytest.raises(RuntimeError) as error:
        runtime.get_wandb_secret("online", "kaggle")
    assert "private-unexpected-key" not in str(error.value)
    assert error.value.__suppress_context__


def dataset_layout(root):
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (root / split / kind).mkdir(parents=True)
    return root


def test_dataset_explicit_and_unique_autodiscovery(tmp_path):
    root = dataset_layout(tmp_path / "inputs" / "inatrc")
    assert runtime.discover_dataset_root(str(root), None, "full") == str(root.resolve())
    assert runtime.discover_dataset_root("", tmp_path / "inputs", "full") == str(root.resolve())


def test_dataset_ambiguous_autodiscovery_requires_explicit_path(tmp_path):
    dataset_layout(tmp_path / "first")
    dataset_layout(tmp_path / "second")
    with pytest.raises(RuntimeError, match="Lebih dari satu"):
        runtime.discover_dataset_root("", tmp_path, "smoke")


def test_missing_dataset_is_allowed_only_for_smoke(tmp_path):
    assert runtime.discover_dataset_root("", tmp_path, "smoke") is None
    with pytest.raises(RuntimeError, match="belum ditemukan"):
        runtime.discover_dataset_root("", tmp_path, "full")
    with pytest.raises(RuntimeError, match="train/val/test"):
        runtime.discover_dataset_root(tmp_path, None, "smoke")


def test_all_three_dataset_splits_are_required(tmp_path):
    root = dataset_layout(tmp_path)
    (root / "test" / "labels").rmdir()
    with pytest.raises(RuntimeError, match="train/val/test"):
        runtime.discover_dataset_root(root, None, "full")


def test_results_zip_is_unique_and_excludes_secrets_data_teacher(tmp_path, monkeypatch):
    run = tmp_path / "run-one"
    for relative in ["weights/best.pt", "weights/last.pt", "metrics.json", "results.csv",
                     "epoch_history.csv", "checkpoint_manifest.csv", "val/PR_curve.png",
                     "validation/val_batch0_pred.jpg", "best-val/val_batch1_pred.jpg", "reports/coco_area.json",
                     "synthetic_data/train/images/a.png", "diagnostic_teacher/weights/best.pt",
                     "wandb/debug.log", ".env", "anything.py"]:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("WANDB_API_KEY", "zip-private-key")
    first = runtime.create_results_zip(run)
    second = runtime.create_results_zip(run)
    assert first != second
    with zipfile.ZipFile(first) as archive:
        entries = {Path(name).as_posix() for name in archive.namelist()}
    assert "run-one/weights/best.pt" in entries
    assert "run-one/weights/last.pt" in entries
    assert "run-one/epoch_history.csv" in entries
    assert "run-one/checkpoint_manifest.csv" in entries
    assert "run-one/val/PR_curve.png" in entries
    assert "run-one/validation/val_batch0_pred.jpg" in entries
    assert "run-one/best-val/val_batch1_pred.jpg" in entries
    assert "run-one/reports/coco_area.json" in entries
    assert not any("synthetic_data" in entry or "diagnostic_teacher" in entry or ".env" in entry
                   or "wandb" in entry or "anything.py" in entry for entry in entries)


@pytest.mark.parametrize("content", ['{"WANDB_API_KEY": "forbidden"}', '{"private": "zip-private-key"}'])
def test_results_zip_rejects_credential_before_writing_archive(tmp_path, monkeypatch, content):
    monkeypatch.setenv("WANDB_API_KEY", "zip-private-key")
    run = tmp_path / "run-one"
    run.mkdir()
    (run / "metadata.json").write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match="credential"):
        runtime.create_results_zip(run)
    assert not list(tmp_path.glob("*.zip"))


def test_empty_result_bundle_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="Tidak ada artifact"):
        runtime.create_results_zip(tmp_path)
