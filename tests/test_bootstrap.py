"""Run notebook bootstrap helpers locally, without installing or contacting GitHub."""

import ast
import json
import subprocess
import sys

import nbformat
import pytest

from inatrc_kd.config import REPO_ROOT


@pytest.fixture
def bootstrap(tmp_path):
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb", as_version=4)
    source = next(cell.source for cell in notebook.cells
                  if cell.cell_type == "code" and "def prepare_git_source" in cell.source)
    # Load only reusable helpers, so platform setup and training never execute.
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    logs = tmp_path / "logs"
    logs.mkdir()
    namespace = {"LOG_DIR": logs, "BOOTSTRAP_READY": False, "_active_log": None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "runner-bootstrap", "exec"), namespace)
    return namespace


def git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True,
                                   stderr=subprocess.STDOUT).strip()


@pytest.fixture
def local_source(tmp_path):
    source = tmp_path / "origin-fixture"
    source.mkdir()
    git(source, "init", "--quiet")
    (source / ".gitignore").write_text(".venv/\noutputs/\n.uv-cache/\n", encoding="utf-8")
    (source / "research.py").write_text("version = 1\n", encoding="utf-8")
    git(source, "add", ".")
    git(source, "-c", "user.name=Notebook Test", "-c", "user.email=notebook@example.invalid",
        "commit", "--quiet", "-m", "Fixture initial source")
    first = git(source, "rev-parse", "HEAD")
    (source / "research.py").write_text("version = 2\n", encoding="utf-8")
    git(source, "add", "research.py")
    git(source, "-c", "user.name=Notebook Test", "-c", "user.email=notebook@example.invalid",
        "commit", "--quiet", "-m", "Fixture second source")
    return source, first, git(source, "rev-parse", "HEAD")


def prepare(bootstrap, repo, source, commit):
    with bootstrap["setup_stage"]("source"):
        return bootstrap["prepare_git_source"](repo, source.as_uri(), commit)


def record_commands(bootstrap):
    commands = []
    actual = bootstrap["run_logged"]

    def recording(command, **kwargs):
        commands.append(command)
        return actual(command, **kwargs)

    bootstrap["run_logged"] = recording
    return commands


def test_first_fetch_is_pinned_shallow_without_remote_and_reuse_does_not_fetch(
    bootstrap, local_source, tmp_path
):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    commands = record_commands(bootstrap)
    assert prepare(bootstrap, checkout, source, commit) == commit
    fetch = next(command for command in commands if command[1] == "fetch")
    assert fetch[fetch.index("--depth") + 1] == "1"
    assert "--no-tags" in fetch and fetch[-2:] == [source.as_uri(), commit]
    assert git(checkout, "rev-parse", "HEAD") == commit
    assert git(checkout, "rev-list", "--count", "HEAD") == "1"
    assert git(checkout, "remote") == ""
    assert (checkout / "research.py").read_text() == "version = 2\n"
    assert json.loads((checkout / ".git" / "inatrc_source.json").read_text()) == {
        "url": source.as_uri(), "commit": commit,
    }
    commands.clear()
    assert prepare(bootstrap, checkout, source, commit) == commit
    assert not any(command[1] in {"init", "fetch", "checkout"} for command in commands)


def test_initialized_repo_without_head_can_resume(bootstrap, local_source, tmp_path):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    assert prepare(bootstrap, checkout, source, commit) == commit
    assert git(checkout, "rev-list", "--count", "HEAD") == "1"


def test_interrupted_fetch_keeps_identity_and_retry_succeeds(bootstrap, local_source, tmp_path):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    actual = bootstrap["run_logged"]

    def interrupted(command, **kwargs):
        if command[1] == "fetch":
            raise RuntimeError("Simulated interrupted fetch")
        return actual(command, **kwargs)

    bootstrap["run_logged"] = interrupted
    with pytest.raises(RuntimeError, match="Tahap source gagal.*Simulated interrupted fetch"):
        prepare(bootstrap, checkout, source, commit)
    marker = checkout / ".git" / "inatrc_source.json"
    assert json.loads(marker.read_text()) == {"url": source.as_uri(), "commit": commit}
    assert list(checkout.iterdir()) == [checkout / ".git"]
    bootstrap["run_logged"] = actual
    assert prepare(bootstrap, checkout, source, commit) == commit
    log = (bootstrap["LOG_DIR"] / "source.log").read_text(encoding="utf-8")
    assert "GAGAL" in log and "Selesai" in log  # Reattempt appends, preserving failure diagnosis.


@pytest.mark.parametrize("difference", ["url", "commit"])
def test_different_recorded_source_is_rejected_without_overwrite(
    bootstrap, local_source, tmp_path, difference
):
    source, first, commit = local_source
    checkout = tmp_path / "checkout"
    prepare(bootstrap, checkout, source, commit)
    original = (checkout / "research.py").read_bytes()
    commands = record_commands(bootstrap)
    url = (tmp_path / "another-source").as_uri() if difference == "url" else source.as_uri()
    requested = first if difference == "commit" else commit
    with bootstrap["setup_stage"]("source"), pytest.raises(RuntimeError, match="URL/commit berbeda"):
        bootstrap["prepare_git_source"](checkout, url, requested)
    assert (checkout / "research.py").read_bytes() == original
    assert git(checkout, "rev-parse", "HEAD") == commit
    assert not commands


def test_existing_head_without_marker_must_match_requested_commit(bootstrap, local_source, tmp_path):
    source, first, commit = local_source
    checkout = tmp_path / "checkout"
    prepare(bootstrap, checkout, source, commit)
    (checkout / ".git" / "inatrc_source.json").unlink()
    with pytest.raises(RuntimeError, match="commit berbeda"):
        prepare(bootstrap, checkout, source, first)
    assert git(checkout, "rev-parse", "HEAD") == commit


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_non_git_source_is_rejected_without_deletion(bootstrap, local_source, tmp_path, kind):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    if kind == "directory":
        checkout.mkdir()
        sentinel = checkout / "user.txt"
    else:
        sentinel = checkout
    sentinel.write_text("Keep user content", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bukan checkout Git"):
        prepare(bootstrap, checkout, source, commit)
    assert sentinel.read_text() == "Keep user content"


def test_empty_git_with_user_file_does_not_fetch(bootstrap, local_source, tmp_path):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    sentinel = checkout / "user.txt"
    sentinel.write_text("Keep user content", encoding="utf-8")
    commands = record_commands(bootstrap)
    with pytest.raises(RuntimeError, match="tanpa commit memiliki file pengguna"):
        prepare(bootstrap, checkout, source, commit)
    assert sentinel.read_text() == "Keep user content"
    assert not any(command[1] == "fetch" for command in commands)


@pytest.mark.parametrize("kind", ["tracked", "untracked"])
def test_user_changes_are_rejected_and_preserved(bootstrap, local_source, tmp_path, kind):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    prepare(bootstrap, checkout, source, commit)
    target = checkout / ("research.py" if kind == "tracked" else "user_notes.txt")
    target.write_text("Keep this edit", encoding="utf-8")
    with pytest.raises(RuntimeError, match="perubahan/file pengguna"):
        prepare(bootstrap, checkout, source, commit)
    assert target.read_text() == "Keep this edit"


def test_ignored_venv_and_cache_can_be_reused(bootstrap, local_source, tmp_path):
    source, _, commit = local_source
    checkout = tmp_path / "checkout"
    prepare(bootstrap, checkout, source, commit)
    for directory in (".venv", ".uv-cache", "outputs"):
        path = checkout / directory
        path.mkdir()
        (path / "keep.txt").write_text("Keep cached state")
    commands = record_commands(bootstrap)
    assert prepare(bootstrap, checkout, source, commit) == commit
    assert not any(command[1] == "fetch" for command in commands)
    assert all((checkout / directory / "keep.txt").read_text() == "Keep cached state"
               for directory in (".venv", ".uv-cache", "outputs"))


@pytest.mark.parametrize("phase", ["source", "dependencies", "preflight"])
def test_setup_logs_phase_duration_and_stdout_stderr(bootstrap, capsys, phase):
    with bootstrap["setup_stage"](phase):
        print("Readable progress")
        print("Readable stderr", file=sys.stderr)
    output = capsys.readouterr()
    assert "Mulai" in output.out and "Selesai:" in output.out
    assert "detik" in output.out and "Readable stderr" in output.err
    log = (bootstrap["LOG_DIR"] / f"{phase}.log").read_text(encoding="utf-8")
    assert "Readable progress" in log and "Readable stderr" in log
    assert f"[{phase}]" in log and "Selesai:" in log


def test_failed_setup_clears_ready_reports_log_and_blocks_training(bootstrap):
    bootstrap["BOOTSTRAP_READY"] = True
    with pytest.raises(RuntimeError, match="Tahap preflight gagal.*CUDA ABI fixture.*Log lengkap") as captured:
        with bootstrap["setup_stage"]("preflight"):
            raise ValueError("CUDA ABI fixture")
    assert bootstrap["BOOTSTRAP_READY"] is False
    assert captured.value.__suppress_context__ is True
    log_path = bootstrap["LOG_DIR"] / "preflight.log"
    assert str(log_path) in str(captured.value)
    assert "GAGAL setelah" in log_path.read_text(encoding="utf-8")
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb", as_version=4)
    training = next(cell.source for cell in notebook.cells
                    if cell.cell_type == "code" and "command = UV_RUN" in cell.source)
    # Running just this cell after a failure stops at the guard, before building any command.
    with pytest.raises(AssertionError, match="preflight"):
        exec(compile(training, "runner-training-after-failure", "exec"), bootstrap)


@pytest.mark.parametrize("capture", [False, True])
def test_command_output_is_logged_and_secret_values_are_redacted(
    bootstrap, monkeypatch, capsys, capture
):
    monkeypatch.setenv("WANDB_API_KEY", "fixture-parent-secret")
    bootstrap["CHILD_ENV"] = {"WANDB_API_KEY": "fixture-child-secret"}
    command = [sys.executable, "-c",
               "import sys; print('fixture-parent-secret fixture-child-secret'); "
               "print('stderr progress', file=sys.stderr)"]
    with bootstrap["setup_stage"]("dependencies"):
        result = bootstrap["run_logged"](command, capture=capture)
    assert result.returncode == 0
    log = (bootstrap["LOG_DIR"] / "dependencies.log").read_text(encoding="utf-8")
    screen = capsys.readouterr()
    for output in (log, screen.out, screen.err, result.stdout):
        assert "fixture-parent-secret" not in output and "fixture-child-secret" not in output
    assert "[REDACTED] [REDACTED]" in log and "stderr progress" in log
    if capture:
        assert "[REDACTED] [REDACTED]" in result.stdout
        assert "stderr progress" in result.stdout
    else:
        assert "[REDACTED] [REDACTED]" in screen.out


def test_nonzero_command_stops_phase_and_keeps_failure_output(bootstrap, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "fixture-failure-secret")
    command = [sys.executable, "-c",
               "import sys; print('fixture-failure-secret failed operation'); sys.exit(17)"]
    with pytest.raises(RuntimeError, match="Tahap dependencies gagal.*exit 17.*Log lengkap") as captured:
        with bootstrap["setup_stage"]("dependencies"):
            bootstrap["run_logged"](command, capture=True)
    log = (bootstrap["LOG_DIR"] / "dependencies.log").read_text(encoding="utf-8")
    assert "[REDACTED] failed operation" in log and "GAGAL" in log
    assert "fixture-failure-secret" not in log + str(captured.value) + capsys.readouterr().out
    assert bootstrap["BOOTSTRAP_READY"] is False


def test_failed_phase_exception_cannot_leak_environment_secret(bootstrap, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "fixture-exception-secret")
    with pytest.raises(RuntimeError) as captured:
        with bootstrap["setup_stage"]("source"):
            raise ValueError("Bad credential fixture-exception-secret")
    log = (bootstrap["LOG_DIR"] / "source.log").read_text(encoding="utf-8")
    screen = capsys.readouterr()
    assert "fixture-exception-secret" not in log + str(captured.value) + screen.out + screen.err
    assert "Bad credential [REDACTED]" in str(captured.value)


def test_entire_source_cell_copies_snapshot_then_rejects_changed_source(tmp_path):
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb", as_version=4)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    namespace = {}
    exec(compile(code[0], "runner-source-config", "exec"), namespace)
    input_root = tmp_path / "input"
    snapshot = input_root / "snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (snapshot / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    namespace.update(WORK_ROOT=str(tmp_path), INPUT_ROOT=str(input_root),
                     OUTPUT_ROOT=str(tmp_path / "outputs"), SNAPSHOT_ROOT=str(snapshot))
    source = next(cell for cell in code if "def prepare_git_source" in cell)
    exec(compile(source, "runner-source-snapshot", "exec"), namespace)
    checkout = tmp_path / "inatrc-kd-yolov26"
    assert (checkout / "pyproject.toml").read_bytes() == (snapshot / "pyproject.toml").read_bytes()
    provenance = json.loads((tmp_path / "source_provenance.json").read_text())
    assert provenance["kind"] == "snapshot" and provenance["commit"] is None
    assert len(provenance["source_sha256"]) == 64 and len(provenance["uv_lock_sha256"]) == 64
    assert namespace["BOOTSTRAP_READY"] is False
    # Repeating the same source succeeds, preserving any ignored venv state.
    (checkout / ".venv").mkdir()
    marker = checkout / ".venv" / "keep.txt"
    marker.write_text("Keep venv")
    exec(compile(source, "runner-source-repeat", "exec"), namespace)
    assert marker.read_text() == "Keep venv"
    original = (checkout / "pyproject.toml").read_bytes()
    (snapshot / "pyproject.toml").write_text("Changed snapshot", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Tahap source gagal.*berbeda dari snapshot.*Log lengkap"):
        exec(compile(source, "runner-source-changed-snapshot", "exec"), namespace)
    assert (checkout / "pyproject.toml").read_bytes() == original
    assert marker.read_text() == "Keep venv"
    log = (tmp_path / "outputs" / "setup_logs" / "source.log").read_text(encoding="utf-8")
    assert "Selesai:" in log and "GAGAL setelah" in log


def preflight_namespace(bootstrap, tmp_path):
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "01_kaggle_runner.ipynb", as_version=4)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    exec(compile(code[0], "runner-preflight-config", "exec"), bootstrap)
    source = next(cell for cell in code if "COLAB SECRET BEGIN" in cell)
    environment = {
        "torch_file": str((tmp_path / "torch" / "__init__.py").resolve()),
        "torch_version": "2.10.0+cu128",
        "torchvision_file": str((tmp_path / "torchvision" / "__init__.py").resolve()),
        "torchvision_version": "0.25.0+cu128",
        "python": "3.13.9",
    }
    bootstrap.update(INPUT=tmp_path, REPO=tmp_path / "checkout", CHILD_ENV={},
                     UV_RUN=["diagnostic-fixture"], KERNEL_TORCH=dict(environment))
    calls = []

    def successful(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"environment": environment}))

    bootstrap["run_logged"] = successful
    return source, environment, calls


def test_preflight_success_sets_ready_after_matching_kernel_environment(bootstrap, tmp_path):
    source, _, calls = preflight_namespace(bootstrap, tmp_path)
    exec(compile(source, "runner-preflight-success", "exec"), bootstrap)
    assert bootstrap["BOOTSTRAP_READY"] is True
    assert bootstrap["data_root"] is None
    assert len(calls) == 1 and "--require-gpu" in calls[0]
    assert "Selesai:" in (bootstrap["LOG_DIR"] / "preflight.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["torch_file", "torch_version", "torchvision_file", "torchvision_version", "python"])
def test_preflight_kernel_mismatch_stops_before_training(bootstrap, tmp_path, field):
    source, environment, _ = preflight_namespace(bootstrap, tmp_path)
    environment[field] = "different-environment"
    bootstrap["BOOTSTRAP_READY"] = True
    with pytest.raises(RuntimeError, match="Tahap preflight gagal.*berbeda.*Log lengkap"):
        exec(compile(source, "runner-preflight-mismatch", "exec"), bootstrap)
    assert bootstrap["BOOTSTRAP_READY"] is False


def test_full_kd_requires_teacher_before_preflight_subprocess(bootstrap, tmp_path):
    source, _, calls = preflight_namespace(bootstrap, tmp_path)
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (dataset / split / kind).mkdir(parents=True)
    bootstrap.update(MODE="full", METHOD="native", DATASET_ROOT=str(dataset), TEACHER_CKPT="")
    with pytest.raises(RuntimeError, match="Tahap preflight gagal.*checkpoint teacher.*Log lengkap"):
        exec(compile(source, "runner-preflight-full-kd", "exec"), bootstrap)
    assert calls == [] and bootstrap["BOOTSTRAP_READY"] is False
