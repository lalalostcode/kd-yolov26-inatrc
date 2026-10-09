"""Small file helpers shared by the experiment commands."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def write_json(path: str | Path, value: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(root: Path) -> dict:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    digest = hashlib.sha256()
    for folder in ("src", "configs", "notebooks"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(bytes.fromhex(sha256(path)))
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(bytes.fromhex(sha256(path)))
    return {"git_commit": commit, "git_dirty": bool(dirty) if dirty is not None else None,
            "source_sha256": digest.hexdigest(),
            "uv_lock_sha256": sha256(root / "uv.lock") if (root / "uv.lock").exists() else None}
