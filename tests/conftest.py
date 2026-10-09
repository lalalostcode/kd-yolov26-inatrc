"""All test caches/settings live in ignored writable workspace output."""
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "outputs" / "test-runtime"
for name, variable in (("ultralytics", "YOLO_CONFIG_DIR"), ("matplotlib", "MPLCONFIGDIR")):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)


def pytest_sessionstart(session):
    import torch
    torch.set_num_threads(4)
