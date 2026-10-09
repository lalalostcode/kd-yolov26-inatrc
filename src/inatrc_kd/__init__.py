"""InaTRC YOLO26 research pipelines, including isolated Stage B KD adaptations."""

import os
from pathlib import Path

# Keep library settings in writable experiment space, including on Kaggle.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "outputs" / ".ultralytics"))

__version__ = "0.1.0"
