"""Package penelitian InaTRC; alur eksperimen utama ada di train.py."""

import os
from pathlib import Path

# Simpan pengaturan library di output yang boleh ditulis, termasuk pada Kaggle.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "outputs" / ".ultralytics"))

__version__ = "0.1.0"
