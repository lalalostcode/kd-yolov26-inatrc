# Kompatibilitas Python 3.13

Kernel Kaggle pengguna melaporkan Python 3.13. Batas awal proyek berhenti pada
3.12, sehingga runner berhenti sebelum setup. Dukungan kini `>=3.11,<3.14`;
default lokal tetap Python 3.12. Ultralytics 8.4.155, W&B 0.28.1, dan pasangan
CPU Torch 2.10.0/torchvision 0.25.0 tetap dikunci.

## Strategi runtime CUDA

- Venv memakai interpreter kernel yang sama dengan `--system-site-packages`.
- Frozen sync melewati pemasangan Torch dan torchvision; pemanggilan setelahnya
  selalu `uv run --frozen --no-sync`.
- Path dan versi Torch/torchvision serta versi Python kernel dan subprocess
  dibandingkan sebelum training. Tidak ada penggantian wheel CUDA otomatis.
- Python 3.13 memerlukan Torch ≥2.7 dengan pasangan torchvision ≥0.22 yang
  sesuai. Pasangan 2.4–2.6 hanya diterima pada Python 3.11/3.12. Nightly dan
  prerelease gagal eksplisit.
- Preflight memeriksa dependency aktif melalui Requires-Dist, operasi tensor,
  torchvision NMS, dan konversi NumPy → Torch → device → NumPy. Jadi ABI NumPy
  yang tidak kompatibel juga menghentikan run.

Sumber: [tabel resmi torchvision](https://github.com/pytorch/vision#installation),
[metadata Ultralytics 8.4.155](https://github.com/ultralytics/ultralytics/blob/v8.4.155/pyproject.toml),
[W&B 0.28.1](https://pypi.org/project/wandb/0.28.1/), dan
[panduan uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Verifikasi yang dijalankan

Environment Python 3.13 dibuat terpisah di `outputs/python313-venv`, tanpa
mengganti venv Python 3.12. Instalasi frozen berhasil pada Windows dengan
Python **3.13.9**, Torch **2.10.0+cpu**, torchvision **0.25.0+cpu**, NumPy
**2.5.3**, Ultralytics **8.4.155**, dan W&B **0.28.1**.

```powershell
uv lock --cache-dir .uv-cache
uv lock --check --offline --cache-dir .uv-cache

$env:UV_PROJECT_ENVIRONMENT = 'outputs/python313-venv'
uv sync --frozen --python 3.13 --no-python-downloads --cache-dir .uv-cache
uv run --frozen --no-sync --python 3.13 --no-python-downloads --cache-dir .uv-cache pytest -q --basetemp outputs/pytest-python313-01
uv run --frozen --no-sync --python 3.13 --no-python-downloads --cache-dir .uv-cache python -m inatrc_kd preflight --device cpu --output-root outputs/python313-preflight
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

- Resolver menghasilkan lock baru berisi **73 packages**; lock freshness lulus.
- Python 3.13: **173 passed, 3 skipped**, 107.84 detik. Tiga skip memerlukan CUDA.
- Regresi Python 3.12.12: **173 passed, 3 skipped**, 92.22 detik, setelah frozen
  sync venv lokal dengan lock baru. Perintah: `uv run --frozen --no-sync --offline
  --cache-dir .uv-cache pytest -q --basetemp outputs/pytest-python312-compat-01`.
- Delapan smoke satu epoch: s/n × none/Native/CrossKD/CSAKD. Semua selesai,
  menghasilkan `best.pt`, melakukan 5 update optimizer, serta memuat ulang
  checkpoint dan memvalidasinya tanpa teacher. Semua teacher KD tetap frozen.
- CLI preflight CPU lulus, termasuk tensor, NMS, dan NumPy/Torch bridge.
- Frozen sync dry run Linux x86_64/Python 3.13, tanpa Torch/torchvision dan tanpa
  dev dependency, lulus: rencana memasang **52 packages**. Ini pemeriksaan
  rencana instalasi lintas platform, bukan instalasi/eksekusi Linux sebenarnya.

Log lokal berada di `outputs/python313-pytest.log`,
`outputs/python313-preflight/preflight.json`, dan
`outputs/python313-linux-sync-plan.log`; log regresi 3.12 berada di
`outputs/python312-compat-pytest.log`. Full training, multiseed, test-set
evaluation, dan W&B online tidak dijalankan. Tidak ada git push oleh Codex.

## Penggunaan ulang di Kaggle

1. Gunakan notebook terbaru dan perbarui source repository lengkap. Perubahan
   diperlukan pada `pyproject.toml`, `uv.lock`, dan `src/inatrc_kd/preflight.py`;
   mengganti batas Python pada cell lama saja tidak cukup.
2. Jika clone GitHub, push perubahan sendiri lalu isi `REPO_COMMIT` dengan SHA
   baru. Jika memakai snapshot Input, ganti snapshot dengan versi terbaru.
3. Mulai sesi baru, aktifkan GPU/Internet, dan jalankan default `MODE="smoke"`.
   Versi/path paket, CUDA, GPU, dan hasil probe tercatat pada preflight.

Runtime CUDA Kaggle/Colab asli belum dijalankan di sini. Torch/torchvision
pengguna belum diketahui saat verifikasi lokal; kompatibilitas runtime tersebut
tetap harus lolos preflight dan smoke GPU. Cache offline Python 3.12 tidak
dianggap cukup untuk kernel 3.13; siapkan cache Linux sesuai interpreter baru.
