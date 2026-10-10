# Benchmark COCO8 diagnostik

Tambahan terisolasi pada sembilan notebook mandiri, 10 Oktober 2026. Fokusnya satu epoch dari pretrained sampai checkpoint/validation; hasil diberi label **benchmark diagnostic / nonfinal**. Tidak mengukur manfaat KD atau kualitas InaTRC.

## Perubahan minimum

- Tambahkan `BENCHMARK_TEST = False` pada konfigurasi. Ubah menjadi True untuk benchmark. Generator tetap menghasilkan sembilan notebook yang sama dengan 16 code cell; tidak ada notebook/framework/bootstrap/dependency manager baru.
- Benchmark memakai [COCO8 resmi Ultralytics](https://docs.ultralytics.com/datasets/detect/coco8/): `coco8.yaml`, 80 kelas, 4 train/4 val, tanpa test. Loader/validator standar Ultralytics menggantikan loader khusus InaTRC **hanya pada cabang benchmark**.
- Gunakan pretrained `yolo26m/s/n.pt`, seed 42, satu epoch, 320 px, batch 2, workers 0. Diagnostik memakai AMP off, warmup 0, `nbs=2`, dan plots off. Parameter full InaTRC tidak berubah. Tidak ada audit/fingerprint/hash gambar, COCO area, FLOPs/latency tambahan, atau ZIP pada benchmark.
- `BenchmarkTrainer` mewarisi trainer Native/baseline atau `CustomKDTrainer` yang sudah ada. Loss deteksi, Native `distill_model`/`dis`, CrossKD/CSAKD, feature alignment, freezing, dan optimizer tetap memakai implementasi yang sama. Hook sebelum step hanya mengamati gradient; step asli masih dipanggil. Proteksi OOM/nonfinite tetap mencegah retry/fallback diam-diam.
- `check_teacher` tetap lima kelas secara default; cabang diagnostik meminta 80 kelas secara eksplisit. Native dan custom KD memeriksa checkpoint teacher hasil notebook 01 dengan tag COCO8, satu epoch/optimizer update, seed 42, dan nama kelas yang sesuai. Pretrained mentah, teacher sintetis, atau teacher InaTRC tidak dapat menggantikan teacher benchmark.
- Pakai logger epoch/manifest yang sudah ada, W&B disabled. Tidak menambah sistem tracking. Output minimum memakai `best.pt`, `last.pt`, `results.csv`, CSV epoch/manifest yang ada, args/resolved config, dan `benchmark_summary.json`; pointer run terakhir adalah `benchmark_latest.json`. Folder run unik melindungi hasil lama.
- Setup tetap pip interpreter kernel, mempertahankan Torch/torchvision bawaan dan pemeriksaan ABI/CUDA. Benchmark menonaktifkan dependency COCO area/Secrets W&B. Kaggle/Colab memerlukan device GPU; CPU hanya pilihan eksplisit untuk test lokal.

Audit awal menemukan pembatas 3.250 gambar, lima kelas, train/val/test, loader aman, dan evaluator lima kelas pada jalur InaTRC. Fungsi tersebut **tidak dihapus atau digeneralisasi**. Cabang benchmark melewatinya. Snapshot sebelum/sesudah membuktikan seluruh preset training/evaluasi/dataset/KD/tracking InaTRC identik; AST `run_experiment`, audit/loader, logger, evaluator, dan seluruh kelas/persamaan KD identik. Perubahan bersama hanya perluasan guard teacher yang dipilih secara eksplisit untuk benchmark.

## Cara menjalankan sembilan kondisi di Kaggle

Gunakan [tabel sembilan notebook pada README](../README.md#notebook-mandiri-untuk-kaggle-dan-colab). Upload file `.ipynb` langsung, aktifkan GPU/Internet. Source repository, clone GitHub, uv, dan venv tidak diperlukan.

Pada notebook 01:

```python
BENCHMARK_TEST = True
DEVICE = "0"
SEED = 42
```

Run All melatih teacher YOLO26m pada COCO8 satu epoch. `MODE`, `DATASET_ROOT`, dan jumlah InaTRC diabaikan. Download resmi dataset/pretrained terjadi jika belum tersedia; `MODEL_WEIGHTS` dapat menunjuk pretrained COCO `.pt` lokal sesuai ukuran bila dilampirkan sebagai Input. Jika setup meminta restart karena paket sudah di-import dengan versi lain, restart session lalu Run All.

Simpan hasil teacher dengan **Save Version**. Lampirkan `weights/best.pt` sebagai Input pada semua notebook KD (04–09), lalu isi:

```python
BENCHMARK_TEST = True
DEVICE = "0"
SEED = 42
TEACHER_CKPT = "/kaggle/input/teacher-coco8/.../weights/best.pt"
TEACHER_SHA256 = ""  # opsional: isi hash yang sama dari benchmark_summary.json teacher
```

Gunakan checkpoint teacher yang sama pada Native s/n, CrossKD s/n, dan CSAKD s/n. Notebook 02/03 adalah baseline s/n; tidak perlu teacher. Masing-masing Run All hanya menjalankan satu kondisi, tanpa otomatis mengulang kondisi gagal. CrossKD/CSAKD tetap konfigurasi pilot.

Di akhir terlihat `PASS (cuda:0)` beserta best epoch jika seluruh pemeriksaan berhasil. Jika exception muncul, run adalah FAIL; baca pesan dan `benchmark_summary.json` pada folder run tersebut. mAP rendah tetap dapat PASS. Simpan folder `/kaggle/working/outputs` melalui Save Version. Tidak diperlukan API key W&B atau ZIP pada benchmark.

Untuk InaTRC, ubah flag False. Protokol full/smoke, lima kelas, tiga split, checkpoint teacher penelitian, opsi metrik, dan W&B kembali mengikuti konfigurasi lama. Tidak ada teacher COCO8 yang diterima pada KD InaTRC. Colab memakai pilihan yang sama, GPU runtime, dan path checkpoint; blok Drive tetap opsional/dikomentari.

## Kriteria PASS yang diperiksa kode

Satu epoch harus selesai, loader memuat 4 train/4 val dengan head 80 kelas, loss finite, backward menghasilkan gradient student/adapter finite dan nonzero, adapter masuk optimizer, serta minimal satu optimizer update terbukti. KD harus menghasilkan `dis_loss` nonzero dengan komponen metode asli, teacher frozen/eval tanpa gradient dan state tidak berubah. `best.pt` dimuat ulang sebagai detector tanpa teacher/projector, kemudian validation val COCO8 harus berhasil dengan metrik finite.

Tag checkpoint dan status PASS ditulis setelah validasi berhasil. Raw pretrained tidak diberi tag teacher terlatih. Observer gradient tidak mengubah loss/gradient/learning rate. Tidak ada loss dummy atau kondisi KD yang diam-diam berubah menjadi baseline.

## Hasil aktual

Environment lokal: Windows, Python 3.12.12, Torch 2.10.0 **CPU**, Ultralytics 8.4.155. COCO8 dan pretrained asli diunduh dari release resmi Ultralytics. Test menyalin notebook dan input ke system-temp di luar checkout, memakai `python -I`, dan memblokir import `inatrc_kd`. Pip jaringan dilewati harness; dependency lokal yang sudah terpasang dipakai.

| Eksperimen | CPU lokal | Kaggle GPU | Error jika gagal |
| --- | --- | --- | --- |
| Teacher m | **PASS** | **NOT TESTED** | — |
| Baseline s | **PASS** | **NOT TESTED** | — |
| Baseline n | **PASS** | **NOT TESTED** | — |
| Native KD s | **PASS** | **NOT TESTED** | — |
| Native KD n | **PASS** | **NOT TESTED** | — |
| CrossKD s | **PASS** | **NOT TESTED** | — |
| CrossKD n | **PASS** | **NOT TESTED** | — |
| CSAKD s | **PASS** | **NOT TESTED** | — |
| CSAKD n | **PASS** | **NOT TESTED** | — |

Seluruh kondisi menyelesaikan **1 epoch dan 2 optimizer step**, dua batch loss finite, serta checkpoint reload/val. Semua enam KD memakai satu teacher COCO8 yang dilatih oleh fixture notebook 01, dua batch loss KD asli, gradient student/projector nonzero, dan teacher frozen/state tetap. State student awal sama antar-baseline/KD untuk ukuran dan seed yang sama. Tidak ada audit InaTRC, evaluasi test, COCO area/profil tambahan, atau ZIP yang dipanggil. Logger mencatat satu epoch, best epoch 1, tanpa duplikasi final validation. Checkpoint bertag juga dimuat kembali oleh test.

| Pengujian | Hasil |
| --- | --- |
| Sembilan benchmark COCO8 pretrained CPU mandiri | **9 passed**, 97 deselected, 109,08 detik |
| Regresi seluruh repository tanpa smoke, Python 3.12 | **458 passed, 3 skipped**, 28 deselected, 82,62 detik |
| Notebook/runtime/data/COCO/tracking tanpa smoke, Python 3.13 | **254 passed**, 19 deselected, 51,53 detik |
| Sembilan regresi smoke sintetis InaTRC, `BENCHMARK_TEST=False` | **9 passed**, 97 deselected, 139,39 detik |
| Generator `--check`, lock freshness, `git diff --check` | Lolos |

Tiga test skipped memerlukan CUDA. Unit tambahan hanya memeriksa pilihan benchmark, bypass InaTRC, 80 kelas, teacher terlatih, guard default lima kelas, GPU platform, dan pemilihan satu cabang Run All. Test yang sudah ada tetap memeriksa gradient/loss, optimizer adapter, checkpoint, serta logging.

Perintah aktual:

```powershell
uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py -m smoke -k coco8_benchmark_cpu -q --basetemp outputs/coco8-benchmark/pytest-smoke --junitxml outputs/coco8-benchmark/pytest-smoke.xml
uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest -m "not smoke" -q --basetemp outputs/coco8-benchmark/regression --junitxml outputs/coco8-benchmark/regression.xml
uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py -m smoke -k standalone_smoke_runs -q --basetemp outputs/coco8-benchmark/inatrc-regression-smoke --junitxml outputs/coco8-benchmark/inatrc-regression-smoke.xml

$env:UV_PROJECT_ENVIRONMENT = "outputs/python313-venv"
uv run --frozen --no-sync --offline --cache-dir .uv-cache --python 3.13 pytest tests/test_standalone_runtime.py tests/test_standalone_notebooks.py tests/test_data.py tests/test_coco.py tests/test_config_tracking.py -m "not smoke" -q --basetemp outputs/coco8-benchmark/python313
Remove-Item Env:UV_PROJECT_ENVIRONMENT

uv run --frozen --no-sync --offline --cache-dir .uv-cache python scripts/generate_standalone_notebooks.py --check
uv lock --check --offline --cache-dir .uv-cache
git diff --check
```

Perintah uv hanya pengembangan/test lokal; notebook Kaggle tetap memakai pip. Fixture benchmark lokal membutuhkan asset resmi di `outputs/coco8-benchmark/assets` dan tidak melakukan download otomatis ketika pytest berjalan. URL asset yang dipakai ada pada `assets/sources.json`. Log/summary sembilan kondisi ada di `outputs/coco8-benchmark/smoke-results/`; JUnit dan perbandingan protokol ada di folder induknya. Checkpoint di system-temp dibersihkan setelah test, sehingga path checkpoint pada log historis bukan file unduhan permanen.

## Blocker dan batas verifikasi

**CUDA tidak tersedia pada environment pengujian lokal.** Belum ada sesi GPU Kaggle/Colab yang dijalankan; seluruh status platform tersebut NOT TESTED. Instalasi pip pada image Kaggle nyata, ABI/CUDA platform, waktu download/setup, dan pemakaian VRAM GPU belum terbukti. Notebook memeriksanya dan berhenti bila tidak kompatibel, tanpa mengganti Torch otomatis atau fallback CPU.

CPU PASS membuktikan integrasi pipeline pada pretrained/data COCO8 nyata, bukan keberhasilan GPU atau kualitas KD. Dataset delapan gambar ini tidak digunakan untuk tuning/menyimpulkan peningkatan mAP. Tidak ada full training penelitian, multiseed, evaluasi test, commit, remote baru, atau push. Langkah berikutnya adalah menjalankan sembilan notebook tersebut pada GPU Kaggle dengan checkpoint teacher 01 yang sama.
