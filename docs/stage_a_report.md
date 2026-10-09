# Laporan penyerahan Tahap A — 9 Oktober 2026

Fondasi repository selesai dan smoke sintetis terbukti berjalan di Windows CPU. Seluruh spesifikasi dan notebook dalam `reference/` telah dibaca; referensi asli dipertahankan. Audit paper/source dan keputusan adaptasi ada di [audit.md](audit.md). Belum ada full training, multiseed, evaluasi test set, remote Git, push, atau unggahan repository.

## Tree aktual

```text
.gitignore
.python-version                 Python lokal 3.12
pyproject.toml
uv.lock
README.md
configs/
  dataset.yaml  training.yaml  kd.yaml  tracking.yaml
src/inatrc_kd/
  __init__.py  __main__.py
  cli.py  config.py  io.py
  data.py  preflight.py  pipeline.py
  train.py  evaluate.py  tracking.py  summary.py
  kd/
    __init__.py  native.py  crosskd.py  csakd.py
notebooks/
  01_kaggle_runner.ipynb
tests/
  conftest.py  test_data.py  test_compatibility.py
  test_config_tracking.py  test_runner.py
  test_pipeline_safety.py  test_smoke.py  test_summary.py
docs/
  audit.md  stage_a_report.md
reference/
  CODEX_INSTRUKSI_LENGKAP_YOLO26_KD_UV_KAGGLE.md
  02-inatrc-yolo26n-baseline-full-wandb-audit.ipynb
  Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf
  paper.pdf
```

`.venv/`, `.uv-cache/`, dan `outputs/` tersedia lokal dan diabaikan Git. Git sudah ada tanpa commit/remote; seluruh deliverable source siap di-review dan di-commit sendiri oleh pengguna.

## Keputusan teknis

- Satu pipeline untuk teacher m dan student s/n. Config full mempertahankan protokol baseline; default smoke adalah 1 epoch, 160 px, batch 2, workers 0, AMP off, `nbs=2`. Smoke selalu memakai 20 gambar sintetis, tanpa bobot pretrained atau teacher penelitian.
- Ultralytics `8.4.155`, W&B `0.28.1`, Python `>=3.11,<3.13`; lock lokal Torch `2.10.0+cpu`/torchvision `0.25.0+cpu` dari index CPU resmi. `uv.lock` dibuat resolver, bukan diedit manual.
- Native memakai API upstream `distill_model` dengan `dis=6.0`. Teacher sintetis m lima kelas hanya diagnosis dan ditolak untuk full. CrossKD/CSAKD memiliki audit/config/stub yang gagal eksplisit dengan `NotImplementedError`.
- Audit dataset membaca sumber, memvalidasi gambar/label/bbox, dan mencatat fingerprint serta duplikasi identik lintas split. Loader membandingkan jumlah/nilai label yang diterima dan menyimpan cache di output. Output di dalam root sumber ditolak sebelum membuat direktori. Kebocoran temporal/perseptual belum diperiksa.
- Evaluasi memuat ulang student `YOLO(best.pt)` pada val, dengan `nms=None` dan precision float32; checkpoint tidak mengandung teacher/projector. AP kelas yang tidak valid bernilai null. Parameter/FLOPs terpisah dari speed/latency, VRAM, dan durasi.
- W&B manual satu run, disabled tanpa login, offline/online opsional, secret melalui environment/Kaggle Secrets. Epoch logging tidak membutuhkan checkpoint tiap epoch dan tidak terduplikasi oleh final validation. Artifact final berisi best.pt serta hasil penting. Final `last.pt` dipertahankan bila tersedia, tetapi stripping optimizer upstream membatasi resume.
- OOM gagal tanpa mengurangi batch atau membangun ulang optimizer. NaN/Inf gagal tanpa mengulang epoch dari checkpoint. Extension kecil tetap terisolasi; site-packages tidak diedit.

## Setup PowerShell dan Kaggle

Di root repository dengan uv tersedia:

```powershell
uv sync --frozen
uv run --frozen pytest
uv run --frozen python -m inatrc_kd run --stage student --student n --method none --seed 42 --mode smoke
```

Instruksi instalasi uv, Linux, kelima perintah CLI, dan Git manual ada di [README](../README.md).

Upload [runner](../notebooks/01_kaggle_runner.ipynb) ke Kaggle, aktifkan GPU, isi URL repository dengan SHA commit 40 karakter atau snapshot source, lalu pilih stage/student/method/seed/mode. Run All default menjalankan satu smoke sintetis. Bila InaTRC dilampirkan, sumber asli ikut diaudit. Semua hasil masuk `/kaggle/working/outputs`; simpan dengan Save Version. Teacher antar-sesi dilampirkan melalui Kaggle Input atau artifact W&B.

Runner membuat `.venv --system-site-packages` dengan interpreter kernel, melakukan frozen sync tanpa memasang Torch/torchvision, dan selalu memakai `uv run --frozen --no-sync` berikutnya. Versi/lokasi Torch dibandingkan sebelum/sesudah; Requires-Dist, CUDA tensor, dan NMS diperiksa. Ketidakcocokan gagal tanpa mengganti wheel. Offline membutuhkan binary uv Linux dan cache dependency Linux yang sesuai kernel, selain source snapshot.

## Verifikasi yang benar-benar dijalankan

Environment aktual: Windows, Python **3.12.12**, uv **0.9.8**, Torch **2.10.0+cpu**, torchvision **0.25.0+cpu**, Ultralytics **8.4.155**, W&B **0.28.1**. Instalasi frozen sync berhasil. Opsi cache/offline berikut dipakai agar verifikasi memakai cache workspace setelah dependency terpasang.

| Perintah | Hasil aktual |
|---|---|
| `uv lock --check --offline --cache-dir .uv-cache` | Lulus; 73 package resolved, lock sesuai pyproject |
| `uv run --frozen --offline --cache-dir .uv-cache python -m inatrc_kd --help` | Lulus; audit/preflight/run/evaluate/summarize tersedia |
| `uv run --frozen --offline --cache-dir .uv-cache python -m inatrc_kd preflight --device cpu` | Lulus; JSON, lokasi/versi, operasi tensor, NMS, dependency aktif |
| `uv run --frozen --offline --cache-dir .uv-cache pytest -q --basetemp outputs/pytest-final-01` | **120 passed, 1 skipped, 29.09 detik** |
| `uv run --frozen --offline --cache-dir .uv-cache python -m inatrc_kd run --student n --method none --seed 42 --mode smoke --device cpu --output-root outputs/cli-verification --wandb-mode disabled` | Lulus; integrasi CLI menghasilkan best.pt dan metrik |

Suite mencakup fixture audit valid/invalid, gambar rusak, pasangan hilang, polygon, bbox, fingerprint, label kosong/duplikat; config dan CLI; forward/loss/backward m/s/n; Native m→s/n dengan student/projector gradient finite, teacher frozen, serta hook stabil; notebook nbformat/AST/defaults; logging lifecycle dan error; custom KD gate; output source protection; rekap yang mengecualikan smoke/failed. Test CUDA/AMP dilewati karena CUDA tidak tersedia.

Dua smoke training nyata dalam suite menghasilkan best.pt, results.csv, resolved config, metadata, AP lima kelas, dan evaluasi ulang student tanpa teacher:

- [Baseline n: hasil lengkap](../outputs/pytest-final-01/test_training_checkpoint_round0/experiments/student-yolo26n-none-seed42-smoke-20261009T055226Z-5baec602/run_summary.json), [best.pt](../outputs/pytest-final-01/test_training_checkpoint_round0/experiments/student-yolo26n-none-seed42-smoke-20261009T055226Z-5baec602/weights/best.pt).
- [Native m→n: hasil lengkap](../outputs/pytest-final-01/test_training_checkpoint_round1/experiments/student-yolo26n-native-seed42-smoke-20261009T055230Z-2bd7eac3/run_summary.json), [best.pt](../outputs/pytest-final-01/test_training_checkpoint_round1/experiments/student-yolo26n-native-seed42-smoke-20261009T055230Z-2bd7eac3/weights/best.pt).

Keduanya selesai 1 epoch; mAP50-95 **0.0** pada data diagnosis. Angka ini bukan hasil skripsi. Checkpoint student memiliki 2.505.750 parameter; estimasi 0,3616 GFLOPs pada 160 px. Test OOM yang sengaja diinjeksi pada forward pertama juga berjalan melalui trainer nyata: status failed, batch tetap 2, tanpa retry atau latest-success pointer.

## Batas verifikasi dan berikutnya

InaTRC asli belum tersedia: audit nyata dan training teacher/baseline penelitian belum dijalankan. Notebook belum dieksekusi pada Kaggle; CUDA/AMP/VRAM dan portabilitas cache Linux belum terbukti. W&B disabled diuji pada run nyata; lifecycle online/offline diuji dengan mock, belum memakai layanan W&B. Kurva dan pilot latency full sudah disiapkan, tetapi smoke menonaktifkannya. Artefak diagnosis lokal di atas diabaikan Git dan tidak ikut upload source.

Berikutnya membutuhkan instruksi pengguna: verifikasi runner pada Kaggle dengan dataset aktual, menyetujui protokol pilot GPU, dan memilih keputusan adaptasi CrossKD/CSAKD yang dirinci dalam audit. Setelah persetujuan Tahap B, validasi routing gradient, AMP, VRAM, hook, serta checkpoint s/n. Full training, multiseed, dan test-set final tetap menunggu instruksi terpisah.
