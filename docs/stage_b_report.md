# Penyerahan Tahap B — implementasi dan pilot lokal

CrossKD dan CSAKD sekarang terimplementasi sebagai adaptasi YOLO26, terintegrasi pada CLI dan runner Kaggle yang sama. Native tetap memakai implementasi resmi. Pengguna menyetujui mulai Tahap B; full training, multiseed, evaluasi test set, remote Git, push, dan upload tidak dijalankan.

## Perubahan utama

- `kd/crosskd.py`: fitur intermediate student head → adapter terdaftar → frozen teacher suffix; QFL dan GIoU direct-distance pada one-to-many. Gradient tidak diputus dengan no_grad pada cross route.
- `kd/csakd.py`: enam pasangan module backbone/neck dari graph; cross supervising, AAWM, dan AGKA. Varian persamaan paper menjadi default; varian fungsi loss kode penulis tersedia secara eksplisit.
- `kd/common.py`, `kd/trainer.py`, `kd/probe.py`: lifecycle shared, integrasi sebelum optimizer/EMA, provenance source, serta probe gradient KD saja. Tidak ada perubahan site-packages.
- Config KD, tracking, CLI, notebook, train/evaluasi logging dan tests diperbarui. Bobot custom masih **bobot diagnosis/pilot**, belum protokol penelitian final.

Keputusan matematis, axis perhatian, normalisasi, split head, kanal, serta perbedaan terhadap paper/kode asli dirinci dalam [desain Tahap B](stage_b_design.md). Implementasi tidak diklaim sebagai reproduksi persis paper.

## Pengujian aktual

Environment tetap Windows/Python 3.12.12, uv 0.9.8, Torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.155, W&B 0.28.1. Tidak ada dependency baru di luar lock yang sudah tersedia.

| Perintah aktual | Hasil |
|---|---|
| `uv lock --check --offline --cache-dir .uv-cache` | Lulus, 73 package |
| `uv run --frozen --offline --cache-dir .uv-cache pytest -q --basetemp outputs/pytest-stage-b-final-02` | **143 passed, 3 skipped, 80.68 detik** |
| `uv run --frozen --offline --cache-dir .uv-cache pytest -q tests/test_runner.py --basetemp outputs/pytest-stage-b-runner-final` | 13 passed, 0.27 detik; setelah markdown notebook diperbarui |
| `uv run --frozen --offline --cache-dir .uv-cache python -m inatrc_kd preflight --custom-probe crosskd --student s --device cpu --output-root outputs/stage-b-probes/crosskd-s` | Lulus; JSON probe tersimpan |
| CLI `run --method crosskd --student s --mode smoke` dan `run --method csakd --student s --mode smoke`, device CPU/W&B disabled | Keduanya lulus; log dan best.pt tersimpan di outputs/stage-b-pilot |

Redirection ke log digunakan pada sebagian perintah; log suite final berada di [stage-b-pytest-final.log](../outputs/stage-b-pytest-final.log). Parameter CLI lengkap untuk dua run terpisah mencakup `--output-root outputs/stage-b-pilot --device cpu --wandb-mode disabled`.

Tes memeriksa persamaan QFL/GIoU numerik, decode stride, CSAKD per-sample AAWM/literal L2, varian author, attention non-square, dan gradient cross feature. Probe KD-only pada m→s/n melakukan dua update AdamW: gradient student/adapter/neck finite nonzero, teacher frozen/eval/tanpa gradient, hash teacher tidak berubah, hook stabil dan nol setelah cleanup. Exception membersihkan cache. CPU BF16 autocast juga dijalankan untuk memastikan adapter/perhitungan KD FP32 benar; ini tidak menggantikan pengujian CUDA/AMP GPU.

Tiga test GPU dilewati: satu Native dan dua custom KD. Seluruh regresi fondasi dataset/config/tracking/checkpoint/OOM sebelumnya tetap lulus.

## Delapan pilot sintetis dengan teacher tetap

Suite menjalankan s dahulu, kemudian n; masing-masing none/native/crosskd/csakd, seed 42, 1 epoch, 160 px, batch 2, nbs=2. Satu checkpoint m diagnostik dipakai pada seluruh KD. Setiap run memiliki lima update optimizer dan menghasilkan best.pt, CSV epoch/history, config, metadata, metrik serta reload/validasi student biasa pada val.

Pemeriksaan pembanding lulus: initial student hash sama antar empat kondisi pada setiap ukuran; fingerprint dataset sama; hash checkpoint teacher sama pada semua KD; hash teacher tetap sebelum/sesudah training; parameter/FLOPs inference identik antar metode pada ukuran yang sama.

| Student | Metode | Update optimizer | Durasi training CPU (s) | Parameter inference | GFLOPs @160 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| s | none | 5 | 3.382 | 9,951,734 | 1.407898 | 0.0 |
| s | native | 5 | 4.490 | 9,951,734 | 1.407898 | 0.0 |
| s | crosskd | 5 | 4.453 | 9,951,734 | 1.407898 | 0.0 |
| s | csakd | 5 | 4.965 | 9,951,734 | 1.407898 | 0.0 |
| n | none | 5 | 2.385 | 2,505,750 | 0.361600 | 0.0 |
| n | native | 5 | 3.284 | 2,505,750 | 0.361600 | 0.0 |
| n | crosskd | 5 | 6.557 | 2,505,750 | 0.361600 | 0.0 |
| n | csakd | 5 | 3.771 | 2,505,750 | 0.361600 | 0.0 |

Semua angka di atas **nonfinal**. Durasi adalah satu pengukuran training CPU per run, bukan benchmark kecepatan metode. mAP sintetis tidak menilai kualitas penelitian. Teacher acak hanya memiliki satu forward kalibrasi BatchNorm tanpa gradient/optimizer dan dilarang untuk full.

[Rekap CSV](../outputs/stage-b-comparison.csv) dan [rekap JSON](../outputs/stage-b-comparison.json) mencantumkan path lengkap seluruh run. Artefak tersebut lokal dan diabaikan Git, sebagaimana venv, cache, serta checkpoint; source repository tetap siap di-commit manual.

## Penggunaan dan batas berikutnya

```powershell
uv sync --frozen
uv run --frozen pytest
uv run --frozen python -m inatrc_kd run --student s --method crosskd --mode smoke
uv run --frozen python -m inatrc_kd run --student n --method csakd --mode smoke
```

Kaggle memakai notebook yang sama, METHOD dipilih dari none/native/crosskd/csakd, MODE default smoke. Aturan uv yang mewarisi Torch CUDA dan memakai `--no-sync` tetap dipertahankan.

Implementasi dan pilot CPU selesai. **GPU/Kaggle, VRAM, AMP CUDA, dataset InaTRC asli, teacher penelitian, dan layanan W&B online belum diuji.** Pilot GPU/dataset nyata serta penetapan bobot KD final harus diselesaikan sebelum hasil dianggap siap penelitian. Full training/Tahap C dan test-set final tetap membutuhkan instruksi terpisah.
