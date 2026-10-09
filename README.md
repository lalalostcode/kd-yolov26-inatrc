# InaTRC · YOLO26 Knowledge Distillation

Repository skripsi dengan fondasi Tahap A dan **implementasi/pilot Tahap B**: audit InaTRC, teacher YOLO26m, baseline YOLO26s/n, Native/CrossKD/CSAKD, evaluasi `best.pt`, W&B manual, dan satu runner Kaggle. Default CLI/notebook adalah **smoke sintetis nonfinal**; tidak ada full training atau multiseed otomatis.

Audit referensi awal ada dalam [docs/audit.md](docs/audit.md). Native memakai API resmi Ultralytics; CrossKD/CSAKD memakai adaptasi yang dijelaskan di [desain Tahap B](docs/stage_b_design.md). Default CSAKD mengikuti persamaan paper; varian fungsi loss kode penulis tersedia secara eksplisit. Bobot KD custom masih bobot pilot, belum konfigurasi final InaTRC. Hasil terkini ada dalam [laporan Tahap B](docs/stage_b_report.md).

Verifikasi Python 3.13 terbaru: **173 test passed, 3 skipped** (CUDA/AMP GPU tidak tersedia). Smoke baseline/Native/CrossKD/CSAKD pada s dan n menghasilkan `best.pt` yang dimuat ulang dan divalidasi. Dataset InaTRC asli, GPU Kaggle, dan layanan W&B online belum diuji. Detail environment, perubahan lock, dan strategi mempertahankan CUDA kernel ada di [laporan Python 3.13](docs/python313_compatibility.md). [Laporan Tahap A](docs/stage_a_report.md) dan [laporan Tahap B](docs/stage_b_report.md) dipertahankan sebagai catatan hasil sebelumnya.

## Struktur

```text
configs/                    dataset, training, KD, tracking
src/inatrc_kd/              konfigurasi, audit, training, evaluasi, KD, rekap
notebooks/01_kaggle_runner.ipynb
tests/                      logika audit/config dan regresi smoke
docs/audit.md               audit referensi dan keputusan kompatibilitas
reference/                  spesifikasi dan referensi asli
pyproject.toml · uv.lock    environment dan dependency terkunci
```

Output setiap eksperimen ada dalam direktori unik di `outputs/`, dengan identitas stage/model/method/seed/mode. Training yang berhasil menyimpan `best.pt`, `results.csv`, metrik agregat dan AP per kelas, resolved config, metadata environment/source, dan ringkasan run. `latest_run.json` hanya menunjuk run terakhir; hasil run lama tidak ditimpa.

Untuk membaca kode, mulai dari `cli.py → config.py → train.py → evaluate.py`. Notebook hanya menyiapkan environment dan memanggil CLI. `data.py` memeriksa gambar/label, `preflight.py` memeriksa environment/model, dan `pipeline.py` menghubungkan loader dengan Ultralytics. `tracking.py` mencatat hasil; `summary.py` merekap beberapa run; `io.py` menyediakan helper file/hash. Folder `kd/` memisahkan algoritme KD dari alur training. Docstring di awal setiap file dan komentar pada langkah penting menjelaskan perannya secara singkat.

## Setup Windows PowerShell

Gunakan Python 3.11/3.12/3.13. Default lokal tetap 3.12 di `.python-version`; runner memakai interpreter kernel Kaggle/Colab. Tidak perlu aktivasi venv manual.

```powershell
# Sekali saja bila uv belum terpasang:
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/0.9.8/install.ps1 | iex"

Set-Location "C:\path\to\inatrc-kd-yolov26"  # sesuaikan lokasi repository Anda
uv sync --frozen
uv run --frozen python -m inatrc_kd preflight --device cpu
uv run --frozen pytest
uv run --frozen python -m inatrc_kd run --stage student --student n --method none --seed 42 --mode smoke --device cpu --wandb-mode disabled
```

Linux:

```bash
curl -LsSf https://astral.sh/uv/0.9.8/install.sh | sh
cd /path/to/inatrc-kd-yolov26
uv sync --frozen
uv run --frozen python -m inatrc_kd preflight --device cpu
uv run --frozen pytest
uv run --frozen python -m inatrc_kd run --student n --method none --seed 42 --mode smoke --device cpu
```

`uv sync --frozen` menggunakan lock yang telah disiapkan. Maintainer memeriksa kesesuaiannya dengan `uv lock --check` setelah mengubah dependency. Jangan edit lock secara manual.

Lock memilih Torch/Torchvision **CPU** secara eksplisit untuk verifikasi lokal; tidak mengklaim GPU Windows aktif. Untuk eksperimen GPU, gunakan strategi Kaggle di bawah. Versi Ultralytics `8.4.155` dan W&B `0.28.1` mengikuti baseline teruji setelah audit API; versi efektif tetap dicatat setiap run.

## Satu eksperimen di Kaggle

1. Upload [notebooks/01_kaggle_runner.ipynb](notebooks/01_kaggle_runner.ipynb), aktifkan GPU, dan lampirkan dataset InaTRC bila tersedia.
2. Isi `REPO_URL` dan **commit SHA lengkap 40 karakter**, atau `SNAPSHOT_ROOT` yang berisi `pyproject.toml` + `uv.lock`. Clone membutuhkan internet. Snapshot source disalin dari Input ke Working dan diberi fingerprint.
3. Pilih `STAGE`, `STUDENT`, `METHOD`, `SEED`, `MODE`. Default: student n / none / 42 / smoke. Teacher selalu m. Isi `DATASET_ROOT` bila deteksi otomatis menemukan lebih dari satu root.
4. Run All mem-bootstrap uv melalui interpreter kernel, menyiapkan venv, memeriksa CUDA/dependency, lalu menjalankan **satu** eksperimen.
5. Lihat `best.pt` dan metrics pada cell terakhir. Simpan melalui **Save Version** atau W&B artifact; filesystem sesi Kaggle tidak permanen.

Untuk Colab, gunakan notebook yang sama: uncomment empat assignment pada blok `COLAB PATHS` di cell konfigurasi, aktifkan GPU melalui Runtime → Change runtime type, lalu isi `REPO_URL` + `REPO_COMMIT` atau snapshot. Upload input ke `/content/inputs`; blok `COLAB DRIVE` yang dikomentari dapat dipakai jika input ada di Google Drive. Jika W&B online, buat secret `WANDB_API_KEY` di panel 🔑 Secrets, aktifkan Notebook access, dan uncomment dua baris pada blok `COLAB SECRET` di cell Secrets. Setup uv, pemeriksaan Torch, dan pipeline tetap dipakai bersama. Output default `/content/outputs` harus diunduh/disalin ke Drive sebelum sesi berakhir. Python kernel harus 3.11/3.12/3.13; incompatibility tetap gagal eksplisit. Blok Colab diuji lokal dengan mock Secrets, bukan pada runtime GPU Colab nyata. API Secrets mengikuti [source resmi Colab](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py).

Dataset menggunakan layout asli `ROOT/{train,val,test}/{images,labels}`. Audit mencatat pasangan file, kelas, bbox YOLO lima kolom, label kosong, distribusi, fingerprint, dan duplikasi byte identik lintas split. Polygon/format lain atau data tidak valid menyebabkan kegagalan jelas. Tidak ada konversi, relabel, penghapusan, atau split ulang otomatis. Jumlah referensi 3.250 gambar dikonfigurasi di `configs/dataset.yaml`. Pemeriksaan hash tidak membuktikan tidak adanya kebocoran temporal.

Smoke selalu memakai data sintetis dan inisialisasi model acak tanpa mengunduh bobot. InaTRC yang terpasang tetap diaudit, tetapi hasil smoke bukan metrik penelitian. KD native smoke dapat menggunakan teacher diagnostik acak bila checkpoint belum diberikan; ini bukan checkpoint teacher penelitian.

Strategi CUDA runner:

```text
uv venv --python <interpreter-kernel> --system-site-packages --no-python-downloads .venv
uv sync --frozen --no-dev --python <interpreter-kernel> --no-python-downloads --no-install-package torch --no-install-package torchvision
uv run --frozen --no-sync --python <interpreter-kernel> python -m inatrc_kd preflight --device 0 --require-gpu
uv run --frozen --no-sync --python <interpreter-kernel> python -m inatrc_kd run ...
```

Torch/Torchvision CUDA preinstalled diwarisi sebagai pengecualian lock yang eksplisit; versi aktifnya boleh berbeda dari pasangan CPU lokal terkunci jika masih memenuhi dependency proyek dan pasangan versi resmi yang kompatibel. Paket lain mengikuti lock. **Jangan hilangkan `--no-sync` dari pemanggilan Kaggle setelah setup**: sync otomatis dapat memasang Torch CPU dari lock. Runner membandingkan lokasi/versi Torch dan torchvision serta versi Python sebelum dan setelah setup. Preflight memeriksa pasangan versi serta dependency wajib `Requires-Dist` Torch/Torchvision yang benar-benar aktif, lalu melakukan operasi CUDA kecil, torchvision NMS, dan konversi NumPy ↔ Torch, bukan hanya `cuda.is_available()`. Ketidakcocokan dependency/CUDA/ABI/OOM menyebabkan kegagalan jelas; runner tidak mengganti wheel Torch atau menurunkan batch otomatis.

**Python 3.13:** menurut [tabel resmi torchvision](https://github.com/pytorch/vision#installation), gunakan Torch ≥2.7 dengan pasangan torchvision ≥0.22 yang sesuai. Torch 2.4–2.6 hanya diterima pada Python 3.11/3.12; nightly/prerelease tidak diterima. Lock CPU tetap Torch 2.10.0/torchvision 0.25.0, dengan wheel CPython 3.13. Strategi ini mempertahankan ABI kernel; mengganti kernel ke 3.12 sambil memakai wheel CPython 3.13 tidak didukung. Jika notebook lama masih berhenti pada batas 3.12, perbarui notebook **dan** snapshot/commit repository yang berisi `pyproject.toml`, `uv.lock`, serta `src` terbaru. Gunakan sesi baru untuk menghindari venv/source lama.

Kaggle GPU **belum dinyatakan teruji** oleh verifikasi lokal. Lock lintas environment tidak menjamin kompatibilitas CUDA. Python, dependency aktif, GPU, CUDA, hash lock, dan commit/fingerprint source dicatat.

Untuk internet nonaktif, lampirkan snapshot source, binary uv Linux 0.9.8, dan cache uv Linux sesuai versi Python kernel yang telah diisi dependency beserta build backend. Isi `OFFLINE=True`, `UV_BINARY`, dan `UV_CACHE_INPUT`. Cache disalin ke Working sebelum sync `--offline`. Source saja tidak cukup; paket yang tidak tersedia menghasilkan kegagalan jelas. Untuk full pretrained training offline, lampirkan bobot pretrained COCO yang sesuai model (m/s/n, head 80 kelas) dan pilih path melalui `MODEL_WEIGHTS` atau CLI `--weights`. Lampirkan teacher InaTRC lima kelas secara terpisah melalui `TEACHER_CKPT`/`--teacher-ckpt` untuk KD. Bobot baseline yang telah selesai fine-tuning bukan inisialisasi student. Portabilitas cache harus diverifikasi pada Kaggle; jangan mengasumsikan wheelhouse biasa menggantikan URL dalam frozen lock.

## Pilihan run dan hasil

Verifikasi native KD satu kali, tanpa full training:

```powershell
uv run --frozen python -m inatrc_kd run --stage student --student n --method native --seed 42 --mode smoke --device cpu --wandb-mode disabled
```

Konfigurasi penelitian berada dalam YAML terpisah. Baseline referensi menggunakan imgsz 640, epochs 100, batch 16, patience 25, AdamW, lr0 0.001, weight_decay 0.0005, dan AMP pada GPU. Val memakai protokol prediksi yang sama untuk semua model; metrik utama mAP50-95, disertai mAP50, precision, recall, dan AP lima kelas. Test tidak dipakai untuk tuning/debugging atau pemilihan checkpoint.

Contoh berikut adalah **perintah manual untuk nanti**, setelah protokol disetujui dan dataset tersedia:

```powershell
# Teacher satu kali; simpan best.pt untuk seluruh KD.
uv run --frozen python -m inatrc_kd run --stage teacher --method none --seed 42 --mode full --dataset-root "/path/to/InaTRC" --device 0

# Baseline student; masing-masing mulai dari pretrained student.
uv run --frozen python -m inatrc_kd run --stage student --student s --method none --seed 42 --mode full --dataset-root "/path/to/InaTRC" --device 0
uv run --frozen python -m inatrc_kd run --stage student --student n --method none --seed 42 --mode full --dataset-root "/path/to/InaTRC" --device 0
```

Di Kaggle, gunakan notebook dengan MODE=full setelah protokol final disetujui; jangan menjalankan perintah lokal tersebut pada venv CPU yang sedang dipakai. Teacher antarsesi disimpan sebagai Kaggle output/dataset atau W&B artifact, lalu dilampirkan ke Input dan dipilih melalui `TEACHER_CKPT`. Gunakan checkpoint teacher tetap untuk seluruh kondisi KD. Pilot sintetis keempat kondisi sudah tersedia pada s/n; verifikasi GPU dan teacher InaTRC nyata masih diperlukan. Multiseed adalah Tahap C.

Diagnosis custom KD satu run dan probe gradient:

```powershell
uv run --frozen python -m inatrc_kd preflight --custom-probe crosskd --student s --device cpu
uv run --frozen python -m inatrc_kd run --method crosskd --student s --mode smoke
uv run --frozen python -m inatrc_kd run --method csakd --student n --mode smoke
```

`configs/kd.yaml` memilih bobot CrossKD dan CSAKD serta `variant: paper|author`. Adapter/routing hanya ada saat training. Output mencatat initial student hash, teacher hash/state, jumlah update optimizer, config adaptasi, dan revision source. Test tidak digunakan selama development.

W&B default disabled. Pilih offline/online pada runner atau `--wandb-mode`; project/entity dapat dikonfigurasi. Untuk online, gunakan Kaggle Secret `WANDB_API_KEY` atau variabel environment. Jangan commit/cetak API key. Hanya satu run W&B manual per eksperimen: metrik epoch, final val, status, biaya training, serta artifact `best.pt` dan metrics. Upload checkpoint tiap epoch default `false` di `configs/tracking.yaml`; aktifkan hanya bila diperlukan. `last.pt` dipertahankan, tetapi finalisasi normal Ultralytics membuang optimizer state dan mengubah epoch sehingga checkpoint akhir tidak dijamin dapat di-resume. Resume membutuhkan checkpoint training yang terinterupsi dan masih menyimpan state lengkap. Kegagalan logging/checkpoint harus tetap terlihat.

Evaluasi ulang `best.pt` memakai split val; ringkasan CLI `summarize` membaca hasil yang tersimpan. Jalankan `uv run --frozen python -m inatrc_kd <command> --help` untuk opsi `audit`, `preflight`, `run`, `evaluate`, dan `summarize`. Parameter/FLOPs dilaporkan terpisah dari latency, VRAM, dan durasi; latency perkiraan bukan benchmark deployment final.

## GitHub: Anda yang push

Repository tidak membuat remote, upload source, atau menjalankan push. Venv, secrets, dataset besar, bobot, cache, W&B, serta output training diabaikan Git. Commit source, notebook, configs, README, pyproject, dan uv.lock.

Periksa dahulu dengan `git status` dan `git remote -v`. Bila belum ada repository Git, jalankan `git init`; bila sudah ada, cukup lanjutkan commit. Perintah berikut dijalankan **manual oleh Anda**:

```powershell
git add .
git commit -m "Add Stage A YOLO26 KD research foundation"
git remote add origin https://github.com/USER/REPO.git
git push -u origin HEAD
```

Jika remote origin sudah ada, gunakan remote yang benar tanpa menambahkannya lagi. Setelah push, salin SHA commit dari `git rev-parse HEAD` ke notebook Kaggle.
