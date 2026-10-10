# InaTRC · YOLO26 Knowledge Distillation

Repository skripsi dengan fondasi Tahap A dan **implementasi/pilot Tahap B**: audit InaTRC, teacher YOLO26m, baseline YOLO26s/n, Native/CrossKD/CSAKD, evaluasi `best.pt`, serta W&B manual. Jalur utama penggunaan adalah **sembilan notebook mandiri Kaggle/Colab**: semua kode eksperimen tampil langsung dalam notebook, tanpa clone GitHub, uv, atau import modul repository saat dijalankan. CLI lokal tetap tersedia dan default smoke.

**Periksa `MODE` sebelum Run All:** notebook teacher, baseline, dan Native KD default **full**; CrossKD/CSAKD default **smoke**. Semua memakai seed 42 dan W&B disabled. Pengujian implementasi dibatasi pada smoke sintetis satu epoch; full training dan multiseed tidak dijalankan oleh pengembang.

Verifikasi notebook mandiri: **187 passed** pada Python 3.12 dan **187 passed** pada Python 3.13 untuk suite tanpa smoke; **sembilan smoke CPU mandiri lolos di luar checkout**. State student awal baseline/KD cocok pada s/n dengan seed sama. Instalasi pip/provider Secrets memakai mock; GPU Kaggle/Colab belum diuji. Perintah, bukti, dan batas verifikasi ada di [laporan notebook mandiri](docs/standalone_notebooks.md).

Audit referensi awal ada dalam [docs/audit.md](docs/audit.md). Native memakai API resmi Ultralytics; CrossKD/CSAKD memakai adaptasi yang dijelaskan di [desain Tahap B](docs/stage_b_design.md). Default CSAKD mengikuti persamaan paper; varian fungsi loss kode penulis tersedia secara eksplisit. Bobot KD custom masih bobot pilot, belum konfigurasi final InaTRC. Hasil terkini ada dalam [laporan Tahap B](docs/stage_b_report.md).

Catatan verifikasi runner repository sebelumnya: **277 passed, 3 skipped** pada suite lengkap Python 3.12; **162 passed, 1 skipped** untuk notebook/bootstrap/kompatibilitas Python 3.13. CUDA/AMP GPU tidak tersedia. Smoke baseline/Native/CrossKD/CSAKD pada s dan n menghasilkan `best.pt` yang dimuat ulang dan divalidasi. Dataset InaTRC asli, GPU Kaggle, dan layanan W&B online belum diuji. Perintah dan batas verifikasi ada di [laporan notebook eksperimen](docs/notebook_experiments.md). Detail environment, perubahan lock, dan strategi mempertahankan CUDA kernel ada di [laporan Python 3.13](docs/python313_compatibility.md). [Laporan Tahap A](docs/stage_a_report.md) dan [laporan Tahap B](docs/stage_b_report.md) dipertahankan sebagai catatan hasil sebelumnya.

## Struktur

```text
configs/                    dataset, training, KD, tracking
src/inatrc_kd/              konfigurasi, audit, training, evaluasi, KD, rekap
notebooks/standalone/        sembilan notebook mandiri; jalur utama Kaggle/Colab
scripts/generate_standalone_notebooks.py  generator lokal notebook mandiri
scripts/standalone_runtime.py  helper setup yang ditanam sebagai kode notebook
notebooks/01_kaggle_runner.ipynb  runner repository / template lama
notebooks/experiments/       sembilan runner berbasis source repository
scripts/generate_experiment_notebooks.py  generator runner repository
tests/                      logika audit/config dan regresi smoke
docs/audit.md               audit referensi dan keputusan kompatibilitas
reference/                  spesifikasi dan referensi asli
pyproject.toml · uv.lock    environment dan dependency terkunci
```

Output setiap eksperimen ada dalam direktori unik di `outputs/`, dengan identitas stage/model/method/seed/mode. Training yang berhasil menyimpan `best.pt`, `results.csv`, metrik agregat dan AP per kelas, resolved config, metadata environment/source, dan ringkasan run. `latest_run.json` hanya menunjuk run terakhir; hasil run lama tidak ditimpa.

Untuk membaca notebook mandiri, ikuti heading Bahasa Indonesia dari **Konfigurasi Eksperimen** hingga **Ringkasan dan Unduh Hasil**. Definisi fungsi dipisahkan dari cell yang menjalankannya; hanya algoritme KD yang diperlukan notebook tersebut yang dimuat. Untuk source repository lokal, mulai dari `cli.py → config.py → train.py → evaluate.py`. `data.py` memeriksa gambar/label, `preflight.py` memeriksa environment/model, dan `pipeline.py` menghubungkan loader dengan Ultralytics. `tracking.py` mencatat hasil; `summary.py` merekap beberapa run; `io.py` menyediakan helper file/hash. Folder `kd/` memisahkan algoritme KD dari alur training. Docstring di awal setiap file dan komentar pada langkah penting menjelaskan perannya secara singkat.

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

## Notebook mandiri untuk Kaggle dan Colab

| Nomor | Notebook | Model | Metode | MODE default |
| --- | --- | --- | --- | --- |
| 01 | [Teacher](notebooks/standalone/01_teacher_yolo26m.ipynb) | YOLO26m | none | full |
| 02 | [Baseline s](notebooks/standalone/02_baseline_yolo26s.ipynb) | YOLO26s | none | full |
| 03 | [Baseline n](notebooks/standalone/03_baseline_yolo26n.ipynb) | YOLO26n | none | full |
| 04 | [Native s](notebooks/standalone/04_native_yolo26s.ipynb) | YOLO26s | native | full |
| 05 | [Native n](notebooks/standalone/05_native_yolo26n.ipynb) | YOLO26n | native | full |
| 06 | [CrossKD s](notebooks/standalone/06_crosskd_yolo26s.ipynb) | YOLO26s | crosskd | smoke |
| 07 | [CrossKD n](notebooks/standalone/07_crosskd_yolo26n.ipynb) | YOLO26n | crosskd | smoke |
| 08 | [CSAKD s](notebooks/standalone/08_csakd_yolo26s.ipynb) | YOLO26s | csakd | smoke |
| 09 | [CSAKD n](notebooks/standalone/09_csakd_yolo26n.ipynb) | YOLO26n | csakd | smoke |

1. Upload **satu** notebook ke Kaggle, aktifkan GPU dan Internet, lalu tambahkan dataset InaTRC melalui **Add Input**. Untuk memeriksa setup dahulu, pilih **03 Baseline n** dan ubah `MODE = "smoke"`.
2. Baca **Konfigurasi Eksperimen**. Stage/model/metode sudah sesuai nama notebook; isi `DATASET_ROOT` jika deteksi otomatis menemukan lebih dari satu dataset, pilih seed/device/W&B, dan isi `TEACHER_CKPT` untuk KD full.
3. Run All memasang dependency memakai **pip pada interpreter kernel**, memeriksa GPU/ABI dan dataset, lalu menjalankan **tepat satu eksperimen**. Tidak perlu mengisi URL/commit repository atau mengunggah file `src/` dan `configs/`.
4. Setiap epoch mencetak metrik, fitness, `is_best`, dan `best_epoch`. Hasil, checkpoint, serta ZIP tersedia di **Ringkasan dan Unduh Hasil**. Di Kaggle, simpan dengan **Save Version** atau unduh hasil sebelum sesi berakhir.

Kaggle/Colab terdeteksi otomatis. Di Colab, aktifkan GPU melalui **Runtime → Change runtime type**, isi path dataset yang tersedia, dan uncomment blok Google Drive bila diperlukan. Blok Drive tetap dikomentari agar penggunaan Kaggle langsung bekerja. Output default `/kaggle/working/outputs` atau `/content/outputs` mengikuti platform.

Dependency utama dipin ke Ultralytics `8.4.155`, W&B `0.28.1`, dan faster-coco-eval `1.6.7`; versi aplikasi lainnya ditanam dari lock repository. Pip membuat rencana instalasi terlebih dahulu. Jika rencana mencoba memasang/mengganti Torch atau torchvision kernel, notebook berhenti sebelum perubahan tersebut. Paket yang telah di-resolve dipasang dengan `--no-deps`, lalu versi/lokasi Torch, CUDA, torchvision NMS, dan NumPy ↔ Torch diperiksa. Jika paket sudah di-import dengan versi berbeda, restart kernel setelah setup sesuai pesan error. Log setup tersimpan di folder output; setup pertama tetap memerlukan unduhan dependency, tetapi tidak mengambil source/history GitHub.

Untuk W&B, biarkan `WANDB_MODE = "disabled"` selama smoke awal. Mode `offline` tidak membutuhkan login. Mode `online` mengambil `WANDB_API_KEY` dari environment, **Kaggle Secrets**, atau **Colab Secrets** sesuai platform. Kaggle: tambahkan secret bernama `WANDB_API_KEY` dan aktifkan akses notebook. Colab: tambahkan secret dengan nama sama di panel 🔑 dan aktifkan **Notebook access**. Jangan menulis key langsung di cell; nilainya tidak dimasukkan ke konfigurasi, log, atau ZIP.

Semua KD full memerlukan **teacher YOLO26m InaTRC lima kelas yang sama**. Jalankan notebook 01 lebih dahulu, simpan `best.pt`, lalu lampirkan checkpoint tersebut sebagai Kaggle Input atau pilih path di Colab. Pertahankan checkpoint teacher yang sama untuk Native/CrossKD/CSAKD; hash dicatat setiap run. Teacher sintetis hanya dibuat untuk smoke dan ditandai sebagai diagnosis nonfinal. CrossKD/CSAKD tetap konfigurasi **pilot Tahap B**, bukan konfigurasi penelitian final.

Full mempertahankan protokol baseline: 640 px, batch 16, maksimal 100 epoch, patience 25, AdamW, lr0 0.001, weight decay 0.0005, warmup 3, AMP, deterministic, serta augmentasi yang sama. Semua model full dimulai dari pretrained COCO sesuai ukuran; hash checkpoint dan state awal student dicatat untuk membandingkan baseline/KD pada ukuran dan seed yang sama. Batch tidak diturunkan otomatis ketika OOM.

Audit membaca layout `ROOT/{train,val,test}/{images,labels}`, mempertahankan urutan lima kelas, dan memeriksa pasangan file, gambar rusak, bbox YOLO lima kolom, label kosong, anotasi duplikat, serta fingerprint isi file. Audit semua split tidak berarti mengevaluasi test set: training/pemilihan checkpoint/evaluasi akhir hanya memakai train/val. Cache loader ditempatkan di output, bukan dataset Input. Jumlah referensi 3.250 gambar harus cocok atau di-override secara eksplisit. Hash identik lintas split dilaporkan; kebocoran temporal/perseptual belum diperiksa.

Evaluasi memuat ulang `YOLO(best.pt)` tanpa teacher dan memakai val: `nms=None`, confidence efektif 0.001, IoU 0.7, max_det 300, augmentasi off, dan FP32. Metrik utama tetap mAP50–95 Ultralytics. **COCO AP/AP50/AP-S/AP-M/AP-L** disimpan terpisah dengan area dalam piksel gambar asli dan maxDets `[1,10,100]`; angka ini memakai protokol berbeda dari max_det 300. AP kelas/kelompok ukuran tanpa ground truth yang valid menjadi null. Parameter/FLOPs dihitung dari checkpoint lima kelas sebelum fusion; latency perkiraan, VRAM, dan durasi training dilaporkan terpisah.

Logging lokal selalu aktif, termasuk saat W&B disabled. `epoch_history.csv` dan `checkpoint_manifest.csv` mencatat epoch serta kejadian penyimpanan checkpoint. `best_epoch` mengikuti penyimpanan `best.pt`, termasuk fitness seri ketika Ultralytics menimpa checkpoint dengan epoch terbaru. Final validation tidak menambahkan epoch training duplikat. ZIP hasil menyertakan `best.pt`, `last.pt` bila tersedia, `results.csv`, konfigurasi, metadata, metrik, serta kurva. Finalisasi normal Ultralytics dapat menghapus optimizer state dari checkpoint akhir.

Detail alur, hasil verifikasi aktual, dan batasnya ada di [panduan notebook mandiri](docs/standalone_notebooks.md). GPU Kaggle/Colab, dataset InaTRC nyata, serta layanan W&B online belum dibuktikan oleh smoke CPU lokal.

### Pemeliharaan dan runner repository

Generator dijalankan **lokal oleh maintainer**. Notebook hasilnya dapat dipakai tanpa generator maupun repository:

```powershell
uv run --frozen python scripts/generate_standalone_notebooks.py
uv run --frozen python scripts/generate_standalone_notebooks.py --check
uv run --frozen pytest tests/test_standalone_notebooks.py tests/test_standalone_runtime.py tests/test_coco.py tests/test_config_tracking.py
```

Jangan mengedit sembilan salinan setup satu per satu. Ubah source/generator, regenerasi, lalu periksa diff. Notebook berisi fungsi/kelas Python biasa; tidak ada source tersembunyi melalui `exec`, base64, atau modul sementara.

[Runner umum](notebooks/01_kaggle_runner.ipynb) dan [sembilan runner repository](docs/notebook_experiments.md) tetap tersedia untuk penggunaan berbasis source/commit. Jalur lama memakai Git/uv dan mendukung snapshot offline beserta cache dependency Linux; kebutuhan tersebut **hanya berlaku untuk runner repository**, bukan notebook mandiri. Setup lokal PowerShell/Linux di atas juga tetap memakai uv.

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

Di Kaggle/Colab, gunakan notebook mandiri dengan MODE=full untuk protokol baseline/Native yang dipilih. CrossKD/CSAKD full memerlukan keputusan konfigurasi pilot terlebih dahulu. Jangan menjalankan perintah full lokal tersebut pada venv CPU yang sedang dipakai. Teacher antarsesi disimpan sebagai Kaggle output/dataset atau W&B artifact, lalu dilampirkan ke Input dan dipilih melalui `TEACHER_CKPT`. Gunakan checkpoint teacher tetap untuk seluruh kondisi KD. Pilot sintetis keempat kondisi sudah tersedia pada s/n; verifikasi GPU dan teacher InaTRC nyata masih diperlukan. Multiseed adalah Tahap C.

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

Jika remote origin sudah ada, gunakan remote yang benar tanpa menambahkannya lagi. Notebook mandiri dapat diunggah langsung ke Kaggle/Colab tanpa push. SHA commit hanya diperlukan jika Anda memilih runner repository lama.
