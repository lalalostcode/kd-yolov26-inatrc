# Sembilan notebook mandiri — Kaggle dan Colab

Notebook di `notebooks/standalone/` memuat kode konfigurasi, audit dataset, training, algoritme yang diperlukan, logging, dan evaluasi langsung dalam cell. File `.ipynb` dapat diunggah sendiri: tidak perlu clone GitHub, uv, file `src/`, atau YAML repository saat dijalankan. `scripts/generate_standalone_notebooks.py` hanya digunakan lokal untuk menghasilkan dan menjaga konsistensi sembilan notebook.

Notebook referensi `01-inatrc-yolo26n-baseline-kaggle-fixed-v2.ipynb` menjadi acuan alur dan parameter baseline. Pemeriksaan dataset, reproducibility, checkpoint, dan KD yang sudah ada dalam repository tetap dipertahankan. Source Ultralytics/KD tidak disalin sebagai framework baru; notebook memakai dependency Ultralytics terpin dan adaptasi KD yang diperlukan.

## Pilihan eksperimen

| File | Model | Metode | MODE default |
| --- | --- | --- | --- |
| [01_teacher_yolo26m.ipynb](../notebooks/standalone/01_teacher_yolo26m.ipynb) | YOLO26m | none | full |
| [02_baseline_yolo26s.ipynb](../notebooks/standalone/02_baseline_yolo26s.ipynb) | YOLO26s | none | full |
| [03_baseline_yolo26n.ipynb](../notebooks/standalone/03_baseline_yolo26n.ipynb) | YOLO26n | none | full |
| [04_native_yolo26s.ipynb](../notebooks/standalone/04_native_yolo26s.ipynb) | YOLO26s | native | full |
| [05_native_yolo26n.ipynb](../notebooks/standalone/05_native_yolo26n.ipynb) | YOLO26n | native | full |
| [06_crosskd_yolo26s.ipynb](../notebooks/standalone/06_crosskd_yolo26s.ipynb) | YOLO26s | crosskd | smoke |
| [07_crosskd_yolo26n.ipynb](../notebooks/standalone/07_crosskd_yolo26n.ipynb) | YOLO26n | crosskd | smoke |
| [08_csakd_yolo26s.ipynb](../notebooks/standalone/08_csakd_yolo26s.ipynb) | YOLO26s | csakd | smoke |
| [09_csakd_yolo26n.ipynb](../notebooks/standalone/09_csakd_yolo26n.ipynb) | YOLO26n | csakd | smoke |

Setiap notebook menjalankan tepat satu eksperimen. Seed default 42, W&B disabled. Ubah `MODE` menjadi `smoke` untuk memeriksa setup sebelum full; smoke memakai data sintetis nonfinal, satu epoch, 160 px, batch 2, workers 0, AMP off, dan `nbs=2`. Hasil smoke bukan hasil penelitian InaTRC. CrossKD/CSAKD tetap konfigurasi pilot Tahap B.

Penyederhanaan terbaru beserta pengukuran/test ada di [laporan audit dan optimasi](standalone_optimization.md). Teacher terkunci pada seed 42; seed student tetap bisa diganti. `EVALUATE_COCO_AREA = False` dan `CREATE_RESULTS_ZIP = False` adalah default baru untuk seluruh notebook. Gunakan opsi COCO yang sama di semua eksperimen yang dibandingkan.

Pilihan tambahan `BENCHMARK_TEST = True` menjalankan COCO8 resmi dengan pretrained YOLO26, 80 kelas, satu epoch, 320 px/batch 2/workers 0, seed 42. Ia mengabaikan `MODE` dan konfigurasi dataset InaTRC; semua KD memakai teacher hasil benchmark notebook 01. Hasil CPU dan instruksi GPU ada di [laporan COCO8](coco8_benchmark.md). Default flag False mempertahankan jalur InaTRC; tidak ada clone/uv/import repository atau notebook tambahan.

## Urutan cell

Heading memakai Bahasa Indonesia dan komentar singkat. Fungsi/kelas ditampilkan sebagai kode Python biasa dan didefinisikan sebelum cell yang memanggilnya.

1. **Konfigurasi Eksperimen** — preset model/metode, mode, seed, path, device, dan W&B.
2. **Instalasi Dependency** — pip interpreter kernel, rencana instalasi, perlindungan Torch/torchvision, serta log setup.
3. **Pemeriksaan GPU** — Python, dependency aktif, CUDA, torchvision NMS, dan NumPy–Torch.
4. **Penemuan dan Audit Dataset** — root Input/path Colab, layout, pasangan file, bbox, gambar, distribusi, dan fingerprint.
5. **Model dan Teacher** — model sesuai ukuran dan checkpoint teacher untuk KD.
6. **Algoritme KD** — algoritme yang diperlukan eksperimen tersebut; baseline/teacher tidak memuat implementasi KD custom.
7. **Logging per Epoch** — pencatatan lokal dan satu integrasi W&B manual.
8. **Evaluasi best.pt** — definisi evaluator, dipakai setelah training selesai.
9. **Training** — satu run, lalu evaluator memuat ulang checkpoint terbaik tanpa teacher.
10. **Ringkasan dan Unduh Hasil** — best epoch, metrik, checkpoint, dan ZIP bila diaktifkan.

## Setup Kaggle

Upload satu notebook, aktifkan GPU dan Internet, lalu tambahkan dataset melalui Add Input. Full memerlukan layout `ROOT/{train,val,test}/{images,labels}`. Root dataset dapat ditemukan otomatis jika hanya ada satu kandidat; jika ada beberapa, isi `DATASET_ROOT`. Dataset sumber hanya dibaca, dan cache loader diarahkan ke output.

Untuk pemeriksaan awal, gunakan notebook 03 dengan `MODE = "smoke"`, lalu Run All. Setelah setup/smoke GPU berhasil, isi dataset nyata dan gunakan mode full sesuai eksperimen yang dipilih. Notebook 01 menghasilkan teacher; simpan `best.pt` melalui Save Version lalu lampirkan checkpoint sebagai Input pada semua notebook KD full. Output default adalah `/kaggle/working/outputs`.

Notebook tidak memerlukan URL repository, commit SHA, snapshot source, atau file Python lain. Internet tetap diperlukan untuk paket pip yang belum tersedia dan bobot pretrained COCO full yang belum dilampirkan. Folder source GitHub bukan bagian setup ini.

## Setup Colab dan Secrets

Platform dideteksi otomatis. Aktifkan GPU melalui Runtime → Change runtime type, lalu isi path dataset. Blok mount Google Drive tetap dikomentari; uncomment hanya bila dataset/checkpoint/output ingin memakai Drive. Output default `/content/outputs` harus diunduh atau disalin sebelum sesi berakhir.

W&B disabled/offline tidak memerlukan login. Untuk online, sediakan secret `WANDB_API_KEY` di Kaggle Secrets atau panel 🔑 Colab dan aktifkan akses notebook. Environment dengan nama sama juga dapat dipakai. Provider Secrets mengikuti platform otomatis; key tidak ditulis langsung dalam cell, konfigurasi, log, atau ZIP. Pilihan project/entity berada pada konfigurasi.

Pip memakai `sys.executable`, sehingga paket masuk ke interpreter kernel. Ultralytics `8.4.155` dan W&B `0.28.1` dipin; `faster-coco-eval==1.6.7` hanya dipasang bila opsi COCO aktif. Dependency aplikasi lain berasal dari lock repository. Bila versi dan dependency graph sudah valid, resolver/instalasi dilewati. Jika belum, rencana pip diperiksa dahulu: perubahan Torch/torchvision bawaan ditolak. Instalasi paket yang telah di-resolve menggunakan `--no-deps`. Pemeriksaan GPU tetap memvalidasi versi/lokasi paket aktif dan operasi CUDA/NMS/NumPy–Torch sekali sebelum training. Dependency ter-import dengan versi lama dapat memerlukan restart kernel sesuai pesan error.

Log setup dan rencana pip tersimpan di folder `setup_logs/` pada output. Log memuat tahapan/durasi dan diagnosis; secret disamarkan. Kegagalan setup/preflight menghentikan notebook sebelum training. OOM menghentikan run tanpa mengurangi batch otomatis.

## Kesetaraan konfigurasi dan evaluasi

Full memakai 640 px, batch 16, maksimal 100 epoch, patience 25, AdamW, lr0 0.001, weight decay 0.0005, warmup 3, AMP, deterministic, dan augmentasi/default efektif yang sama. Konfigurasi efektif tersimpan setiap run. Baseline dan KD pada ukuran/seed yang sama dimulai dari pretrained COCO ukuran student yang sama; hash checkpoint dan state awal student dicatat, sehingga kesetaraan inisialisasi dapat diperiksa.

Semua KD full memakai teacher YOLO26m InaTRC lima kelas **yang sama**, frozen dan eval. Hash teacher dicatat. Teacher sintetis hanya untuk diagnosis smoke dan tidak diterima sebagai teacher full. Native menggunakan API resmi Ultralytics; CrossKD/CSAKD memakai mekanisme adaptasi yang didokumentasikan di [desain Tahap B](stage_b_design.md). Konfigurasi custom masih pilot.

Urutan kelas dipertahankan: Kelas-1 (Non-truk), Kelas-2 (Truk 2 sumbu), Kelas-3 (Truk 3 sumbu), Kelas-4 (Truk 4 sumbu), Kelas-5 (Truk >=5 sumbu). Audit memeriksa semua split, 3.250 gambar sesuai referensi atau override eksplisit, label lima kolom, kelas/koordinat finite, batas bbox, gambar rusak, pasangan hilang, duplicate stems, label kosong, serta anotasi duplikat. Fingerprint memakai isi file; duplikasi byte identik lintas split dilaporkan. Pemeriksaan temporal/perseptual belum dilakukan. Loader harus menerima jumlah gambar yang sesuai hasil audit. Test tidak digunakan untuk evaluasi/pemilihan checkpoint.

Audit lengkap dijalankan baru setiap run. Loader mencocokkan hash file dengan audit itu sebelum melewati decode integritas tambahan; cache loader masih diperiksa terhadap isi/label. Setelah training, seluruh byte dan inventaris file diperiksa kembali terhadap fingerprint awal. Perubahan label/gambar/nama, penambahan, atau penghapusan file menggagalkan run. Tidak ada penggunaan hasil audit lama berdasarkan mtime.

Evaluasi akhir memuat ulang `YOLO(best.pt)` dan memakai val dengan `nms=None`, confidence efektif 0.001, IoU 0.7, max_det 300, augmentasi off, dan FP32. Metrik utama mAP50–95 Ultralytics, disertai mAP50, precision, recall, serta AP per kelas; kelas tanpa evaluasi valid diberi null. Parameter/FLOPs berasal dari checkpoint lima kelas sebelum fusion. Latency perkiraan, VRAM, dan durasi training dipisahkan.

Jika `EVALUATE_COCO_AREA = True`, COCO AP/AP50/AP-S/AP-M/AP-L disimpan sebagai metrik final tambahan, dengan IoU 0.50:0.05:0.95, area dalam piksel gambar asli, dan maxDets `[1,10,100]`. Protokol ini berbeda dari max_det 300 pada metrik utama. Prediksi kosong menghasilkan AP nol jika ground truth valid tersedia; kelas/kelompok ukuran tanpa ground truth valid bernilai null. Nama file gambar tidak harus numerik. Jika nonaktif, export prediksi JSON dan evaluasi COCO tambahan dilewati; metrik utama tetap sama.

## Transparansi epoch dan checkpoint

Logging lokal selalu berjalan, termasuk ketika W&B disabled. Setiap epoch mencatat loss deteksi/KD yang tersedia, metrik val, learning rate, fitness, durasi, `is_best`, dan `best_epoch`, serta menampilkan ringkasan singkat di layar. W&B manual mencatat sekali per epoch; final validation tidak membuat epoch duplikat.

`best_epoch` mengikuti kejadian penyimpanan `best.pt`. Jika fitness seri dan Ultralytics menimpa checkpoint dengan epoch terbaru, manifest dan ringkasan menunjuk epoch penyimpanan terbaru tersebut. Riwayat tidak bergantung pada checkpoint terpisah per epoch.

Hasil meliputi `best.pt`, `last.pt` bila tersedia, `results.csv`, `epoch_history.csv`, `checkpoint_manifest.csv`, konfigurasi, fingerprint, metadata/provenance, metrik JSON/CSV, dan kurva. Artifact final W&B online menyertakan hasil penting dan best epoch. ZIP hanya dibuat jika `CREATE_RESULTS_ZIP = True` dan mengecualikan dataset sintetis, checkpoint teacher diagnostik, dan secret. Kaggle Save Version dapat menyimpan folder tanpa ZIP. `latest_run.json` menunjuk run terakhir; direktori unik menjaga hasil sebelumnya. Finalisasi Ultralytics dapat menghapus optimizer state, sehingga checkpoint final tidak selalu dapat dipakai resume.

## Verifikasi dan batas

Pengujian notebook memeriksa nbformat/sintaks, sembilan preset/default, generator konsisten, tepat satu eksperimen, serta ketiadaan import repository/clone/uv/source tersembunyi. Pengujian runtime mencakup rencana pip dan perlindungan Torch, Secrets Kaggle/Colab dengan mock, audit dataset, teacher, gradient/optimizer adapter, teacher frozen, reload checkpoint tanpa teacher, logging fitness seri/final validation, dan kasus COCO kosong/nama nonnumerik/ground truth tidak tersedia.

Hasil implementasi awal pada Windows CPU, 10 Oktober 2026, sebelum penyederhanaan ini (hasil terbaru ada di [laporan optimasi](standalone_optimization.md)):

| Pemeriksaan | Hasil |
| --- | --- |
| Suite notebook/runtime/COCO/tracking, Python 3.12, tanpa smoke | **187 passed**, 10 deselected |
| Suite yang sama, Python 3.13, tanpa smoke | **187 passed**, 10 deselected |
| Sembilan notebook, smoke satu epoch di luar checkout | **9 passed**, 136 detik |
| Evaluator COCO, termasuk validasi Ultralytics CPU dengan prediksi kosong | **36 passed** |
| Regresi bootstrap Git lama, fixture lokal di luar sandbox MSYS | **28 passed** |
| `uv lock --check`, generator `--check`, `git diff --check` | Lolos |

Perintah suite utama yang benar-benar dijalankan:

```powershell
uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py tests/test_standalone_runtime.py tests/test_coco.py tests/test_config_tracking.py -m "not smoke" -q --basetemp outputs/pytest-standalone-final-unit

$env:UV_PROJECT_ENVIRONMENT = "outputs/python313-venv"
uv run --frozen --no-sync --offline --cache-dir .uv-cache --python 3.13 pytest tests/test_standalone_notebooks.py tests/test_standalone_runtime.py tests/test_coco.py tests/test_config_tracking.py -m "not smoke" -q --basetemp outputs/pytest-standalone-final-py313
Remove-Item Env:UV_PROJECT_ENVIRONMENT

uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py -m smoke -q --basetemp outputs/pytest-standalone-isolated-03
```

Smoke menyalin hanya satu notebook per proses ke folder sementara di luar checkout, menjalankan Python `-I`, dan memblokir import `inatrc_kd`. Harness menggunakan dependency terpin yang sudah terpasang; **instalasi pip dilewati**, bukan diuji melalui jaringan. Pip dry-run, penolakan penggantian Torch, restart karena paket lama masih di-import, dan provider Secrets diperiksa dengan mock. Custom KD berjalan pada namespace `__main__` seperti kernel notebook, sehingga kelas inline dapat diserialisasi dengan benar.

Pada implementasi awal, semua smoke menghasilkan `best.pt`, `last.pt`, `results.csv`, satu baris `epoch_history.csv`/manifest, config, metadata, metrik val/COCO, serta ZIP. COCO/ZIP sekarang opsional. Checkpoint lima kelas dimuat ulang dan divalidasi tanpa teacher/adapter. Masing-masing melakukan **5 update optimizer**, `best_epoch=1`, dan tidak menulis cache ke dataset sumber. Bukti/log smoke awal disimpan lokal di `outputs/pytest-standalone-isolated-03/`; folder sementara checkpoint dibersihkan setelah test.

Hash state student awal hasil smoke sama pada baseline/Native/CrossKD/CSAKD, seed 42:

| Student | SHA-256 state awal |
| --- | --- |
| YOLO26s | `3e6fdba7b4a148854f51b4a3f63287cd8453b1a370142bdfa82258b1a7c82c69` |
| YOLO26n | `0a7376620c862bd25112b1d862b6ad34d4afd22b780fe7d3ee6ac826fecff046` |

Ini membuktikan kesamaan inisialisasi **smoke acak**, bukan hasil training pretrained COCO full. Protokol full dibuktikan lewat konfigurasi/guard; hash pretrained dan state aktual tetap dicatat ketika pengguna menjalankan full.

Provenance kode memakai SHA-256 gabungan source seluruh code cell dengan digest diganti `CODE_SHA256_PLACEHOLDER` saat menghitung. `code_hash_scheme` menjelaskan aturan ini; test mengulang perhitungannya. Pin aplikasi selain Torch diambil generator dari lock. [File distribusi faster-coco-eval 1.6.7](https://pypi.org/project/faster-coco-eval/1.6.7/#files) menyediakan wheel Python 3.11–3.13; [pip install](https://pip.pypa.io/en/stable/cli/pip_install/) menjadi acuan dry-run/report dan `--no-deps`.

Percobaan awal E2E terhenti karena sandbox melarang system-temp; retry yang disetujui memakai fixture lokal berhasil. Percobaan custom awal memakai namespace harness terpisah sehingga pickle tidak menemukan kelas `__main__`; harness diperbaiki dan sembilan smoke ulang lolos. Regresi unit luas awal mencatat 385 passed, 3 skipped, dengan 9 kegagalan fixture Git akibat izin named objects MSYS; seluruh 28 test bootstrap kemudian lolos pada retry lokal yang disetujui. Tidak ada akses GitHub atau push dalam test bootstrap.

Laporan [runner repository sebelumnya](notebook_experiments.md), [Tahap A](stage_a_report.md), dan [Tahap B](stage_b_report.md) merupakan hasil pekerjaan terdahulu, bukan bukti bahwa notebook mandiri sudah berjalan di GPU.

Verifikasi implementasi hanya menjalankan smoke sintetis satu epoch. Tidak ada default full, multiseed, evaluasi test set, atau git push. Keberhasilan CPU lokal tidak membuktikan runtime GPU Kaggle/Colab, InaTRC nyata, W&B online, atau pip jaringan platform sudah berhasil. Sebelum eksperimen penelitian, jalankan smoke di platform tujuan lalu periksa log, checkpoint, dan konfigurasi. CrossKD/CSAKD full tetap membutuhkan keputusan konfigurasi pilot.

## Regenerasi lokal

```powershell
uv run --frozen python scripts/generate_standalone_notebooks.py
uv run --frozen python scripts/generate_standalone_notebooks.py --check
uv run --frozen pytest tests/test_standalone_notebooks.py tests/test_standalone_runtime.py tests/test_coco.py tests/test_config_tracking.py
```

Notebook hasilnya mandiri dan tidak membutuhkan perintah ini saat digunakan di Kaggle/Colab. Ubah source/generator bersama, bukan sembilan notebook satu per satu. Runner Git/uv lama tetap dipertahankan di `notebooks/experiments/` dan `notebooks/01_kaggle_runner.ipynb`.
