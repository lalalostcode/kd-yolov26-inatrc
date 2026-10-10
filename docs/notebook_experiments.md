# Sembilan notebook eksperimen — 9 Oktober 2026

Sembilan notebook di `notebooks/experiments/` dibuat dari `notebooks/01_kaggle_runner.ipynb` melalui `scripts/generate_experiment_notebooks.py`. Urutannya teacher m, baseline s/n, Native s/n, CrossKD s/n, dan CSAKD s/n. Setiap notebook mengunci stage/model/metode dan memanggil satu eksperimen melalui CLI bersama. Seluruh notebook default smoke sintetis, seed 42, dan W&B disabled. Algoritme, parameter KD, serta protokol training/evaluasi tidak diubah.

Pengguna mengisi URL repository dan commit SHA lengkap setelah mengunggah source sendiri, atau memilih snapshot. Seed, mode, dataset, checkpoint teacher, device, dan W&B berada di cell konfigurasi. Semua KD full menggunakan teacher YOLO26m InaTRC lima kelas yang sama. Teacher acak hanya untuk diagnosis smoke. Blok Colab tetap dikomentari.

Bootstrap memakai fetch satu commit dengan depth 1 tanpa remote Git, serta penggunaan ulang source, venv, dan cache. Metadata URL/commit disimpan di `.git/inatrc_source.json`. Checkout tanpa HEAD dapat melanjutkan fetch bila hanya berisi `.git`; file pengguna dan identitas source yang berbeda menyebabkan kegagalan tanpa penghapusan. Checkout bersih dari versi lama tanpa marker tetap diterima jika commit sesuai, lalu diberi marker.

Log `source.log`, `dependencies.log`, dan `preflight.log` ditambahkan di `OUTPUT_ROOT/setup_logs/`. Tahap dan durasi muncul di layar. Download menampilkan progres; kegagalan menyebut tahap, ringkasan error, dan lokasi log. Nilai API key W&B dari environment/Secrets disamarkan. Environment lengkap tidak dicetak. Guard mencegah cell training dijalankan setelah bootstrap gagal. Pemeriksaan lokasi/versi Torch, torchvision, Python, CUDA, NMS, dan NumPy tetap digunakan; wheel Torch kernel tidak diganti otomatis.

## Verifikasi yang dijalankan

Environment lokal Windows menggunakan Python 3.12.12 dan environment Python 3.13.9 yang telah dibuat sebelumnya. Torch lokal adalah CPU. Tidak ada dependency penelitian atau lock yang diubah dalam pekerjaan ini.

| Perintah | Hasil |
| --- | --- |
| `uv lock --check --offline --cache-dir .uv-cache` | Lulus, 73 paket resolved |
| `uv run --frozen --no-sync --cache-dir .uv-cache python scripts/generate_experiment_notebooks.py --check` | Lulus, sembilan file sesuai template |
| `uv run --frozen --no-sync --cache-dir .uv-cache pytest tests/test_experiment_notebooks.py -q --basetemp outputs/pytest-nine-notebooks-01` | 75 passed, 1,98 detik |
| `uv run --frozen --no-sync --cache-dir .uv-cache pytest -q --basetemp outputs/pytest-nine-notebooks-full-01` | 277 passed, 3 skipped, 98,72 detik |

Verifikasi notebook/bootstrap/kompatibilitas pada Python 3.13:

```powershell
$env:UV_PROJECT_ENVIRONMENT = 'outputs/python313-venv'
uv run --frozen --no-sync --python 3.13 --no-python-downloads --cache-dir .uv-cache pytest tests/test_runner.py tests/test_bootstrap.py tests/test_experiment_notebooks.py tests/test_compatibility.py -q --basetemp outputs/pytest-nine-notebooks-python313-02
```

Hasil: **162 passed, 1 skipped, 28,36 detik**. Skip membutuhkan CUDA GPU yang tidak tersedia. Perintah ini memakai venv Python 3.13 lokal yang sudah terpasang, bukan bootstrap runtime Kaggle.

Pemeriksaan notebook mencakup validasi nbformat, sintaks semua cell, preset dan guard perubahan preset, satu pemanggilan eksperimen, parity setup template, default smoke, teacher wajib pada full, frozen sync dengan pengecualian Torch/torchvision, Secrets, serta blok Colab. Generator terbukti deterministik dan `--check` tidak menulis file yang hilang/berubah.

Pengujian fetch memakai repository fixture lokal dengan dua commit dan URL file, tanpa menghubungi GitHub. Fetch pertama menghasilkan satu commit; pengulangan tidak fetch lagi; repo tanpa HEAD dan simulasi fetch terputus berhasil dilanjutkan. Perubahan tracked/untracked, source berbeda, dan file pengguna tetap dipertahankan. Seluruh cell source juga dijalankan menggunakan snapshot fixture. Preflight notebook dijalankan dengan hasil environment mock untuk memeriksa penolakan perbedaan Torch/torchvision/Python sebelum training. Tes CUDA/ABI yang sesungguhnya tetap berada dalam suite dan membutuhkan GPU.

Fixture Git awalnya terhalang named object MSYS pada sandbox Windows. Retry lokal dengan izin eksekusi yang sesuai berhasil. Perintah awal verifikasi Python 3.13 juga salah menunjuk `tests/test_preflight.py` yang tidak ada; tidak ada test yang berjalan pada percobaan itu. Perintah di atas memakai `tests/test_compatibility.py` dan lulus.

Suite lengkap menjalankan regresi smoke CPU satu epoch pada baseline/Native/CrossKD/CSAKD untuk s dan n, termasuk gradient, teacher frozen, update optimizer, logging, checkpoint, serta pemuatan ulang dan validasi `best.pt` tanpa teacher. Tidak ada full training, multiseed, evaluasi test set, git push, atau perubahan dataset sumber.

## Batas dan langkah berikutnya

Sembilan notebook belum dijalankan pada runtime GPU Kaggle atau Colab. Tes fetch lokal tidak mengukur kecepatan jaringan GitHub/Kaggle. Dataset InaTRC nyata dan W&B online belum diuji. Traceback Kaggle terbaru belum tersedia; pemisahan notebook dan perbaikan bootstrap bukan bukti bahwa semua error runtime telah selesai.

Push perubahan secara manual, ambil SHA commit tersebut, lalu upload notebook **03 Baseline n** ke Kaggle dan jalankan dengan default smoke. Isi `REPO_URL` dan `REPO_COMMIT`, aktifkan GPU/Internet, dan gunakan log setup bila gagal. Lanjutkan smoke notebook lain sebelum memilih full dan teacher penelitian yang sama untuk seluruh KD.
