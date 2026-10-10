# Audit dan penyederhanaan sembilan notebook mandiri

Refactor minimal pada 10 Oktober 2026. Sembilan file di `notebooks/standalone/` tetap mandiri, menampilkan kode Python biasa, dan menjalankan satu eksperimen per Run All. Tidak ada framework, bootstrap, notebook eksperimen, atau sistem konfigurasi baru. Source modular lokal, runner lama, dependency lock, dan algoritme KD dipertahankan.

## Temuan audit dan keputusan

| Komponen | Temuan sebelum perubahan | Keputusan dan alasan |
| --- | --- | --- |
| Setup pip berulang | Resolver/dry-run tetap dipanggil walaupun semua pin sudah cocok | Lewati pip hanya jika versi **dan dependency wajib, termasuk transitif**, valid. Paket hilang/konflik tetap memakai resolver. |
| Pemeriksaan environment | Operasi tensor/NMS/NumPy diulang pada cell GPU dan awal run; metadata Torch diperiksa lagi sesudah setup | Operasi nyata sekali di cell GPU. Training memakai laporan itu, dengan guard jika device berubah. Metadata dependency dari setup dipakai kembali. |
| Probe model acak | Forward/backward model diagnostik dijalankan sebelum setiap eksperimen | Keluarkan dari jalur notebook; probe tetap tersedia pada CLI dan pengujian lokal. Training nyata masih memeriksa loss/gradient, teacher, optimizer, checkpoint, serta protokol. |
| Audit penuh berulang | Decode gambar dan validasi label diulang setelah training; pemeriksaan decode tambahan pada loader | Audit lengkap baru sekali setiap run. Loader mencocokkan hash dengan audit tersebut; pemeriksaan akhir membaca **seluruh byte dan inventaris file** kembali. |
| COCO area | Export prediksi dan evaluasi tambahan wajib | `EVALUATE_COCO_AREA = False` secara default. Jika aktif, evaluator/protokol COCO lama tetap dipakai. Metrik utama selalu tersedia. |
| ZIP hasil | Kompresi seluruh hasil wajib, padahal Kaggle Save Version sudah menyimpan folder | `CREATE_RESULTS_ZIP = False` secara default. Fungsi bundle tetap tersedia untuk unduhan manual. |
| Tabel akhir seluruh epoch | Riwayat dicetak ulang setelah sudah tampil per epoch | Akhir run hanya menampilkan ringkasan/checkpoint. CSV dan logging per epoch tetap lengkap. |
| Manifest/CSV/metadata | Sebagian isinya juga ada pada JSON/results.csv | Dipertahankan: manifest mengaudit kejadian save/fitness seri; epoch CSV memuat loss KD dan best epoch; AP per kelas, provenance, dan rekap tetap berguna untuk penelitian. Tidak ada bukti manfaat runtime dari menghapusnya. |

Pengurangan definisi probe memperbaiki keterbacaan. Pengurangan jumlah baris sendiri tidak dipakai sebagai bukti percepatan. Definisi COCO tetap terlihat dalam notebook agar opsi bisa diaktifkan tanpa import file lain; dependency evaluasi tambahan dimuat hanya ketika diperlukan.

## Integritas dataset

Audit lengkap tetap memeriksa train/val/test, pasangan image-label, duplicate stems, gambar rusak, lima kolom YOLO, finite/class ID/koordinat/batas bbox, label kosong, anotasi duplikat, distribusi, jumlah 3.250 atau override eksplisit, serta duplikasi byte identik lintas split. Tidak ada perubahan gambar, anotasi, split, atau urutan lima kelas.

Hash per file berasal dari pembacaan pada audit **run yang sedang berjalan**. Data YAML di output membawa hash itu ke loader. Sebelum loader meneruskan scan/cache Ultralytics, byte image/label harus cocok. Tanpa manifest fresh, loader tetap memakai pemeriksaan gambar lama. Cache lama tidak dipercaya berdasarkan nama/mtime; jumlah gambar dan label yang diterima loader masih dicocokkan.

Sesudah evaluasi, `verify_dataset_unchanged` membaca seluruh file lagi, membandingkan daftar nama, SHA-256 setiap file, dan fingerprint dengan awal run. Gambar/label berubah, termasuk perubahan berukuran sama dengan mtime tetap, nama berubah, file bertambah/hilang, dan cache yang tidak cocok menggagalkan run. Decode/parse tambahan hanya dilewati untuk byte yang sudah terbukti identik. Format fingerprint tetap sama; tidak ada cache audit antar-run. Hash tambahan tersimpan di output, tidak di dataset Input.

## Yang tetap sama secara ilmiah

- Identitas sembilan notebook dan default mode: teacher/baseline/Native full; CrossKD/CSAKD smoke. Seed default 42, W&B disabled, tidak ada loop model/seed atau retry training otomatis.
- Full: 640 px, batch 16, maksimum 100 epoch, patience 25, AdamW, lr0 0.001, weight decay 0.0005, warmup 3, AMP/deterministic, semua augmentasi dan default efektif lama. OOM tidak mengurangi batch otomatis.
- Arsitektur m/s/n, pretrained COCO, hash checkpoint/state awal, persamaan Native/CrossKD/CSAKD, teacher frozen/eval, gradient, adapter sebelum optimizer, dan pemilihan `best.pt`.
- Teacher notebook terkunci seed **42**; seed student tetap dapat diubah. Semua KD full harus memakai teacher YOLO26m InaTRC lima kelas yang sama. `TEACHER_SHA256` dapat mengunci identitasnya. Teacher sintetis hanya diagnosis smoke. Bobot CrossKD/CSAKD tetap **pilot**.
- Evaluasi memuat ulang `YOLO(best.pt)` tanpa teacher/adapter pada **val**, FP32, `nms=None`, conf 0.001, IoU 0.7, max_det 300, augment off. mAP50–95, mAP50, precision/recall, AP per kelas, parameter/FLOPs checkpoint lima kelas sebelum fusion, latency, VRAM, dan durasi tetap dicatat. AP tanpa evaluasi valid tetap null.
- COCO aktif tetap memakai luas bbox piksel asli, IoU 0.50:0.05:0.95, maxDets `[1,10,100]`, null untuk kelas/ukuran tanpa GT. Pilih opsi COCO yang sama untuk seluruh perbandingan eksperimen; opsi nonaktif tidak mengubah metrik utama.
- Logging sekali per epoch, best epoch dari kejadian save `best.pt` termasuk fitness seri, dan tanpa epoch duplikat dari final validation. W&B manual, project `skripsi-inatrc-yolo26`, Secrets Kaggle/Colab, dan artifact final tetap tersedia.

Snapshot AST konfigurasi sembilan notebook sebelum/sesudah membuktikan `FULL_PROTOCOL`, `SMOKE_PROTOCOL`, `DATASET_PROTOCOL`, `KD_PROTOCOL`, dan `TRACKING_PROTOCOL` identik. Seluruh hash cell implementasi KD identik. Pada `EVALUATION_PROTOCOL` hanya `save_json` wajib menjadi nonaktif secara default; `make_config` menyimpan opsi COCO/ZIP yang dipilih. Ini hanya mengubah export/evaluasi sekunder.

## Sebelum dan sesudah

### Kompleksitas yang dapat dibaca

| Notebook | Baris kode sebelum → sesudah | Definisi fungsi/kelas sebelum → sesudah |
| --- | --- | --- |
| 01 Teacher m | 2.077 → 2.027 | 81 → 76 |
| 02 Baseline s | 2.077 → 2.027 | 81 → 76 |
| 03 Baseline n | 2.077 → 2.027 | 81 → 76 |
| 04 Native s | 2.299 → 2.167 | 87 → 81 |
| 05 Native n | 2.299 → 2.167 | 87 → 81 |
| 06 CrossKD s | 2.422 → 2.372 | 104 → 99 |
| 07 CrossKD n | 2.422 → 2.372 | 104 → 99 |
| 08 CSAKD s | 2.431 → 2.381 | 106 → 101 |
| 09 CSAKD n | 2.431 → 2.381 | 106 → 101 |

Semua tetap 16 code cell dengan heading Bahasa Indonesia, fungsi sebelum pemanggilan, komentar singkat, dan satu pemanggilan eksperimen. Pengurangan terbatas ini disengaja; validasi dan provenance yang diperlukan tetap terlihat.

### Pengukuran aktual CPU lokal

Windows, Python 3.12.12, Torch 2.10.0 CPU, Ultralytics 8.4.155. Tidak ada pengukuran GPU Kaggle atau pemasangan paket melalui jaringan.

| Pengukuran | Sebelum | Sesudah | Batas interpretasi |
| --- | --- | --- | --- |
| Pemeriksaan dataset setelah run: fixture **sintetis** 3.250 gambar 256 px, 3 pasangan bergantian, cache disk hangat | 5,952 / 5,839 / 6,034 detik; median **5,952** | 3,689 / 3,728 / 3,641 detik; median **3,689** | Fingerprint sama. Ini biaya pemeriksaan akhir saja; audit awal tetap lengkap. Bukan InaTRC nyata. |
| Seluruh proses notebook baseline n: 20 gambar sintetis, 1 epoch, 160 px, batch 2; 3 pasangan pada seed tetap | 9,701 / 8,332 / 8,008 detik; median **8,332** | 7,638 / 8,189 / 6,979 detik; median **7,638** | Pip dilewati harness. Import, filesystem, dan beban CPU bervariasi; bukan estimasi kecepatan training GPU. |
| Training dalam tiga pasangan tersebut, median | 3,121 detik | 3,236 detik | Algoritme training tidak dipercepat/diubah; variasi waktu tetap ada. |
| Probe model acak, median | 0,353 detik | Tidak dijalankan | Probe tetap tersedia pada CLI/test. |
| ZIP smoke, median | 0,648 detik | Tidak dibuat secara default | Bisa diaktifkan kembali; ukuran checkpoint full berbeda. |
| Setup ulang dengan metadata paket lokal nyata yang cocok | Sebelumnya selalu memanggil pip dry-run | **0 panggilan pip**, 0,9 detik pemeriksaan metadata | Waktu setup lama/jaringan Kaggle tidak diukur. Graph transitif dan hash versi/lokasi Torch diperiksa. |

Pada smoke 20 gambar, audit penuh menjadi 1 kali + 1 pemeriksaan seluruh byte, dari sebelumnya 2 audit penuh. Pemanggilan pemeriksaan integritas gambar lokal turun 60 → 20; hashing image tetap 60 pemanggilan karena loader/evaluasi/final tetap memverifikasi isi. Decode gambar untuk training/inference sendiri tetap diperlukan. Pemeriksaan GPU nyata menjadi sekali; probe dan ZIP default hilang dari jalur run.

Pembacaan pertama fixture besar memerlukan 84,5 detik, sedangkan pembacaan berikutnya jauh lebih singkat karena cache disk. Angka pertama tersebut **tidak** dipakai sebagai pembanding percepatan. Perbandingan pada tabel memakai tiga pasangan cache hangat. Pengukuran sembilan smoke berbarengan dengan regression test juga tidak dipakai untuk menghitung percepatan sebelum/sesudah.

Nilai primary metric pada seluruh pasangan baseline smoke sama (nol untuk prediksi model acak), hash state awal sama, checkpoint lima kelas berhasil dimuat ulang, dan best epoch tetap 1. Karena metrik nol belum membuktikan seluruh evaluator, unit test tambahan menggunakan hasil nonnol untuk memastikan primary metric/threshold/AP/complexity identik ketika opsi COCO dinyalakan/dimatikan; implementasi penghitungan metrik utama tidak diubah.

Snapshot/profil/log mentah tersedia lokal di `outputs/standalone-optimization/`: `before.json`, `after.json`, `audit-comparison.json`, `baseline-timing.json`, `actual-setup.json`, serta log setiap run. Hasil tidak dimasukkan ke Git; folder checkpoint sementara smoke dibersihkan setelah pemeriksaan.

## Hasil pengujian aktual

| Pemeriksaan | Hasil |
| --- | --- |
| Regresi seluruh repository, Python 3.12, tanpa smoke | **436 passed, 3 skipped, 19 deselected**, 68,32 detik |
| Notebook/runtime/data/COCO/tracking, Python 3.13, tanpa smoke | **232 passed, 10 deselected**, 29,39 detik |
| Sembilan notebook hasil generator akhir, CPU smoke mandiri di luar checkout | **9 passed, 66 deselected**, 117,36 detik |
| Baseline n dengan COCO + ZIP aktif | **1 passed**, 12,77 detik |
| Baseline n tanpa COCO, import `faster_coco_eval` diblokir | **1 passed**, 8,77 detik |
| Setup pada metadata dependency lokal nyata yang sudah cocok, subprocess pip dilarang oleh guard test | Lolos; dependency transitif valid, Torch sebelum/sesudah sama, tidak ada install |
| Snapshot protokol dan hash algoritme sembilan notebook | Identik kecuali pilihan export/evaluasi sekunder di atas |
| `uv lock --check`, generator `--check`, `git diff --check` | Lolos |

Test baru hanya melindungi perubahan: fast path setup/dependency transitif, hash fresh tanpa decode ulang, perubahan source dengan mtime/ukuran tetap, proteksi loader terhadap perbaikan gambar sumber, opsi export/evaluator, seed teacher/student, dan profil panggilan yang redundant. Test lama tetap memeriksa ABI/Python 3.13, Secrets (mock), label/gambar bermasalah, teacher, KD-only gradient student/projector, adapter pada optimizer, hook, best epoch/fitness seri, serta final validation tanpa duplikasi.

Smoke memakai proses `python -I` di system-temp, hanya membawa file notebook, memblokir import `inatrc_kd`, dan memakai dependency terpin yang sudah tersedia. **Pip jaringan dilewati harness**, bukan diklaim teruji. Semua sembilan run menghasilkan `best.pt`, `last.pt`, `results.csv`, satu baris history/manifest, resolved config, metadata, primary metrics/AP/complexity, serta checkpoint audit; masing-masing 5 update optimizer dan `best_epoch=1`. KD teacher frozen/eval dan state tidak berubah. Checkpoint dimuat ulang untuk val tanpa teacher/projector. Source dataset tidak mendapat cache. COCO/ZIP tidak dibuat pada default; keduanya terbukti berjalan ketika diaktifkan.

| Student | SHA-256 state awal smoke baseline/Native/CrossKD/CSAKD, seed 42 |
| --- | --- |
| YOLO26s | `3e6fdba7b4a148854f51b4a3f63287cd8453b1a370142bdfa82258b1a7c82c69` |
| YOLO26n | `0a7376620c862bd25112b1d862b6ad34d4afd22b780fe7d3ee6ac826fecff046` |

Ini kesamaan inisialisasi **smoke**, bukan bukti hasil pretrained COCO full. Hash full aktual akan dicatat saat pengguna menjalankannya. Tiga test skipped adalah pemeriksaan yang memerlukan CUDA, yang tidak tersedia pada environment CPU lokal.

Perintah yang dijalankan setelah refactor:

```powershell
uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest -m "not smoke" -q --basetemp outputs/standalone-optimization/regression-final --junitxml outputs/standalone-optimization/regression-final.xml

$env:UV_PROJECT_ENVIRONMENT = "outputs/python313-venv"
uv run --frozen --no-sync --offline --cache-dir .uv-cache --python 3.13 pytest tests/test_standalone_runtime.py tests/test_standalone_notebooks.py tests/test_data.py tests/test_coco.py tests/test_config_tracking.py -m "not smoke" -q --basetemp outputs/standalone-optimization/python313-final
Remove-Item Env:UV_PROJECT_ENVIRONMENT

uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py -m smoke -q --basetemp outputs/standalone-optimization/final-smoke --junitxml outputs/standalone-optimization/final-smoke.xml

$env:TEST_OPTIONAL_COCO = "1"
$env:TEST_OPTIONAL_ZIP = "1"
uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py -m smoke -k 03_baseline -q --basetemp outputs/standalone-optimization/optional-smoke
Remove-Item Env:TEST_OPTIONAL_COCO
Remove-Item Env:TEST_OPTIONAL_ZIP

uv run --frozen --no-sync --offline --cache-dir .uv-cache pytest tests/test_standalone_notebooks.py -m smoke -k 03_baseline -q --basetemp outputs/standalone-optimization/no-coco-smoke
uv lock --check --offline --cache-dir .uv-cache
uv run --frozen --no-sync --offline --cache-dir .uv-cache python scripts/generate_standalone_notebooks.py --check
git diff --check
```

Percobaan unit pertama mendapat tiga kegagalan fixture/ekspektasi lama setelah API audit dan jalur pip berubah; fixture diperbarui, kemudian seluruh regresi akhir lolos. Test fixture Git memakai repository temporer lokal, tanpa GitHub/remote/push. Sandbox Windows memerlukan eksekusi lokal yang disetujui untuk system-temp/MSYS.

## Menjalankan teacher dan student di Kaggle

1. Pilih satu dari [sembilan notebook pada README](../README.md#notebook-mandiri-untuk-kaggle-dan-colab). Upload file `.ipynb`, aktifkan GPU/Internet dan Add Input InaTRC. Tidak perlu clone, uv, venv, source repository, atau generator.
2. Untuk diagnosis awal pilih **03 Baseline n**, `MODE = "smoke"`, `DEVICE = "0"`, W&B disabled. Run All menjalankan satu eksperimen. Jika dependency aktif berbeda dari yang telah di-import, restart session sesuai pesan, lalu Run All.
3. Untuk teacher gunakan **01**, `MODE = "full"`, `SEED = 42`, dan path InaTRC bila deteksi otomatis ambigu. Periksa konfigurasi penuh sebelum Run All; simpan `weights/best.pt` melalui Save Version. Ini perintah untuk pengguna, bukan training yang dijalankan saat refactor.
4. Baseline menggunakan **02/03**, `MODE = "full"`, seed student yang dipilih. KD menggunakan **04–09** dengan `TEACHER_CKPT` menunjuk **checkpoint teacher yang sama** dari Kaggle Input. Isi `TEACHER_SHA256` sama pada seluruh KD bila ingin guard eksplisit. CrossKD/CSAKD tetap pilot; tentukan bobot/protokol pilot sebelum hasil full dianggap final.
5. Gunakan `EVALUATE_COCO_AREA` yang sama antar-metode. Jika rumusan masalah memerlukan AP-S/M/L, set True **sebelum setup**, agar dependency opsional dipasang. `CREATE_RESULTS_ZIP` hanya pilihan download, tidak memengaruhi metrik. Rerun cell setup/GPU bila dependency atau device diubah.
6. Logging lokal selalu terlihat per epoch, termasuk `is_best` dan `best_epoch`. Akhir run menampilkan path checkpoint dan CSV. Save Version menyimpan folder `/kaggle/working/outputs`; ZIP dapat diaktifkan bila dibutuhkan. Untuk W&B online tambahkan `WANDB_API_KEY` di Kaggle Secrets dan aktifkan akses, tanpa menulis nilainya dalam cell.

Colab memakai notebook yang sama, GPU runtime, path dataset/checkpoint, dan Secrets Colab. Blok Drive tetap dikomentari; uncomment hanya jika diperlukan. Simpan/download output sebelum sesi berakhir.

## Risiko dan batas yang masih ada

GPU Kaggle/Colab, instalasi pip pada image platform nyata, InaTRC asli, W&B online, AMP/CUDA, full training, dan kualitas distilasi belum diverifikasi pada pekerjaan ini. CPU smoke membuktikan integrasi dan artifact; tidak membuktikan peningkatan akurasi atau durasi GPU. Tidak ada full training, multiseed, evaluasi test set, commit, remote baru, atau push.

Kebocoran temporal/perseptual InaTRC belum diperiksa. CrossKD/CSAKD masih adaptasi YOLO26 dengan konfigurasi pilot; audit paper–kode dan keputusan desain tetap ada pada [desain Tahap B](stage_b_design.md). Latency tetap perkiraan, bukan benchmark deployment. Checkpoint final Ultralytics dapat kehilangan optimizer state sehingga resume memerlukan checkpoint training yang masih lengkap.

Penghapusan manifest/provenance, pengurangan hash sumber, cache audit antar-run, atau perubahan algoritme ditolak karena manfaatnya belum terbukti dan dapat mengurangi transparansi penelitian.
