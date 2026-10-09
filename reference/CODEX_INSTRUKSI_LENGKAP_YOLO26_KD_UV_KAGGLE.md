# Instruksi Lengkap untuk Codex — Repository Skripsi YOLO26 Knowledge Distillation

**Peran:** Anda adalah software engineer/research engineer untuk penelitian skripsi object detection. Bangun repository yang rapi, mudah dipahami dan dikonfigurasi, reproducible, dan siap saya unggah sendiri ke GitHub lalu jalankan di Kaggle GPU gratis. **Jangan overengineering.**

## Berkas dan sumber yang WAJIB dibaca terlebih dahulu

1. Notebook yang saya lampirkan: `02-inatrc-yolo26n-baseline-full-wandb-audit(1).ipynb`. Notebook YOLO26n non-KD ini **sudah berhasil dijalankan di Kaggle**. Jadikan referensi *working pipeline* untuk audit dataset, training, validasi, W&B, checkpoint, dan laporan. **Refactor secara hati-hati, jangan menyalin sel notebook mentah-mentah.**
2. Paper CrossKD, **Cross-Head Knowledge Distillation for Object Detection**, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf
3. Kode resmi CrossKD: https://github.com/jbwang1997/CrossKD
4. Paper CSAKD, **Knowledge Distillation via Cross Supervising with Attention for Remote Sensing Object Detection**, BMVC 2025: https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf
5. Implementasi penulis CSAKD + adaptasi CrossKD pada YOLOv8: https://github.com/KefanZhan/YOLOv8-KD
6. Native KD Ultralytics: https://docs.ultralytics.com/guides/knowledge-distillation/ dan https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py
7. Referensi arsitektur YOLO26: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml

**Aturan ilmiah:** paper asli menentukan mekanisme algoritme; repositori resmi menunjukkan implementasi penulis; repo YOLOv8 menjadi referensi integrasi, bukan kode yang bisa langsung ditempel pada YOLO26. Jika ada perbedaan atau kekurangan informasi, catat sebagai pertanyaan terbuka. Jangan mengklaim reproduksi yang belum diuji.

## Konteks dan rancangan penelitian

- Penelitian skripsi nonimplementatif analitik-komparatif, **KD-only**, tanpa pruning/quantization/arsitektur baru.
- Dataset **InaTRC**, deteksi 5 golongan kendaraan; gunakan dataset/split `train`, `val`, `test` yang sama dari notebook. Jangan mengunduh, mengubah, merelabel, membuang, atau membagi ulang data tanpa persetujuan.
- **Teacher:** YOLO26m, fine-tune sekali pada InaTRC; validasi dan simpan `best.pt`; gunakan checkpoint yang sama dan *frozen* pada seluruh KD.
- **Student:** YOLO26s dan YOLO26n.
- **4 kondisi training per student:** tanpa KD, Native KD Ultralytics, CrossKD, CSAKD. Baseline teacher terpisah sebagai referensi performa, bukan kondisi student.
- **Seed tahap awal:** 42, 123, 3407. Target nanti bertambah 2 seed menjadi 5 seed. **Satu teacher + (2 student × 4 kondisi × 3 seed) = 25 training** (tidak termasuk smoke test). Saat 5 seed: **41 training**. Jangan jalankan otomatis.
- Untuk setiap ukuran student dan seed, kontrol tanpa KD dan metode KD **dimulai dari pretrained student yang setara**, bukan checkpoint baseline yang telah selesai fine-tuning (kecuali saya secara eksplisit mengubah desain). Protokol data/evaluasi/fairness dikunci; simpan setiap perubahan konfigurasi.
- Teacher boleh dilatih sekali untuk seluruh percobaan. Laporkan bahwa inferensi/statistik multiseed dilakukan **kondisional terhadap satu checkpoint teacher tetap**.

## Audit notebook baseline: perilaku yang perlu dipertahankan

Notebook referensi menggunakan YOLO26n pretrained, versi referensi Ultralytics `8.4.155`, W&B `0.28.1`, `imgsz=640`, `epochs=100`, `batch=16`, `patience=25`, AdamW (`lr0=0.001`, `weight_decay=0.0005`), AMP, seed 42, serta W&B logging manual. Ini **baseline referensi, bukan larangan penyesuaian**. Jangan mengunci versi lama tanpa mengecek apakah mendukung Native KD dan YOLO26. Jika perlu pembaruan, pilih versi yang diuji, dokumentasikan alasannya, dan jalankan tes regresi baseline.

Notebook melakukan audit lokasi dataset dan tiga split, memastikan pasangan image/label, bounding box YOLO lima kolom, lima kelas, distribusi, label kosong, hash/fingerprint, duplikasi identik lintas split, serta asumsi jumlah gambar 3.250. Audit ini harus dipertahankan secara transparan. **Jika ditemukan anotasi polygon/format lain, laporkan dengan nama berkas dan fail clearly atau sediakan konversi eksplisit yang disetujui; jangan diam-diam skip/menghapus label.** Jadikan asumsi 3.250 gambar dapat dikonfigurasi/terverifikasi menurut versi dataset. Jangan mengklaim tidak ada kebocoran temporal hanya karena tidak ada hash identik.

Notebook menggunakan validasi `best.pt` pada `val` saja, metrik agregat dan AP per kelas, kurva, latensi perkiraan, metadata environment, checkpoint audit, serta W&B artifacts. Pertahankan hasil yang substansial, sederhanakan kode yang berlebihan. Unggah **best.pt dan metrik penting** ke W&B atau output Kaggle; **unggahan checkpoint tiap epoch harus `false` secara default** dan opsional. `last.pt` boleh dipertahankan untuk resume, bukan wajib diunggah tiap epoch. Hindari *duplicate* W&B integrations.

## Bebaskan desain repository, tapi penuhi batasan ini

Anda bebas menentukan jumlah dan nama modul sesuai best practice, **tidak harus meniru struktur folder yang saya berikan sebelumnya**. Minimal pisahkan dengan jelas:

- **Konfigurasi** (`configs/`): dataset/path, training, KD dan experiment tracking; YAML/TOML cukup. Ada default yang jelas dan CLI override untuk student, metode, seed, dan mode.
- **Kode Python** (`src/` atau paket ekuivalen): training teacher, student, KD, dataset audit, evaluation, rekap eksperimen. Jangan taruh algoritme KD penuh dalam notebook.
- **Kaggle runner** (`notebooks/`): satu notebook saja dengan sel konfigurasi yang mudah diubah, setup repo, jalankan perintah, tampilkan path `best.pt` dan ringkasan hasil. `Run All` **secara default hanya smoke test**, bukan full training.
- **Tests**: tes kecil pada logika penting seperti hook lifecycle, feature shapes, finite loss, gradient routing, checkpoint reload.
- **README**: panduan singkat local setup, Kaggle, pilihan run dan cara menyimpan hasil.

Jangan membuat framework generik, registry dinamis, banyak class wrapper yang tidak perlu, database, dashboard, Docker, workflow cloud, hyperparameter sweeps, atau menduplikasi seluruh paket Ultralytics. Abstraksi dibuat **hanya jika memecahkan kebutuhan nyata**.

## `uv` wajib untuk Python environment

- Siapkan `pyproject.toml` dan **commit-ready `uv.lock`**, serta `.python-version` bila membantu kompatibilitas. Jelaskan perintah local Windows PowerShell dan Linux: `uv sync --frozen` lalu `uv run ...` (tidak wajib aktivasi venv manual). `uv` membuat `.venv` lokal, bukan memasukkannya ke Git.
- `uv` digunakan sebagai manajer dependensi + virtual environment; tidak perlu Poetry/Conda/pipenv. Untuk Kaggle, `uv` bisa di-bootstrap saat notebook berjalan, kemudian gunakan perintah dengan interpreter environment yang eksplisit (`uv run ...`, bukan secara keliru memanggil Python kernel lain).
- **Perhatian Kaggle CUDA/PyTorch:** jangan otomatis mengganti paket PyTorch berakselerasi GPU dengan CPU wheel atau mengunduh CUDA wheel yang sangat besar tanpa perlu. Periksa Python, Torch, CUDA, GPU, `torch.cuda.is_available()` dan kesesuaian dependency. Jika perlu memakai paket PyTorch preinstalled Kaggle, rancang strategi `uv` yang jelas dan terdokumentasi (termasuk `--system-site-packages`/pengecualian paket bila kompatibel). Uji dengan GPU Kaggle; jangan menjamin sebelum diuji.
- Pastikan dependensi terkunci cukup untuk reproduksibilitas, tetapi jangan mengklaim lock lintas environment menjamin kompatibilitas CUDA. Tampilkan versi aktif ke metadata run.

## GitHub-ready, tetapi SAYA yang push

- Buat `.gitignore` untuk `.venv/`, `.env`, secrets, `.ipynb_checkpoints/`, `wandb/`, `runs/`, `outputs/`, datasets besar, `.pt`, cache, log, dan artefak training. Jangan mengabaikan notebook, `README.md`, configs nonrahasia, `pyproject.toml` atau `uv.lock`.
- Sediakan contoh `.env.example` hanya bila perlu, tanpa credential asli. W&B `WANDB_API_KEY` dari Kaggle Secrets atau variabel environment; jangan dicetak.
- Jangan membuat GitHub repo, mengatur remote, meminta token GitHub, melakukan `git push`, atau mengunggah source/checkpoint otomatis. Saya akan push sendiri. Siapkan tree yang siap di-commit; beri saya instruksi singkat `git init`, `git add`, `git commit`, `git remote add`, `git push` untuk dijalankan manual (jika repo belum punya Git).
- Di Kaggle, notebook boleh clone URL repository yang saya konfigurasi atau memakai snapshot Kaggle Input bila akses internet nonaktif. Simpan commit SHA yang dijalankan agar hasil dapat dilacak.

## Kaggle notebook UX yang saya inginkan

Bagian atas notebook menyediakan pilihan sesederhana berikut (nama parameter boleh diperbaiki):

```python
STAGE = "student"         # teacher / student
STUDENT = "s"             # s / n; diabaikan untuk teacher
METHOD = "none"           # none / native / crosskd / csakd
SEED = 42
MODE = "smoke"           # smoke / full; default smoke
DATASET_ROOT = "..."      # Kaggle Input, dapat ditemukan atau diset eksplisit
TEACHER_CKPT = "..."      # Kaggle Input/W&B artifact; diperlukan untuk KD
```

Notebook harus memiliki cell untuk: (a) membaca repo yang di-clone pada commit yang dipilih; (b) memeriksa dependensi/GPU dan login W&B secara aman; (c) memilih config; (d) preflight validasi dataset/model/teacher; (e) menjalankan **satu** eksperimen sesuai pilihan; (f) menampilkan ringkasan dan menyimpan output.

Di Kaggle gunakan path yang sesuai `/kaggle/input` (read-only) dan `/kaggle/working/outputs` (writable). `best.pt`, `metrics.json`/CSV, resolved config, metadata/commit, dan file `results.csv` disimpan per run dengan nama unik **student+method+seed** supaya tidak tertimpa. Untuk checkpoint teacher antar-sesi, dokumentasikan cara memakai Kaggle Input / artifact W&B. Ingat output Kaggle perlu disimpan lewat Save Version/dataset atau W&B; filesystem sesi tidak permanen.

W&B config proyek/nama run harus dapat diubah, tanpa hardcoded identitas pengguna. Yang penting ada metrik per epoch, final val metrics, AP per kelas, status sukses/gagal, informasi GPU/VRAM/durasi, lokasi artifact `best.pt`; upload semua checkpoint per epoch opsional dan default mati. W&B tidak boleh membuat dua run duplikat untuk satu eksperimen.

## Adaptasi KD yang benar secara ilmiah

**Native:** gunakan API KD resmi Ultralytics (`distill_model` dan parameter terkait) jika versi yang dipilih memang mendukungnya. Audit sebelum implementasi; jangan menulis ulang Native KD.

**CrossKD:** review paper dan source resmi. Intinya memakai **fitur student yang diteruskan ke sebagian head teacher** lalu melakukan distilasi prediksi. Kode `OtherKD/CrossKD.py` pada repo YOLOv8 hanyalah sebagian implementasi, karena routing cross-head ada di `tasks.py`/`trainer.py`. Harus ada jalur gradien dari KD loss melalui *frozen teacher head* menuju fitur student (bobot teacher tetap frozen; jangan membungkus seluruh jalur cross-head dengan `no_grad()`). Perhatikan alignment kanal, decoding bboxes, dan cabang head yang dipakai.

**CSAKD:** review paper dan implementasi `OtherKD/CSAKD.py`, termasuk kebutuhan **student features, teacher features, dan cross_features** yang dihasilkan melalui routing lain. Periksa channel/spatial compatibility untuk YOLO26m→s/n; jika perlu adapter training-only, pastikan parameternya ikut optimizer. Jangan diam-diam mengganti CSAKD menjadi MSE/attention loss generik yang berbeda.

**YOLO26:** introspeksi runtime untuk fitur P3/P4/P5, channels dan detection head. Konfigurasi referensi YOLO26 memiliki `reg_max=1` dan `end2end=True` (jalur one-to-many/one-to-one). Jangan menyalin asumsi YOLOv8 yang memakai DFL/multiple bins atau indeks layer hardcoded. Penentuan layer dari graph/Detect inputs lebih aman; lakukan shape print dan smoke test pada m, s, dan n. **Jangan memodifikasi arsitektur student untuk inferensi** demi KD; adapter/routing training-only harus tidak menambah beban inference student.

Gunakan ekstensi/subclassing terisolasi terhadap Ultralytics jika memang dibutuhkan; **jangan mengedit site-packages atau mem-fork seluruh Ultralytics sebagai pilihan pertama**. Jika API internal tidak memungkinkan dan patch minimal mutlak diperlukan, laporkan solusi, risikonya, dan meminta persetujuan.

Untuk custom KD, sebelum dianggap siap, buktikan: forward pada satu batch, loss finite, nonzero grad student/adapter, teacher frozen, tidak ada leak tensor via hooks, AMP/GPU bila tersedia, `best.pt` tersimpan, reload dengan `YOLO(best.pt)` dan validasi student tanpa teacher. **Paper fidelity > sekadar kode berjalan.**

## Protokol ilmiah, evaluasi, dan penghematan Kaggle

- Pisahkan mode `smoke` (batch/epoch sedikit untuk diagnosis, output diberi label nonfinal) dari mode `full` (config penelitian final). Jangan memakai angka smoke sebagai hasil skripsi.
- Pilih satu protokol mAP: `mAP50-95` sebagai utama, `mAP50`, precision, recall, **AP tiap 5 kelas**. Pastikan pilihan `nms` / e2e dan hyperparameter validasi tetap sama untuk semua model; jangan secara tak sengaja bandingkan jalur prediksi berbeda.
- Laporkan parameter dan FLOPs (**kompleksitas**) terpisah dari latency, VRAM, durasi training (**biaya/efisiensi**). Metode KD training-only normalnya tidak mengurangi parameter student jika arsitektur student tetap.
- **Jangan gunakan test set saat memilih model, debugging, atau tuning.** Hanya val selama development; uji test final setelah konfigurasi dibekukan dan atas persetujuan saya.
- Perbandingan kondisi harus fair: split, augmentasi, ukuran input, budget training/early stopping, seed, inisialisasi, dan prosedur evaluasi dikontrol/dicatat. Jika batch YOLO26m perlu dikecilkan karena OOM, beri catatan bahwa kondisi teacher berbeda; jangan diam-diam mengubah baseline student.
- Tahap 3 seed adalah **eksploratif**. Simpan hasil per seed, rata-rata, simpangan baku, selisih terhadap baseline, dan confidence/effect-size sesuai desain yang memungkinkan. Jangan mengklaim p-value signifikan hanya karena tiga/lima seed; statistik final diputuskan kemudian.
- Bila muncul mismatch versi, data, channel, CUDA OOM, atau output loss, **fail loudly** dengan informasi yang cukup untuk debugging; tidak boleh silently skip image/label atau menonaktifkan KD tanpa pemberitahuan.

## TAHAP PENGERJAAN (WAJIB BERTAHAP)

**Tahap A — kerjakan SEKARANG:**
1. Audit notebook dan sumber KD; tulis ringkasan bagian yang dipertahankan, bagian yang direfactor, dan gap teknis untuk YOLO26.
2. Pilih struktur repo minimal berdasarkan kebutuhan nyata; buat `pyproject.toml`, `uv.lock`, `.gitignore`, configs, source package, README, Kaggle notebook.
3. Implementasikan dataset audit, teacher YOLO26m dan **dua baseline tanpa KD** (s/n), serta evaluasi `best.pt`/W&B; jangan jalankan full training. Siapkan satu smoke test ringan/sintetis dengan ukuran kecil, dan tes import, config, GPU check, serta dataset preflight bila input tersedia.
4. Buat jalur konfigurasi KD/native dan stub/modul terisolasi untuk CrossKD/CSAKD **jika diperlukan**, tetapi **JANGAN mengklaim CrossKD/CSAKD sudah terimplementasi** sebelum paper fidelity dan tests terpenuhi. Bila belum diimplementasikan, pilihannya harus memberikan `NotImplementedError` yang jelas, bukan diam-diam menjalankan baseline.
5. Jelaskan bagaimana mengaktifkan Native KD nanti; jika Anda mengimplementasikannya sekarang, uji sebatas smoke test setelah dependensi terverifikasi.
6. **Berhenti**, laporkan tree repo, perintah setup local PowerShell dan Kaggle, cara menjalankan satu smoke run, tests yang benar-benar dieksekusi beserta hasilnya, masalah yang belum selesai, dan apa yang perlu saya approve. Jangan melanjutkan full training, multiseed, atau adaptasi custom KD tanpa instruksi saya.

**Tahap B (nanti, setelah persetujuan):** implementasi dan pilot Native KD, CrossKD, CSAKD berdasarkan paper; validasi bentuk tensor/gradients/VRAM/checkpoint pada s lalu n; bandingkan dengan baseline.

**Tahap C (nanti, setelah persetujuan):** training final 3 seed; evaluasi dan analisis awal; kemudian tambah 2 seed jika biaya Kaggle memungkinkan. Tidak perlu rerun seed lama jika protokol tetap.

## Definisi selesai untuk Tahap A

Repo siap saya push sendiri ke GitHub; README ringkas dapat diikuti oleh pengguna baru; `uv` membuat venv, lockfile tercatat; satu notebook Kaggle mampu clone/memakai snapshot, setup dan memanggil package dengan konfigurasi terpisah; baseline/teacher script ada; dataset audit aman dan tidak mengubah sumber; output minimal `best.pt` + metrics + resolved config untuk run training sesungguhnya; W&B optional-safe dan secret tidak bocor; smoke test terbukti bekerja di lingkungan yang tersedia, **jelas sebutkan jika pengujian GPU Kaggle belum dijalankan**; tidak ada training mahal yang otomatis dimulai.
