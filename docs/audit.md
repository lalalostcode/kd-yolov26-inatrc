# Audit working pipeline dan kompatibilitas KD — Tahap A

Spesifikasi utama dibaca penuh dari `reference/CODEX_INSTRUKSI_LENGKAP_YOLO26_KD_UV_KAGGLE.md`. Notebook `reference/02-inatrc-yolo26n-baseline-full-wandb-audit.ipynb` adalah acuan pipeline Kaggle yang sudah berhasil. Dua paper yang dilampirkan dibaca seluruhnya; persamaan perhatian CSAKD halaman 7 juga diperiksa secara visual. Ini audit awal Tahap A; setelah persetujuan pengguna, keputusan adaptasi dan implementasi custom dicatat dalam [desain Tahap B](stage_b_design.md) serta [laporan Tahap B](stage_b_report.md).

Keberhasilan notebook adalah keterangan pengguna. Semua code cell pada salinan referensi memiliki execution_count=null dan outputs kosong; versi Torch/CUDA, GPU, dataset aktual, dan hasil training tidak dapat diverifikasi dari output file tersebut. Metadata Python 3.11 adalah metadata kernel, bukan bukti runtime. Tes Tahap A memakai environment aktif yang dicatat sendiri.

## Perilaku baseline yang dipertahankan

- Dataset InaTRC, lima kelas dengan ID/nama yang sama, dan split train/val/test tetap. Sumber dataset tidak direlabel, dibagi ulang, dihapus, atau diunduh otomatis.
- Audit pasangan image/label, stem ganda, label kosong, YOLO box tepat lima kolom, kelas, finite/rentang bbox, distribusi, hash/fingerprint, serta duplikasi identik lintas split. Polygon/format lain menghasilkan kegagalan dengan nama berkas. Jumlah 3.250 gambar adalah asumsi versi dataset yang dapat dikonfigurasi.
- Protokol full awal: input 640, 100 epoch, batch 16, patience 25, AdamW lr0=0.001/weight_decay=0.0005, AMP, seed terpilih, augmentasi dan deterministic dari notebook. Semua student dimulai dari pretrained ukuran yang setara, bukan baseline yang sudah selesai fine-tuning.
- Reload dan evaluasi `best.pt` pada **val saja**, mAP50-95 utama, mAP50, precision, recall, AP tiap lima kelas, kurva, metadata environment, dan audit checkpoint. `nms=None` dipertahankan seperti baseline; mengganti jalur inference memerlukan protokol baru.
- Logging W&B manual satu run per eksperimen; best.pt dan metrik utama disimpan lokal/artifact. Upload checkpoint setiap epoch menjadi opsional dan default mati. Integrasi W&B bawaan tidak dijalankan bersamaan.

Kode notebook direfactor menjadi konfigurasi, audit data, training/evaluasi, tracking, dan preflight terpisah. Runner Kaggle hanya mengatur pilihan dan memanggil package. Smoke adalah diagnosis sintetis dengan label nonfinal; angka smoke bukan hasil skripsi. Tidak adanya hash identik tidak membuktikan bebas kebocoran temporal/perseptual.

Trainer upstream pada pin ini mencoba mengurangi batch saat OOM di epoch pertama dan memulihkan NaN dari checkpoint. Extension lokal menonaktifkan kedua jalur tersebut agar protokol tidak berubah dan optimizer tidak direstart diam-diam. Batas counter OOM upstream dipasang pada callback awal epoch sebelum forward/backward; guard tambahan mencegah rebuild pipeline. Tes dengan OOM yang diinjeksi membuktikan batch tidak berubah.

## Versi, environment, dan Native KD

Ultralytics **8.4.155** dipertahankan setelah pemeriksaan source resmi, bukan semata karena notebook lama. Tag mengarah ke commit `590427b052171d75dadd813d5f2a9d49a94dfa9d` (17 September 2026). W&B tetap 0.28.1. Source:

- [DistillationModel 8.4.155](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/nn/distill_model.py)
- [BaseTrainer 8.4.155](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/engine/trainer.py)
- [ModelEMA/strip_optimizer 8.4.155](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/utils/torch_utils.py)
- [Parameter resmi](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/cfg/default.yaml) dan [panduan Native KD](https://docs.ultralytics.com/guides/knowledge-distillation/)

Native menggunakan `model.train(distill_model=teacher_checkpoint, dis=6.0, ...)`. Loss/trainer tidak ditulis ulang. Wrapper resmi memilih input Detect melalui `Detect.f`, mendaftarkan projector 1×1–ReLU–1×1 untuk alignment kanal, menjaga teacher frozen/eval, dan menggunakan L2 fitur neck berbobot confidence dari rata-rata logits kedua cabang head. Wrapper dibangun sebelum optimizer. EMA menghapus teacher; finalisasi checkpoint menghapus hook dan menyimpan student biasa. `YOLO(best.pt)` tidak memerlukan teacher. `last.pt` selama training bisa dipakai resume; finalisasi normal upstream melakukan stripping optimizer pada best dan last.

Local uv.lock memakai Torch 2.10.0/torchvision 0.25.0 CPU untuk pengujian Windows. Kaggle secara eksplisit memakai pasangan CUDA preinstalled yang kompatibel, dengan uv environment berakses system site packages dan pengecualian instalasi Torch/torchvision. Ini **pengecualian lock runtime Torch yang dicatat**, bukan jaminan lock yang identik lintas CUDA. Preflight memeriksa Python 3.11/3.12/3.13, versi Ultralytics/W&B, pasangan resmi Torch 2.4→torchvision 0.19 hingga Torch 2.13→torchvision 0.28, path interpreter/paket aktif, GPU/VRAM, operasi tensor, torchvision NMS, dan konversi NumPy ↔ Torch pada device terpilih. Python 3.13 memerlukan Torch ≥2.7/torchvision ≥0.22 sesuai [tabel resmi torchvision](https://github.com/pytorch/vision#installation); pasangan 2.4–2.6 dibatasi sampai Python 3.12. Metadata Requires-Dist Torch/torchvision juga diperiksa terhadap distribusi aktif dengan environment markers dan extra kosong; dependency hilang atau versinya tidak sesuai, misalnya SymPy pada Torch CUDA yang lebih lama, menghentikan preflight. GPU yang diminta tetapi tidak tersedia menghasilkan kegagalan. Tidak ada pemasangan CUDA wheel otomatis. Batas Python awal Tahap A dan hasil tesnya dipertahankan sebagai catatan historis di `stage_a_report.md`.

Teacher untuk Native harus checkpoint lokal YOLO26m detection, lima kelas dengan urutan nama yang sama; hash checkpoint masuk metadata run. Teacher sintetis adalah YOLO26m acak **tanpa training**, diberi `inatrc_diagnostics_only=True` di checkpoint beserta sidecar metadata. Teacher ini hanya untuk smoke dan ditolak dalam mode full. Teacher penelitian tetap harus fine-tune InaTRC sekali dan memakai best.pt tetap yang disetujui.

## CrossKD: mekanisme dan gap adaptasi

[Paper CrossKD CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.pdf), [salinan penulis di arXiv](https://arxiv.org/pdf/2306.11369), dan [repository resmi](https://github.com/jbwang1997/CrossKD) menentukan acuan. Commit yang diaudit: **87e156a46259e3a516b65fc6a049b200b9ab0d54** (23 Juli 2023).

Fitur antara head student masuk sebagian head teacher; cross prediction meniru prediction teacher. Gradient KD melewati layer teacher frozen menuju fitur student. Membungkus seluruh cross route dengan no_grad memutus mekanisme. Paper menggunakan QFL untuk klasifikasi, GIoU untuk box langsung, dan KL untuk regresi distribusi. [Implementasi GFL](https://github.com/jbwang1997/CrossKD/blob/87e156a46259e3a516b65fc6a049b200b9ab0d54/mmdet/models/detectors/crosskd_gfl.py) melakukan alignment statistik dengan asumsi kanal setara; [konfigurasi ATSS](https://github.com/jbwang1997/CrossKD/blob/87e156a46259e3a516b65fc6a049b200b9ab0d54/configs/crosskd/crosskd_r50_atss_r101_fpn_1x_coco.py) memakai QFL/GIoU untuk box langsung.

Adaptasi YOLOv8 penulis CSAKD diaudit pada commit **3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8** (14 Mei 2026):

- [OtherKD/CrossKD.py](https://github.com/KefanZhan/YOLOv8-KD/blob/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8/OtherKD/CrossKD.py) hanya menghitung loss focal BCE dan decoded −log(IoU), berbeda dari direct-box GIoU paper.
- [trainer.py](https://github.com/KefanZhan/YOLOv8-KD/blob/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8/ultralytics/engine/trainer.py) memanggil `teacher(features, CrossKD=True)`; [tasks.py](https://github.com/KefanZhan/YOLOv8-KD/blob/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8/ultralytics/nn/tasks.py) mengimplementasikan routing.
- `forward3` membuat Conv2d baru setiap forward. Adapter tidak terdaftar permanen dan tidak masuk optimizer; pola ini tidak disalin. Trainer juga menggandakan detection loss dalam mode KD dan menjalankan student lagi untuk fitur. Perilaku tersebut perlu ditinjau terhadap fairness.

YOLO26 memiliki reg_max=1; distilasi DFL multi-bin bukan jalur yang sesuai. Posisi cross-head, alignment kanal, loss direct-box, bobot/normalisasi, dan cabang head harus diputuskan pada Tahap B. Hasil adaptasi tidak boleh disebut reproduksi persis tanpa bukti.

## CSAKD: mekanisme dan perbedaan paper/kode

[Paper CSAKD BMVC 2025](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf) dan [kode penulis CSAKD.py](https://github.com/KefanZhan/YOLOv8-KD/blob/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8/OtherKD/CSAKD.py) dibaca. Cross Supervising memakai output modul teacher atas input student, AAWM mengatur bobot dengan similarity fitur GAP, dan AGKA memberi residual attention sebelum feature alignment. Ketiga masukan student/teacher/cross features dibutuhkan; cross features dihasilkan oleh routing `Cross=True` di tasks/trainer, bukan di loss saja.

Paper meneliti teacher YOLOv8s dan student YOLOv8s hasil hybrid quantization; YOLO26m→s/n KD-only merupakan adaptasi pasangan ukuran berbeda. Repository ini tidak menambahkan quantization.

Gap terbuka yang memerlukan keputusan sebelum implementasi:

1. Paper memakai query teacher/key cross; kode membalik urutannya.
2. Persamaan 13 menempatkan pembagi skala setelah softmax; kode menempatkannya dalam softmax.
3. Kode membagi residual dengan dua, menormalisasi fitur, dan menggunakan squared mean loss; persamaan paper menampilkan residual dan L2 tanpa langkah tersebut.
4. AAWM kode mengambil rata-rata similarity seluruh batch sebelum sigmoid; dimensi dan reduksi perlu ditetapkan.
5. Attention kode memakai matmul tensor BCHW langsung. Modul yang didistilasi, dimensi/axis attention, spatial/channel alignment, serta routing graph harus ditentukan eksplisit untuk m→s/n. Adapter yang diperlukan harus terdaftar sebelum optimizer dan hanya ada selama training.

Tidak ada penggantian diam-diam menjadi MSE/attention loss generik. Pada Tahap A opsi csakd berupa NotImplementedError; kini tersedia varian paper/author yang dibedakan eksplisit dalam desain Tahap B.

## Bukti kompatibilitas dan gate berikutnya

[Arsitektur YOLO26 pada pin penelitian](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/cfg/models/26/yolo26.yaml) memiliki P3/P4/P5, reg_max=1, dan dual-head end2end. Pemilihan inference end2end terpisah dari keberadaan kedua head; nms=None mempertahankan jalur baseline. [Detect](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/nn/modules/head.py) mengembalikan dict boxes/scores/feats dan kedua cabang. Input one-to-one di-detach ketika training; cross routing melalui head training penuh dapat kehilangan gradient menuju student.

Tes sintetis package memeriksa m/s/n pada input 160, B=2, lima kelas: input head berasal dari graph, bentuk P3/P4/P5, finite detection loss, backward nonzero, dan pembersihan hook. Preflight juga menerima batch/input/AMP konfigurasi full tanpa menjalankan training. Probe Native resmi m→s/n memeriksa projector terdaftar dan masuk parameter groups AdamW, gradient student/projector finite/nonzero, teacher eval/frozen dengan hash state tetap, serta jumlah hook stabil selama dua batch dan nol setelah cleanup. GPU/AMP hanya aktif bila CUDA tersedia. Hasil yang benar-benar dijalankan dilaporkan pada laporan Tahap A; keberadaan tes bukan klaim GPU Kaggle sudah diuji.

Tahap B telah disetujui pengguna dan implementasi/pilot CPU s/n dikerjakan. AMP/GPU, VRAM, dan pilot InaTRC nyata masih membutuhkan environment serta teacher penelitian. Full training, multiseed, dan test-set final tetap menunggu instruksi terpisah. Statistik nanti kondisional terhadap satu teacher checkpoint tetap.
