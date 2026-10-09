# Tahap B: keputusan adaptasi YOLO26

Tahap B dimulai setelah pengguna memilih “Mulai implementasi Tahap B”. Scope adalah implementasi dan pilot sintetis; full training, multiseed, test-set evaluation, remote Git, dan push tetap tidak dijalankan. Native tetap memakai API resmi tanpa mengganti algoritmenya.

## Sumber yang digunakan

- [CrossKD paper CVPR 2024 dan appendix](https://arxiv.org/pdf/2306.11369); repository resmi [commit 87e156a](https://github.com/jbwang1997/CrossKD/tree/87e156a46259e3a516b65fc6a049b200b9ab0d54), terutama crosskd_atss.py, crosskd_gfl.py, losses/kd_loss.py dan konfigurasi ATSS.
- [CSAKD paper BMVC 2025](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_398/paper.pdf); [kode penulis commit 3433559](https://github.com/KefanZhan/YOLOv8-KD/tree/3433559ab6c9cfdef3b4ac099f1bb2feddd42dc8), terutama OtherKD/CSAKD.py dan routing cross_forward di tasks.py.
- [Ultralytics 8.4.155 DistillationModel](https://github.com/ultralytics/ultralytics/blob/v8.4.155/ultralytics/nn/distill_model.py), trainer, Detect, ModelEMA, dan strip_optimizer pada pin yang sama.

Implementasi merupakan **adaptasi YOLO26m→s/n**, bukan reproduksi persis eksperimen paper. Audit historis Tahap A dipertahankan di audit.md; keputusan di bawah menyelesaikan pilihan implementasi lokal.

## CrossKD

Input Detect diambil dari graph. Fitur antara head student sesudah blok pertama pada cabang box/klasifikasi one-to-many ditangkap dalam forward yang sama dengan detection loss. Adapter Conv 1×1 menyamakan kanal; statistik B/H/W per kanal disejajarkan seperti align_scale kode resmi. Fitur masuk ke suffix head teacher yang frozen/eval, **dengan autograd aktif**. Target berasal dari prediction teacher atas gambar yang sama, dengan no_grad. Detection loss student dihitung sekali.

QFL memakai soft target sigmoid teacher, faktor |sigmoid(cross)−target|^gamma, gamma=1. Regresi memakai 1−GIoU karena reg_max=1, dengan confidence maksimum teacher sebagai bobot. Decode langsung l/t/r/b memakai anchor pusat 0.5 dan stride P3/P4/P5 dari graph. Tidak ada DFL/KL multi-bin atau centerness tambahan.

Pilihan adaptasi yang perlu dipertahankan saat menafsirkan hasil: split sesudah aktivasi blok YOLO, satu dari dua blok intermediate, bukan indeks stack MMDetection; cabang KD one-to-many sesuai nms=None; classification memakai mean dense B/C/anchor, regression memakai jumlah confidence, bukan avg_factor assignment ATSS; corner prediction diurutkan hanya untuk geometri KD bila distance awal negatif. Ground truth dan jalur detector tidak diubah. Routing one-to-one yang berpotensi detach ditolak oleh konfigurasi pilot.

## CSAKD

Enam pasangan module ditentukan dari graph: tiga output lateral backbone yang direferensikan neck, ditambah tiga output neck input Detect. Indeks teramati pada pin ini 4/6/10 dan 16/19/22; angka tersebut tidak di-hardcode. Pre-hook menangkap input student untuk setiap module. Adapter input menyamakan kanal dengan input teacher; module teacher frozen menerima input tersebut dan menghasilkan cross feature. Adapter output student menyamakan kanal keluaran. Student output, teacher output asli, dan cross output harus cocok spasial; mismatch gagal tanpa resize.

Default `variant: paper` mempertahankan orientasi query teacher/key cross, pembagi setelah softmax, residual tanpa pembagi dua, serta L2 norm tanpa feature normalization. AAWM dihitung per sampel dari dot product GAP dibagi sqrt(C), lalu 1−sigmoid. Loss merupakan mean batch L2 norm per sampel, dengan cross term dibobot AAWM. AGKA dan Cross Supervising dijumlahkan atas enam module.

Axis perhatian paper tidak sepenuhnya menentukan layout implementasi. Pilot memilih operasi row attention BCHW dari kode penulis: q@k.transpose(-2,-1), matriks H×H per kanal, dan d_k=C. Ini pilihan dimensi yang eksplisit; tidak diklaim sebagai global HW-token attention. Mengubahnya memerlukan protokol/varian baru.

`variant: author` disediakan sebagai pilihan eksplisit: query cross/key teacher, skala di dalam softmax, residual /2, normalisasi channel, squared mean loss, dan similarity rata-rata batch sebelum sigmoid. Ini mengikuti fungsi loss kode penulis, tetapi routing/adapter tetap adaptasi YOLO26. Nama variant dan semua bobot masuk resolved config/metadata. Tidak ada fallback MSE generik.

Bobot pilot alpha=beta=1.0 dan bobot CrossKD cls=box=1.0 adalah default diagnosis, **belum hyperparameter final InaTRC**. Paper CSAKD juga memiliki ketidakselarasan antara label tabel alpha/beta dan pasangan yang disebut di paragraf; bobot paper tidak dipindahkan otomatis. `dis=6.0` tetap khusus Native.

## Integrasi, reproducibility, dan checkpoint

Custom wrapper mewarisi lifecycle DistillationModel upstream supaya optimizer, EMA, dan stripping checkpoint mengenali student/teacher/projector. CustomKDTrainer membangun wrapper pada set_model_attributes sebelum optimizer dibangun. Tidak ada patch global atau perubahan site-packages. Semua adapter terdaftar sebelum optimizer; pada full, student tetap dimulai dari pretrained COCO yang sama dengan kontrol.

Teacher selalu eval/frozen; hash parameter/buffer dibandingkan sebelum/sesudah training. Hash initial student dicatat setelah trainer membangun model untuk memeriksa fairness antar kondisi. Tensor yang disimpan hook dibersihkan melalui finally setelah loss, termasuk kegagalan; handle dibersihkan setelah training. Perhitungan statistik, attention, decoding dan loss KD memakai FP32 di dalam AMP.

Teacher diagnosis acak mengkalibrasi BatchNorm dengan satu forward sintetis bermomentum 1, tanpa gradient/optimizer. Tanpa kalibrasi, random teacher eval dapat menghasilkan logits konstan sehingga probe CrossKD tidak menguji jalur yang bermakna. Provenance kalibrasi dicatat; teacher ini tetap ditolak untuk full. Pilot pembanding memakai satu checkpoint diagnosis tetap, seed 42, dan data sintetis identik.

Probe melakukan backward pada **loss KD saja**, dua update AdamW, dan memeriksa gradient student/adapter/P3–P5, teacher state, serta hook/cache. Smoke trainer memakai 1 epoch/160/B2/nbs2 dan menyimpan jumlah update optimizer. Final checkpoint tetap student biasa: parameter/FLOPs inference harus sama untuk semua metode pada ukuran student yang sama.

## Cara menjalankan satu diagnosis

```powershell
uv run --frozen python -m inatrc_kd preflight --custom-probe crosskd --student s --device cpu
uv run --frozen python -m inatrc_kd run --method crosskd --student s --mode smoke
uv run --frozen python -m inatrc_kd run --method csakd --student n --mode smoke
```

Di Kaggle, gunakan runner yang sama dan ganti METHOD. Untuk probe AMP GPU: `preflight --custom-probe csakd --student n --device 0 --require-gpu --probe-amp`. Jangan hilangkan `--no-sync` pada environment Kaggle. Smoke tetap diagnosis; verifikasi GPU/Kaggle dan teacher InaTRC nyata merupakan batas eksternal yang belum dipenuhi oleh pengujian CPU.
