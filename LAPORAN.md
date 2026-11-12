# Optimasi Pipeline Pengolahan Gambar dengan Pemrosesan Paralel

## 1. Latar Belakang dan Tujuan
Pengolahan gambar melibatkan jutaan operasi independen (piksel, blok, file). Sifat ini cocok untuk paralelisasi.
Proyek ini membangun pipeline **serial**, menganalisa bagian yang bisa dioptimalkan, lalu **memparalelkannya** dan mengukur hasilnya.

**Tujuan:**
1. Membuat pipeline pengolahan gambar yang benar (serial).
2. Memprofil tiap tahap untuk menemukan bottleneck.
3. Memparalelkan dengan thread dan proses, lalu membandingkannya.
4. Menganalisa speedup, efisiensi, dan batasnya.

## 2. Data
- **Sintetis:** gambar acak berupa array byte mentah (data biner), dibuat dari seed sehingga hasilnya reproducible dan tidak perlu dataset.
- **Asli:** folder berisi jpg/png (`--folder ./gambar`).

## 3. Pipeline (per gambar)
| # | Tahap | Sifat |
|---|-------|-------|
| 1 | Load/generate | I/O atau memori |
| 2 | Resize ke 512x512 | CPU-bound |
| 3 | Grayscale | CPU-bound, ringan |
| 4 | Gaussian blur | CPU-bound (konvolusi) |
| 5 | Sobel (deteksi tepi) | CPU-bound (numpy) |
| 6 | **Transformer mini** | CPU-bound berat (perkalian matriks) |

**Tahap transformer:** peta tepi diubah menjadi 256 patch (8x8 piksel), di-embed ke dimensi 128, lalu melewati 2 layer *encoder* (multi-head self-attention + MLP). Bobotnya acak dan tetap. Tujuannya bukan akurasi, tetapi meniru **beban komputasi** model transformer sungguhan. Untuk model asli (misalnya ViT), tahap ini bisa diganti tanpa mengubah struktur paralelnya.

## 4. Analisa Paralelisme
**Jenis:** *data parallelism*. Operasi yang sama dijalankan pada data berbeda (gambar berbeda).

**Dependensi:**
- **Antar gambar:** tidak ada, jadi paralel sempurna (*embarrassingly parallel*).
- **Antar tahap dalam satu gambar:** berurutan (output tahap N adalah input tahap N+1), jadi tidak bisa diparalelkan antar tahap tanpa teknik pipelining.

**Thread vs proses di Python:**
- *Thread* berbagi memori tetapi dibatasi **GIL**, sehingga kode Python murni tidak berjalan benar-benar paralel. Numpy dan PIL melepas GIL pada sebagian operasi, jadi hasil thread bisa lumayan, tetapi biasanya di bawah proses.
- *Proses* punya interpreter sendiri, sehingga paralel sungguhan. Biayanya lebih besar: pembuatan proses dan serialisasi data (pickle).

**Keputusan desain dan alasannya:**
| Keputusan | Alasan |
|-----------|--------|
| Gambar dibuat/dimuat **di dalam worker** | Menghindari pengiriman array besar antar-proses |
| `chunksize` diatur | Mengurangi overhead komunikasi tiap tugas |
| `OMP_NUM_THREADS=1` | Mencegah *oversubscription*: numpy memakai banyak thread di dalam tiap proses sehingga core berebut |
| Waktu terbaik dari beberapa ulangan | Mengurangi noise sistem |
| Verifikasi `allclose(serial, paralel)` | Memastikan paralelisasi tidak mengubah hasil |

## 5. Metrik
- **Speedup** `S = T_serial / T_paralel`
- **Efisiensi** `E = S / p` (p = jumlah worker). Ideal = 1.
- **Hukum Amdahl:** `S_maks = 1 / (f + (1 - f)/p)`, dengan f = fraksi yang tetap serial. Berapa pun core-nya, `S` tidak melebihi `1/f`.
- **Karp-Flatt:** `f = (1/S - 1/p) / (1 - 1/p)`. Ini fraksi serial yang *terukur*. Jika nilainya naik seiring p, penyebabnya overhead (komunikasi, startup, berebut memori), bukan bagian serial bawaan.

## 6. Cara Menjalankan
```bash
pip install numpy pillow matplotlib
python project.py                          # data sintetis
python project.py --n 120 --size 1024 --repeat 3
python project.py --folder ./gambar        # gambar asli
```
Keluaran: tabel di terminal, `results.csv`, `stage_profile.csv`, dan `results.png`.

## 7. Hasil Awal (diisi dari mesinmu)
Hasil profil dari satu uji coba (1 core, 20 gambar 768px), hanya untuk gambaran:

| Tahap | % waktu |
|-------|---------|
| transformer | ~45% |
| resize | ~23% |
| sobel | ~11% |
| load | ~11% |
| blur | ~9% |
| grayscale | ~1% |

Transformer menjadi bottleneck terbesar, jadi tahap ini yang paling berpengaruh bila dioptimalkan lebih lanjut.

**Tabel speedup (isi dari hasil `results.csv` di mesinmu):**

| Mode | Worker | Waktu (s) | Speedup | Efisiensi |
|------|--------|-----------|---------|-----------|
| serial | 1 | | 1.00 | 1.00 |
| thread | 2 / 4 / 8 | | | |
| process | 2 / 4 / 8 | | | |

## 8. Pembahasan (panduan menulis kesimpulan)
Setelah mendapat angkanya, jawab pertanyaan berikut:
1. Mana yang lebih cepat, thread atau proses? Mengapa (GIL)?
2. Apakah speedup mendekati linear? Di worker berapa mulai melandai?
3. Bagaimana efisiensi berubah saat worker bertambah? Apa penyebabnya (overhead, jumlah core fisik vs logis, memory bandwidth)?
4. Bagaimana nilai Karp-Flatt? Stabil berarti ada bagian serial bawaan, naik berarti overhead.
5. Pada ukuran data kecil, apakah paralel justru lebih lambat? (Cobalah `--n 4`.)
6. Tahap mana yang paling berat dan bagaimana dampaknya bila dioptimalkan?

## 9. Pengembangan Lanjutan
- Ganti transformer mini dengan **ViT asli** (PyTorch/HuggingFace) dan jalankan di GPU. Dengan GPU, paralelisasi pindah ke dalam batch (batching beberapa gambar sekaligus).
- **Pipelining:** tahap 1-5 (CPU) dan tahap 6 (GPU) berjalan bersamaan dengan antrean (producer-consumer).
- Bandingkan dengan `concurrent.futures.ProcessPoolExecutor`, `joblib`, atau `numba`.
- Uji **strong scaling** (data tetap, worker bertambah) dan **weak scaling** (data tumbuh seiring worker).

## 10. Rangkuman untuk Presentasi
- **Konsep:** pipeline gambar serial dipercepat dengan data parallelism.
- **Kajian paralel:** independen antar gambar (cocok untuk proses), berurutan antar tahap, kendala GIL dan overhead.
- **Progres:** kode serial + thread + proses, profiling, benchmark otomatis, verifikasi kebenaran.
