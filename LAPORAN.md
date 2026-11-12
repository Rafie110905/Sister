# Optimasi Pipeline Pengolahan Gambar dengan Pemrosesan Paralel

## 1. Latar Belakang dan Tujuan
Pengolahan gambar melibatkan banyak operasi independen (piksel, blok, file), sehingga cocok untuk paralelisasi.
Proyek ini membangun pipeline **serial**, menganalisa bagian yang bisa dioptimalkan, memparalelkannya, lalu mengukur dan menjelaskan hasilnya.

**Tujuan:**
1. Membuat pipeline pengolahan gambar yang benar (serial).
2. Memprofil tiap tahap untuk menemukan bottleneck.
3. Memparalelkan dengan thread dan proses, lalu membandingkannya.
4. Menganalisa speedup, efisiensi, dan penyebab batasnya.
5. Memperbaiki bottleneck yang ditemukan dan mengukur dampaknya.

## 2. Lingkungan Uji
| Item | Nilai |
|------|-------|
| Sistem operasi | Windows |
| Python | 3.9 |
| Jumlah core terdeteksi | 12 (logis) |
| Library | numpy, Pillow, matplotlib |
| Data | 400 gambar sintetis 1024x1024 (byte acak, dibuat dari seed) |

Data dibuat di dalam program dari seed tetap, sehingga hasilnya reproducible dan tidak membutuhkan dataset. Tiap gambar berupa array byte mentah (data biner) berukuran 1024x1024x3.

## 3. Pipeline (per gambar)
| # | Tahap | Sifat |
|---|-------|-------|
| 1 | Load/generate | Memori |
| 2 | Resize ke 512x512 | CPU-bound |
| 3 | Grayscale | CPU-bound, ringan |
| 4 | Gaussian blur | CPU-bound (konvolusi) |
| 5 | Sobel (deteksi tepi) | CPU dan memori (numpy) |
| 6 | **Transformer mini** | CPU-bound berat (perkalian matriks) |

**Tahap transformer:** peta tepi diubah menjadi 256 patch (8x8 piksel), di-embed ke dimensi 128, lalu melewati 2 layer *encoder* (multi-head self-attention + MLP). Bobotnya acak dan tetap. Tujuannya bukan akurasi, tetapi meniru **beban komputasi** model transformer sungguhan. Untuk model asli (misalnya ViT), tahap ini bisa diganti tanpa mengubah struktur paralelnya.

Hasil akhir tiap gambar adalah satu angka (rata-rata keluaran transformer). Angka ini dipakai sebagai *sidik jari* untuk memastikan hasil paralel sama dengan serial.

## 4. Analisa Paralelisme
**Jenis:** *data parallelism*. Operasi yang sama dijalankan pada data berbeda (gambar berbeda).

**Dependensi:**
- Antar gambar: tidak ada, sehingga paralel sempurna (*embarrassingly parallel*).
- Antar tahap dalam satu gambar: berurutan (output tahap N menjadi input tahap N+1).

**Thread vs proses di Python:**
- *Thread* berbagi memori tetapi dibatasi **GIL** untuk kode Python murni. Numpy dan Pillow melepas GIL pada operasi beratnya, jadi thread tetap bisa berjalan paralel di bagian itu.
- *Proses* punya interpreter sendiri sehingga tidak terkena GIL, tetapi ada biaya pembuatan proses (startup) dan serialisasi data.

**Keputusan desain:**
| Keputusan | Alasan |
|-----------|--------|
| Gambar dibuat di dalam worker | Menghindari pengiriman array besar antar-proses |
| `chunksize` diatur | Mengurangi overhead komunikasi tiap tugas |
| `OMP_NUM_THREADS=1` | Mencegah *oversubscription*: numpy memakai banyak thread di dalam tiap proses sehingga core berebut |
| Waktu terbaik dari beberapa ulangan | Mengurangi noise sistem |
| Startup pool diukur terpisah | Memisahkan biaya sekali bayar dari waktu kerja murni |
| Verifikasi `allclose(serial, paralel)` | Memastikan paralelisasi tidak mengubah hasil |

## 5. Metrik
- **Speedup** `S = T_serial / T_paralel`. Dilaporkan dua versi: `spd_kerja` (pool sudah hidup) dan `spd_total` (termasuk startup).
- **Efisiensi** `E = S / p` (p = jumlah worker). Ideal = 1.
- **Hukum Amdahl:** `S_maks = 1 / (f + (1 - f)/p)`, dengan f = fraksi serial.
- **Karp-Flatt:** `f = (1/S - 1/p) / (1 - 1/p)`. Fraksi serial eksperimental. Jika naik seiring p, penyebabnya overhead, bukan bagian serial bawaan.
- **Titik impas:** jumlah gambar minimum agar total waktu paralel (termasuk startup) lebih kecil dari serial.

## 6. Cara Menjalankan
```bash
pip install numpy pillow matplotlib
python project.py --n 400 --size 1024 --repeat 3   # benchmark utama
python diagnostik.py                               # uji kemampuan paralel CPU (beban murni hitung)
python diagnostik_tahap.py                         # uji scaling per tahap pipeline
```
Keluaran `project.py`: tabel di terminal, `results.csv`, `stage_profile.csv`, dan `results.png`.

## 7. Hasil Eksperimen

### 7.1 Benchmark awal (sebelum perbaikan)
Profil per tahap (serial):

| Tahap | % waktu |
|-------|---------|
| transformer | 48,9 |
| sobel | 21,2 |
| resize | 13,7 |
| load | 11,9 |
| blur | 3,6 |
| grayscale | 0,7 |

Benchmark 400 gambar (serial 21,02 s):

| Mode | Worker | Startup (s) | Kerja (s) | Total (s) | Speedup kerja | Speedup total | Efisiensi | Karp-Flatt |
|------|--------|-------------|-----------|-----------|---------------|---------------|-----------|------------|
| thread | 2 | - | 13,94 | 13,94 | 1,51 | 1,51 | 0,75 | 0,326 |
| thread | 4 | - | 12,89 | 12,89 | 1,63 | 1,63 | 0,41 | 0,484 |
| thread | 8 | - | 13,99 | 13,99 | 1,50 | 1,50 | 0,19 | 0,618 |
| thread | 12 | - | 14,26 | 14,26 | 1,48 | 1,48 | 0,12 | 0,649 |
| proses | 2 | 1,09 | 14,23 | 15,31 | 1,48 | 1,37 | 0,74 | 0,353 |
| proses | 4 | 1,02 | 12,42 | 13,44 | 1,69 | 1,56 | 0,42 | 0,454 |
| proses | 8 | 1,64 | 14,02 | 15,66 | 1,50 | 1,34 | 0,19 | 0,619 |
| proses | 12 | 2,27 | 14,45 | 16,72 | 1,45 | 1,26 | 0,12 | 0,659 |

Semua baris menghasilkan keluaran yang sama dengan serial (`benar = True`).

**Temuan:** speedup mentok di sekitar 1,7x dan **turun** saat worker ditambah. Startup proses di Windows naik dari 1,0 s (4 worker) menjadi 2,3 s (12 worker).

### 7.2 Tes diagnostik: batas perangkat keras
`diagnostik.py` menjalankan beban murni hitung (tanpa gambar, tanpa memori besar). Hasil dua kali jalan:

| Worker | Speedup | Efisiensi |
|--------|---------|-----------|
| 2 | 1,85 / 1,95 | 0,93 / 0,98 |
| 4 | 3,07 / 3,15 | 0,77 / 0,79 |
| 6 | 3,53 / 3,89 | 0,59 / 0,65 |
| 8 | 3,99 / 4,01 | 0,50 / 0,50 |
| 12 | 4,07 / 4,07 | 0,34 / 0,34 |

**Temuan:** laptop mampu sekitar **4x** dan tidak naik lagi setelah 8 worker. Ini ciri CPU dengan core campuran atau hyperthreading, ditambah batas daya, sehingga "12 core" tidak sama dengan 12x. Pipeline gambar hanya mencapai 1,7x, jadi ada selisih yang hilang di dalam pipeline sendiri.

### 7.3 Tes diagnostik per tahap: mencari bottleneck
`diagnostik_tahap.py` memotong pipeline secara kumulatif dan mengukur speedup paralel di tiap potongan (96 gambar). Speedup kumulatif sebelum perbaikan (dua kali jalan):

| Sampai tahap | spd@4 | spd@8 |
|--------------|-------|-------|
| 1. load | 1,82 / 1,83 | 2,01 / 1,62 |
| 2. resize | 2,54 / 2,60 | 2,71 / 2,52 |
| 3. grayscale | 2,65 / 2,56 | 2,85 / 2,84 |
| 4. blur | 2,99 / 2,79 | 3,25 / 3,25 |
| **5. sobel** | **2,12 / 1,95** | **1,85 / 1,75** |
| 6. transformer | 2,17 / 1,89 | 1,78 / 1,24 |

**Temuan:** speedup tumbuh wajar sampai blur (sekitar 3x, sesuai batas laptop), lalu **jatuh tepat di tahap sobel**. Kontribusi sobel sendiri (dihitung dari selisih waktu antar baris) hanya sekitar 1,1x di 4 worker dan sekitar 0,8x di 8 worker, artinya hampir tidak ikut terparalel dan di 8 worker malah lebih lambat dari serial.

**Dugaan penyebab:** sobel versi awal membuat banyak array sementara (sekitar 1 MB per array, puluhan operasi per gambar), sehingga beban bandwidth memori tinggi. Beban seperti ini cepat jenuh ketika banyak core mengakses RAM bersamaan. Hal ini konsisten dengan kontras dua tes: beban murni hitung scaling sampai 4x, sedangkan beban yang memakai memori puncaknya di 4 worker lalu turun. Ini masih dugaan yang didukung bukti tidak langsung, belum dibuktikan dengan alat profil perangkat keras.

### 7.4 Perbaikan: sobel hemat memori
Fungsi `sobel` ditulis ulang dalam bentuk *separable* dengan operasi in-place (sekitar 4 array sementara, dibanding sekitar 20). Versi lama disimpan sebagai `sobel_lama`.
- Hasil numerik sama dengan versi lama (`allclose = True`).
- Mikro-benchmark serial (512x512): 2,91 ms menjadi 1,90 ms per gambar.
- Waktu sobel di profil pipeline: 0,135-0,174 s menjadi 0,057 s per 10 gambar.

### 7.5 Hasil akhir (sesudah perbaikan)
Profil per tahap (serial):

| Tahap | % waktu |
|-------|---------|
| transformer | 50,4 |
| resize | 18,7 |
| load | 15,4 |
| sobel | 10,0 |
| blur | 4,8 |
| grayscale | 0,7 |

Benchmark 400 gambar (serial 19,88 s):

| Mode | Worker | Startup (s) | Kerja (s) | Total (s) | Speedup kerja | Speedup total | Efisiensi | Karp-Flatt |
|------|--------|-------------|-----------|-----------|---------------|---------------|-----------|------------|
| thread | 2 | - | 12,80 | 12,80 | 1,55 | 1,55 | 0,78 | 0,288 |
| thread | 4 | - | 11,14 | 11,14 | 1,78 | 1,78 | 0,45 | 0,414 |
| thread | 8 | - | 12,55 | 12,55 | 1,58 | 1,58 | 0,20 | 0,578 |
| thread | 12 | - | 13,29 | 13,29 | 1,50 | 1,50 | 0,12 | 0,638 |
| proses | 2 | 1,07 | 12,14 | 13,21 | 1,64 | 1,50 | 0,82 | 0,221 |
| proses | 4 | 0,95 | 10,17 | 11,12 | 1,96 | 1,79 | 0,49 | 0,348 |
| proses | 8 | 1,59 | 12,43 | 14,01 | 1,60 | 1,42 | 0,20 | 0,571 |
| proses | 12 | 2,11 | 12,82 | 14,94 | 1,55 | 1,33 | 0,13 | 0,613 |

Titik impas proses: 55 gambar (2 worker), 39 (4 worker), 85 (8 worker), 120 (12 worker).

Tes per tahap sesudah perbaikan:

| Sampai tahap | spd@4 | spd@8 |
|--------------|-------|-------|
| 4. blur | 2,79 | 3,12 |
| 5. sobel | 2,23 | 1,90 |
| 6. transformer | 2,23 | 1,83 |

### 7.6 Perbandingan sebelum dan sesudah
| | Sebelum | Sesudah |
|---|---------|---------|
| Sobel (profil serial) | 0,135-0,174 s | 0,057 s |
| Porsi sobel di pipeline | 18-21% | 10% |
| Serial, 400 gambar | 21,0-22,1 s | 19,9 s |
| Proses 4 worker, waktu kerja | 12,3-12,4 s | 10,2 s |
| Speedup proses 4 worker | 1,69-1,79x | 1,96x |
| Titik impas 4 worker | 48-50 gambar | 39 gambar |

Dibanding serial awal (21,0 s), hasil terbaik sekarang (10,2 s) sekitar **2,1x lebih cepat**. Angka ini gabungan dua hal: sobel yang lebih efisien dan paralelisasi.

## 8. Pembahasan
**1. Thread atau proses?**
Hasilnya hampir sama. Pada 4 worker, proses 1,96x dan thread 1,78x (waktu kerja). Pada 8 dan 12 worker keduanya sekitar 1,5 sampai 1,6x. Thread tidak jauh tertinggal karena numpy dan Pillow melepas GIL pada operasi beratnya, dan thread tidak memiliki biaya startup. Proses sedikit lebih baik di 2 dan 4 worker. Jadi GIL bukan penyebab utama terbatasnya speedup pada pipeline ini.

**2. Apakah speedup mendekati linear?**
Tidak. Puncaknya di 4 worker (1,96x), lalu melandai bahkan turun di 8 dan 12 worker.

**3. Bagaimana efisiensi berubah?**
Turun dari 0,82 (2 worker) menjadi 0,49 (4), 0,20 (8), dan 0,13 (12). Penyebabnya adalah gabungan: batas perangkat keras laptop sekitar 4x (core campuran/hyperthreading dan batas daya, terlihat pada tes diagnostik), rebutan memori, dan overhead startup proses di Windows.

**4. Bagaimana nilai Karp-Flatt?**
Naik seiring worker (proses: 0,22 menjadi 0,61). Karena fraksi serial bawaan seharusnya konstan, kenaikan ini menunjukkan **overhead dan rebutan sumber daya**, bukan sekadar bagian program yang tidak bisa diparalelkan.

**5. Apakah paralel bisa lebih lambat pada data kecil?**
Ya, dan ini terukur lewat titik impas. Proses baru mengalahkan serial setelah 39 gambar (4 worker) sampai 120 gambar (12 worker). Di bawah jumlah itu, biaya startup (1 sampai 2 detik di Windows) lebih besar dari penghematan waktunya. Titik impas naik seiring jumlah worker karena startup makin mahal.

**6. Tahap mana yang paling berat dan apa dampaknya bila dioptimalkan?**
Transformer, sekitar 50% waktu serial. Pada tes per tahap, kontribusi transformer terhadap speedup juga sedang saja (sekitar 2x). Optimasi sobel membuktikan bahwa memperbaiki satu tahap bisa mengubah hasil keseluruhan (1,7x menjadi 2x pada 4 worker), tetapi perbaikan berikutnya akan memberi dampak yang makin kecil.

## 9. Keterbatasan
- **Variasi antar-run** sekitar 5 sampai 10% (waktu serial 19,9 sampai 22,1 s pada run berbeda). Selisih kecil, terutama pada thread, tidak boleh dibaca sebagai klaim kuat.
- **Penyebab keterbatasan scaling** (memori, cache, core campuran) adalah kesimpulan dari bukti tidak langsung. Pembuktian pasti membutuhkan alat profil perangkat keras.
- Perbaikan sobel hanya **sebagian** menutup selisih speedup: pada tes per tahap, speedup di sobel masih turun (2,79 menjadi 2,23 pada 4 worker), jadi sobel bukan satu-satunya penyebab.
- Data sintetis dibuat di memori. Dengan gambar yang dibaca dari disk, tahap load menjadi lebih berat dan angkanya bisa berbeda.
- Transformer memakai bobot acak, jadi pengujian hanya mengukur beban komputasi, bukan akurasi model.

## 10. Pengembangan Lanjutan
- Ganti transformer mini dengan **ViT asli** (PyTorch/HuggingFace) dan jalankan di GPU. Paralelisasi pindah ke dalam batch.
- **Tiling:** proses sobel per blok baris agar muat di cache.
- **Pipelining:** tahap 1-5 (CPU) dan tahap 6 (GPU) berjalan bersamaan dengan antrean (producer-consumer).
- Uji **strong scaling** (data tetap, worker bertambah) dan **weak scaling** (data tumbuh seiring worker).
- Bandingkan dengan `joblib`, `numba`, atau `ProcessPoolExecutor`.

## 11. Kesimpulan
1. Pipeline pengolahan gambar bersifat *embarrassingly parallel* antar gambar, dan versi paralel menghasilkan keluaran yang sama dengan serial.
2. Pada laptop 12 core ini, speedup maksimum yang diukur pada beban murni hitung sekitar **4x**, sedangkan pipeline awal hanya mencapai **1,7x**.
3. Tes per tahap menunjukkan bahwa tahap **sobel** menghambat scaling. Penulisan ulang yang hemat memori mempercepat sobel serial sekitar 2,4 sampai 3x dan menaikkan speedup proses 4 worker dari 1,7x menjadi 2,0x.
4. Total waktu terbaik turun dari 21,0 s (serial awal) menjadi 10,2 s, sekitar **2,1x lebih cepat**.
5. Menambah worker di atas 4 tidak membantu dan justru menurunkan hasil. Startup proses di Windows (1 sampai 2,3 detik) menentukan jumlah data minimum agar paralel berguna (39 sampai 120 gambar).
6. Proses dan thread hampir setara pada pipeline ini karena numpy dan Pillow melepas GIL.

## 12. Rangkuman untuk Presentasi
- **Konsep:** pipeline gambar serial dipercepat dengan data parallelism, diuji dengan thread dan proses.
- **Kajian paralel:** independen antar gambar, berurutan antar tahap; hambatan berupa GIL (kecil), overhead startup, dan rebutan memori (dugaan terkuat).
- **Alur analisa:** benchmark awal (1,7x) -> diagnostik perangkat keras (batas 4x) -> diagnostik per tahap (sobel) -> perbaikan -> hasil 2,1x.
- **Pelajaran utama:** angka speedup rendah bukan berarti kode salah; ukur, cari tahap yang menghambat, perbaiki, lalu ukur lagi.