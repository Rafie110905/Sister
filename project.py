"""
PROYEK PEMROSESAN PARALEL: Pipeline Gambar + Transformer Mini
=============================================================
Tahap per gambar (tiap gambar independen -> data parallelism):
  1. load/generate  2. resize  3. grayscale  4. blur  5. sobel (tepi)
  6. transformer mini (patch embedding + self-attention + MLP)

Mode eksekusi yang dibandingkan:
  - serial
  - thread pool   (dibatasi GIL untuk kode Python; numpy/PIL kadang melepas GIL)
  - process pool  (paralel sungguhan, tiap proses punya interpreter sendiri)

Pemakaian:
  python project.py                       # data sintetis
  python project.py --folder ./gambar     # gambar asli (jpg/png)
  python project.py --n 120 --size 1024 --repeat 3
Hasil: results.csv, stage_profile.csv, dan grafik results.png (jika ada matplotlib)
"""
import os
# Penting: cegah numpy/BLAS memakai banyak thread DI DALAM tiap proses,
# agar tidak terjadi oversubscription saat process pool berjalan.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image, ImageFilter

# ======================= DATA =======================
def load_image(task):
    kind, src, size = task
    if kind == "synthetic":
        rng = np.random.default_rng(src)       # seed = id gambar -> hasil reproducible
        return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
    return Image.open(src).convert("RGB")

# ======================= TAHAP PIPELINE =======================
def sobel_lama(gray):
    """Versi awal: banyak array sementara -> berat di memori (disimpan untuk perbandingan)."""
    a = gray.astype(np.float32)
    gx = (a[:-2, 2:] + 2*a[1:-1, 2:] + a[2:, 2:]) - (a[:-2, :-2] + 2*a[1:-1, :-2] + a[2:, :-2])
    gy = (a[2:, :-2] + 2*a[2:, 1:-1] + a[2:, 2:]) - (a[:-2, :-2] + 2*a[:-2, 1:-1] + a[:-2, 2:])
    return np.hypot(gx, gy)

def sobel(gray):
    """Versi hemat memori: bentuk separable + operasi in-place.
    Sobel-x = haluskan vertikal [1,2,1] lalu selisih horizontal [1,0,-1]
    Sobel-y = selisih vertikal  [1,0,-1] lalu haluskan horizontal [1,2,1]
    Hasil sama dengan sobel_lama, tetapi hanya ~4 array sementara (bukan ~20)."""
    a = gray.astype(np.float32)
    mid = a[1:-1]
    s = a[:-2] + a[2:]; s += mid; s += mid          # haluskan vertikal
    d = a[2:] - a[:-2]                              # selisih vertikal
    gx = s[:, 2:] - s[:, :-2]
    gy = d[:, :-2] + d[:, 2:]; gy += d[:, 1:-1]; gy += d[:, 1:-1]
    gx *= gx; gy *= gy; gx += gy
    return np.sqrt(gx, out=gx)

D, HEADS, PATCH, LAYERS = 128, 4, 8, 2
_rng = np.random.default_rng(42)              # bobot tetap & sama di semua proses
W = {
    "embed": _rng.standard_normal((PATCH*PATCH, D)).astype(np.float32) * 0.1,
    **{f"{n}{l}": _rng.standard_normal((D, D)).astype(np.float32) * 0.05
       for l in range(LAYERS) for n in ("q", "k", "v", "o")},
    **{f"f1_{l}": _rng.standard_normal((D, 4*D)).astype(np.float32) * 0.05 for l in range(LAYERS)},
    **{f"f2_{l}": _rng.standard_normal((4*D, D)).astype(np.float32) * 0.05 for l in range(LAYERS)},
}

def layernorm(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)

def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def mini_transformer(edge_map):
    """Encoder transformer sederhana: patchify -> embedding -> [attention + MLP] x LAYERS."""
    img = np.asarray(Image.fromarray(edge_map.astype(np.float32)).resize((128, 128)))
    img = img / (img.max() + 1e-6)
    n = 128 // PATCH
    patches = img.reshape(n, PATCH, n, PATCH).transpose(0, 2, 1, 3).reshape(n*n, PATCH*PATCH)
    x = patches @ W["embed"]                                   # (token, D)
    dh = D // HEADS
    for l in range(LAYERS):
        h = layernorm(x)
        q, k, v = (h @ W[f"{c}{l}"] for c in "qkv")
        q, k, v = (t.reshape(-1, HEADS, dh).transpose(1, 0, 2) for t in (q, k, v))
        att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh))  # (head, token, token)
        out = (att @ v).transpose(1, 0, 2).reshape(-1, D)
        x = x + out @ W[f"o{l}"]
        x = x + np.maximum(layernorm(x) @ W[f"f1_{l}"], 0) @ W[f"f2_{l}"]
    return float(x.mean())

STAGES = ["load", "resize", "grayscale", "blur", "sobel", "transformer"]

def process(task, profile=False):
    t = {}
    def tick(name, t0):
        t[name] = time.perf_counter() - t0
    t0 = time.perf_counter(); img = load_image(task);                      tick("load", t0)
    t0 = time.perf_counter(); img = img.resize((512, 512));                tick("resize", t0)
    t0 = time.perf_counter(); gray = img.convert("L");                     tick("grayscale", t0)
    t0 = time.perf_counter(); blur = gray.filter(ImageFilter.GaussianBlur(3)); tick("blur", t0)
    t0 = time.perf_counter(); edges = sobel(np.asarray(blur));             tick("sobel", t0)
    t0 = time.perf_counter(); score = mini_transformer(edges);             tick("transformer", t0)
    return (score, t) if profile else score

# ======================= EKSEKUSI =======================
def run_serial(tasks):
    return [process(t) for t in tasks]

def run_threads(tasks, w):
    with ThreadPoolExecutor(w) as ex:
        return list(ex.map(process, tasks))

def _ping(_):
    time.sleep(0.05)       # menahan worker agar semua worker sempat hidup
    return os.getpid()

def measure_pool(tasks, w, repeat):
    """Pisahkan: (a) startup pool, (b) waktu kerja murni saat pool sudah hidup."""
    cs = max(1, len(tasks) // (w * 4))
    t0 = time.perf_counter()
    pool = Pool(w)                                    # spawn w proses (mahal di Windows)
    pool.map(_ping, range(w * 4), chunksize=1)        # tunggu proses siap
    pool.map(process, tasks[:w * 2], chunksize=1)     # pemanasan (impor, cache)
    startup = time.perf_counter() - t0
    res, work = timeit(lambda: pool.map(process, tasks, chunksize=cs), repeat)
    pool.terminate(); pool.join()
    return res, startup, work

def timeit(fn, repeat):
    best, res = float("inf"), None
    for _ in range(repeat):                      # ambil waktu terbaik -> lebih stabil
        t0 = time.perf_counter(); res = fn(); best = min(best, time.perf_counter() - t0)
    return res, best

# ======================= ANALISA =======================
def profile_stages(tasks):
    """Waktu per tahap pada eksekusi serial -> tahap mana yang paling berat?"""
    tot = dict.fromkeys(STAGES, 0.0)
    for t in tasks[:min(10, len(tasks))]:
        _, ts = process(t, profile=True)
        for k, v in ts.items():
            tot[k] += v
    s = sum(tot.values())
    return {k: (v, v / s * 100) for k, v in tot.items()}

def karp_flatt(speedup, p):
    """Fraksi serial eksperimental. Kecil = bagus; naik saat p naik = overhead."""
    return (1/speedup - 1/p) / (1 - 1/p)

def worker_list():
    n, ws, w = cpu_count(), [], 2
    while w < n:
        ws.append(w); w *= 2
    return ws + ([n] if n >= 2 else [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--size", type=int, default=1024); ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    if a.folder:
        tasks = [("file", os.path.join(a.folder, f), 0) for f in sorted(os.listdir(a.folder))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        tasks = [("synthetic", i, a.size) for i in range(a.n)]
    print(f"Gambar: {len(tasks)} | Core: {cpu_count()} | Repeat: {a.repeat}")

    print("\n[1] Profil per tahap (serial)")
    prof = profile_stages(tasks)
    for k, (sec, pct) in prof.items():
        print(f"  {k:<12}{sec:>8.3f}s  {pct:>5.1f}%  {'#' * int(pct/2)}")
    with open(os.path.join(a.outdir, "stage_profile.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stage", "seconds", "percent"])
        for k, (s, p) in prof.items(): w.writerow([k, f"{s:.4f}", f"{p:.1f}"])

    print("\n[2] Benchmark")
    base, ts = timeit(lambda: run_serial(tasks), a.repeat)
    # baris: mode, worker, startup, kerja, speedup_kerja, speedup_total, efisiensi, karp_flatt, benar
    rows = [("serial", 1, 0.0, ts, 1.0, 1.0, 1.0, "", True)]
    for w in worker_list():
        res, t = timeit(lambda: run_threads(tasks, w), a.repeat)
        sp = ts / t
        rows.append(("thread", w, 0.0, t, sp, sp, sp / w, f"{karp_flatt(sp, w):.3f}", bool(np.allclose(base, res))))
    for w in worker_list():
        res, st, t = measure_pool(tasks, w, a.repeat)
        sp, spt = ts / t, ts / (t + st)
        rows.append(("process", w, st, t, sp, spt, sp / w, f"{karp_flatt(sp, w):.3f}", bool(np.allclose(base, res))))

    print(f"  {'mode':<9}{'worker':>7}{'startup':>9}{'kerja':>8}{'total':>8}{'spd_kerja':>10}{'spd_total':>10}{'efisiensi':>10}{'f_serial':>9}{'benar':>7}")
    for m, w, st, t, sp, spt, ef, kf, ok in rows:
        print(f"  {m:<9}{w:>7}{st:>9.2f}{t:>8.2f}{st+t:>8.2f}{sp:>10.2f}{spt:>10.2f}{ef:>10.2f}{kf:>9}{str(ok):>7}")
    print("  (kerja = pool sudah hidup | total = startup + kerja, yaitu bila pool dibuat untuk satu pekerjaan)")

    # titik impas: berapa gambar minimal agar paralel proses mengalahkan serial?
    print("\n[3] Titik impas proses (jumlah gambar minimum agar total paralel < serial)")
    per_serial = ts / len(tasks)
    for m, w, st, t, sp, spt, ef, kf, ok in rows:
        if m != "process": continue
        per_par = t / len(tasks)
        gain = per_serial - per_par
        be = f"{st / gain:.0f} gambar" if gain > 0 else "tidak pernah impas"
        print(f"  {w:>2} worker: startup {st:.2f}s, hemat {gain*1000:.1f} ms/gambar -> impas di {be}")

    with open(os.path.join(a.outdir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "workers", "startup_s", "work_s", "total_s", "speedup_work", "speedup_total", "efficiency", "karp_flatt", "correct"])
        for m, wk, st, t, sp, spt, ef, kf, ok in rows:
            w.writerow([m, wk, f"{st:.4f}", f"{t:.4f}", f"{st+t:.4f}", f"{sp:.3f}", f"{spt:.3f}", f"{ef:.3f}", kf, ok])

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = list(prof); ax[0].barh(names, [prof[k][1] for k in names]); ax[0].set_xlabel("% waktu (serial)")
        ax[0].set_title("Profil per tahap")
        series = {"thread": ("thread", 4), "process (kerja)": ("process", 4), "process (total, +startup)": ("process", 5)}
        for label, (mode, idx) in series.items():
            pts = [(r[1], r[idx]) for r in rows if r[0] == mode]
            if pts: ax[1].plot(*zip(*pts), marker="o", label=label)
        mx = max([r[1] for r in rows] + [2]); ax[1].plot([1, mx], [1, mx], "--", color="gray", label="ideal")
        ax[1].set_xlabel("jumlah worker"); ax[1].set_ylabel("speedup"); ax[1].legend(fontsize=8); ax[1].set_title("Speedup")
        fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "results.png"), dpi=120)
        print("\nGrafik: results.png")
    except ImportError:
        print("\n(matplotlib tidak ada: pip install matplotlib untuk grafik)")

if __name__ == "__main__":
    main()