import argparse
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import ImageFilter

from project import STAGES, load_image, mini_transformer, sobel


def upto(arg):
    task, k = arg
    img = load_image(task)
    if k == 1:
        return float(np.asarray(img)[0, 0, 0])
    img = img.resize((512, 512))
    if k == 2:
        return float(np.asarray(img)[0, 0, 0])
    gray = img.convert("L")
    if k == 3:
        return float(np.asarray(gray)[0, 0])
    blur = gray.filter(ImageFilter.GaussianBlur(3))
    if k == 4:
        return float(np.asarray(blur)[0, 0])
    edges = sobel(np.asarray(blur))
    if k == 5:
        return float(edges.mean())
    return mini_transformer(edges)


def best_time(fn, repeat=2):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--workers", default="4,8")
    a = ap.parse_args()
    workers = [int(x) for x in a.workers.split(",") if int(x) <= max(cpu_count(), 2)]
    tasks = [("synthetic", i, a.size) for i in range(a.n)]

    pools = {}
    for w in workers:
        pools[w] = Pool(w)
        pools[w].map(upto, [(t, 1) for t in tasks[: w * 2]], chunksize=1)

    print(f"Gambar: {a.n} | ukuran: {a.size} | Core: {cpu_count()}\n")
    head = f"{'sampai tahap':<22}{'serial(s)':>10}"
    for w in workers:
        head += f"{'spd@' + str(w):>9}"
    print(head)

    for k in range(1, 7):
        work = [(t, k) for t in tasks]
        ts = best_time(lambda: [upto(x) for x in work])
        line = f"{k}. {STAGES[k - 1]:<19}{ts:>10.2f}"
        for w in workers:
            cs = max(1, len(work) // (w * 4))
            tp = best_time(lambda: pools[w].map(upto, work, chunksize=cs))
            line += f"{ts / tp:>9.2f}"
        print(line)

    for p in pools.values():
        p.terminate()
        p.join()
    print("\nBaca: cari baris di mana speedup turun tajam dibanding baris sebelumnya.")