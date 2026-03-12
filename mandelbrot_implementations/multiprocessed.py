from multiprocessing import Pool
from numba import njit
import numpy as np
import statistics
import time
import math
import os

def generate_set(resolution=1024, x_min=-2, x_max=1, y_min=-1.5, y_max=1.5, n_workers=6, n_chunks=None):
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, resolution // n_chunks)
    chunks, row = [], 0
    while row < resolution:
        row_end = min(row + chunk_size, resolution)
        chunks.append((row, row_end, resolution, x_min, x_max, y_min, y_max))
        row = row_end
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max)]
    with Pool(processes=n_workers) as pool:
        pool.map(_worker, tiny)#warm up
        parts = pool.map(_worker, chunks)
    return np.vstack(parts)

def _worker(args):
    return generate_subset(*args)

@njit(fastmath=True, cache=True)
def generate_subset(row_start, row_end, resolution, x_min, x_max, y_min, y_max):
    x_region = np.linspace(start=x_min, stop=x_max, num=resolution)
    y_region = np.linspace(start=y_min, stop=y_max, num=resolution)
    grid = np.empty((row_end - row_start, resolution), dtype=np.int32)
    for i in range(row_end - row_start):
        for j in range(resolution):
            n = evaluate_point(x=x_region[row_start + i], y=y_region[j], max_iter=100)
            grid[i][j] = n
    return grid

@njit(fastmath=True, cache=True)
def evaluate_point(x:float, y:float, max_iter:int):
    c = x + 1j*y
    z = 0
    for i in range(0, max_iter):
        z = z**2 + c
        if abs(z) > 2: 
            n = i
            break
        else: n = max_iter
    return n

if __name__ == "__main__":
    n_workers = 12
    resolution = 1024
    x_min=-2 
    x_max=1
    y_min=-1.5 
    y_max=1.5
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        generate_subset(row_start=0, row_end=resolution, resolution=resolution, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        times.append(time.perf_counter() - t0)
    t_serial = statistics.median(times)

    #Here we do not use the generate_set function, due to the requirement of not including the creation of workers (and the like) in the timing.
    # for n_workers in range(1, os.cpu_count()+1):
    #     chunk_size = max(1, resolution // n_workers)
    #     chunks, row = [], 0
    #     while row < resolution:
    #         row_end = min(row + chunk_size, resolution)
    #         chunks.append((row, row_end, resolution, x_min, x_max, y_min, y_max))
    #         row = row_end
    #     tiny = [(0, 8, 8, x_min, x_max, y_min, y_max)]            
    #     with Pool(processes=n_workers) as pool:
    #         pool.map(_worker, tiny) #warm up
    #         times = []
    #         for _ in range(10):
    #             t0 = time.perf_counter()
    #             np.vstack(pool.map(_worker, chunks))
    #             times.append(time.perf_counter() - t0)
    #         t_par = statistics.median(times)
    #         std = np.std(times)
    #         ci = 1.96 * std / math.sqrt(10)
    #         speedup = t_serial / t_par
    #         print(f"{n_workers:2d} workers: {t_par:.5f} ± {ci:.5f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")
    chunk_counts = [n_workers * multiple for multiple in [1, 2, 4, 8, 16, 32, 64, 128]]
    for chunk_count in chunk_counts:
        chunk_size = max(1, resolution // chunk_count)
        chunks, row = [], 0
        while row < resolution:
            row_end = min(row + chunk_size, resolution)
            chunks.append((row, row_end, resolution, x_min, x_max, y_min, y_max))
            row = row_end
        tiny = [(0, 8, 8, x_min, x_max, y_min, y_max)]            
        with Pool(processes=n_workers) as pool:
            pool.map(_worker, tiny) #warm up
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                np.vstack(pool.map(_worker, chunks))
                times.append(time.perf_counter() - t0)
            t_par = statistics.median(times)
            std = np.std(times)
            ci = 1.96 * std / math.sqrt(10)
            speedup = t_serial / t_par
            print(f"{chunk_count} chunks, {n_workers:2d} workers: {t_par:.5f} ± {ci:.5f}s, speedup={speedup:.2f}x, eff={speedup/n_workers*100:.0f}%")

#  resolution = 1024
#  1 workers: 0.11720 ± 0.00077s, speedup=0.92x, eff=92%
#  2 workers: 0.07473 ± 0.00047s, speedup=1.45x, eff=72%
#  3 workers: 0.06548 ± 0.00034s, speedup=1.65x, eff=55%
#  4 workers: 0.06187 ± 0.00219s, speedup=1.75x, eff=44%
#  5 workers: 0.04769 ± 0.00036s, speedup=2.27x, eff=45%
#  6 workers: 0.04344 ± 0.00060s, speedup=2.49x, eff=42%
#  7 workers: 0.03871 ± 0.00126s, speedup=2.80x, eff=40%
#  8 workers: 0.03438 ± 0.00106s, speedup=3.15x, eff=39%
#  9 workers: 0.03274 ± 0.00113s, speedup=3.31x, eff=37%
# 10 workers: 0.03036 ± 0.00160s, speedup=3.57x, eff=36%
# 11 workers: 0.02884 ± 0.00160s, speedup=3.76x, eff=34%
# 12 workers: 0.02739 ± 0.00136s, speedup=3.95x, eff=33% Fastest

#  resolution = 2048
#  1 workers: 0.46752 ± 0.00903s, speedup=0.91x, eff=91%
#  2 workers: 0.30242 ± 0.00745s, speedup=1.41x, eff=71%
#  3 workers: 0.26086 ± 0.00451s, speedup=1.64x, eff=55%
#  4 workers: 0.24743 ± 0.00346s, speedup=1.73x, eff=43%
#  5 workers: 0.18663 ± 0.00173s, speedup=2.29x, eff=46%
#  6 workers: 0.17285 ± 0.00194s, speedup=2.47x, eff=41%
#  7 workers: 0.15448 ± 0.00156s, speedup=2.77x, eff=40%
#  8 workers: 0.13181 ± 0.00152s, speedup=3.24x, eff=41%
#  9 workers: 0.12810 ± 0.00205s, speedup=3.34x, eff=37%
# 10 workers: 0.11751 ± 0.00488s, speedup=3.64x, eff=36%
# 11 workers: 0.10688 ± 0.00417s, speedup=4.00x, eff=36% Fastest
# 12 workers: 0.10937 ± 0.00411s, speedup=3.91x, eff=33%

# MP2 M1: 
# res 1024: 48 chunks, 12 workers: 0.02479 ± 0.00096s, speedup=4.37x, eff=36%
# res 2048: 48 chunks, 12 workers: 0.09287 ± 0.00303s, speedup=4.57x, eff=38%

# MP2 M2:
# res 1024:
# 12 chunks, 12 workers: 0.02842 ± 0.00095s, speedup=3.81x, eff=32%
# 24 chunks, 12 workers: 0.02451 ± 0.00075s, speedup=4.41x, eff=37%
# 48 chunks, 12 workers: 0.02605 ± 0.00128s, speedup=4.15x, eff=35%
# 96 chunks, 12 workers: 0.02337 ± 0.00070s, speedup=4.63x, eff=39%
# 192 chunks, 12 workers: 0.02296 ± 0.00080s, speedup=4.71x, eff=39%
# 384 chunks, 12 workers: 0.02426 ± 0.00056s, speedup=4.46x, eff=37%
# 768 chunks, 12 workers: 0.02408 ± 0.00048s, speedup=4.49x, eff=37%
# 1536 chunks, 12 workers: 0.02452 ± 0.00069s, speedup=4.41x, eff=37%
# res 2048:
# 12 chunks, 12 workers: 0.10678 ± 0.00401s, speedup=3.97x, eff=33%
# 24 chunks, 12 workers: 0.09366 ± 0.00250s, speedup=4.53x, eff=38%
# 48 chunks, 12 workers: 0.09325 ± 0.00295s, speedup=4.55x, eff=38%
# 96 chunks, 12 workers: 0.09030 ± 0.00159s, speedup=4.70x, eff=39%
# 192 chunks, 12 workers: 0.08915 ± 0.00139s, speedup=4.76x, eff=40%
# 384 chunks, 12 workers: 0.08784 ± 0.00168s, speedup=4.83x, eff=40%
# 768 chunks, 12 workers: 0.08944 ± 0.00269s, speedup=4.74x, eff=40%
# 1536 chunks, 12 workers: 0.08903 ± 0.00211s, speedup=4.76x, eff=40%
