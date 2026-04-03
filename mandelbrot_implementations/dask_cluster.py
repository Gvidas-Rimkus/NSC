from dask import delayed
from dask.distributed import Client, LocalCluster
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import dask
import time
import statistics

def mandelbrot_dask(resolution, x_min, x_max, y_min, y_max, n_chunks=32):
    chunk_size = max(1, resolution // n_chunks)
    tasks, row_start = [], 0
    while row_start < resolution:
        row_end = min(row_start + chunk_size, resolution)
        tasks.append(delayed(generate_subset)(row_start, row_end, resolution, x_min, x_max, y_min, y_max))
        row_start = row_end
    parts = dask.compute(*tasks)
    return np.vstack(parts)

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
        z = z*z + c
        if z.real*z.real + z.imag*z.imag > 4.0:
            n = i
            break
        else: n = max_iter
    return n

if __name__ == '__main__':
    for resolution in [1024, 2048, 4096, 8192]:
        n_workers = 12
        n_runs = 3

        X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
        client = Client("tcp://10.92.0.46:8786")
        client.run(lambda: generate_subset(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX)) #warm-up

        t_serial_runs = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            generate_subset(0, resolution, resolution, X_MIN, X_MAX, Y_MIN, Y_MAX)
            t_serial_runs.append(time.perf_counter() - t0)
        t1 = statistics.median(t_serial_runs)
        print(f"Serial Numba (T1): {t1:.5f} s")

        chunk_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        print(f"\n{'n_chunks':>10} | {'time (s)':>10} | {'vs 1x':>8} | {'speedup':>8} | {'LIF':>8}")
        print("-" * 56)

        results = []
        for n_chunks in chunk_counts:
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                result = mandelbrot_dask(resolution, X_MIN, X_MAX, Y_MIN, Y_MAX, n_chunks)
                times.append(time.perf_counter() - t0)

            t_med = statistics.median(times)
            speedup = t1 / t_med
            p = n_workers
            lif = p * (t_med / t1) - 1

            results.append((n_chunks, t_med, speedup, lif))
            print(f"{n_chunks:>10} | {t_med:>10.5f} | {t_med/t1:>8.3f} | {speedup:>8.3f} | {lif:>8.3f}")

        best = min(results, key=lambda r: r[1])
        n_chunks_optimal, t_min, speedup_best, lif_min = best
        print(f"\nn_chunks_optimal = {n_chunks_optimal}, t_min = {t_min:.5f} s, LIF_min = {lif_min:.5f}")

        ns = [r[0] for r in results]
        ts = [r[1] for r in results]
        plt.figure()
        plt.plot(ns, ts, marker='o')
        # plt.xscale('log')
        plt.xlabel('n_chunks')
        plt.ylabel('Wall time (s)')
        plt.title(f'Dask Local Chunk Sweep (N={resolution}, workers={n_workers})')
        plt.tight_layout()
        plt.savefig(f'dask_chunk_sweep_cluster_{resolution}.png')
        # plt.show()

        client.close()