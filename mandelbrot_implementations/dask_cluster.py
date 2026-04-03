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
        n_workers = 4
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
        print(f"Resolution: {resolution}")
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

# Resolution: 1024
# Serial Numba (T1): 0.06347 s
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    0.10501 |    1.654 |    0.604 |    5.618
#          2 |    0.10892 |    1.716 |    0.583 |    5.865
#          4 |    0.11847 |    1.867 |    0.536 |    6.467
#          8 |    0.13283 |    2.093 |    0.478 |    7.371
#         16 |    0.16704 |    2.632 |    0.380 |    9.527
#         32 |    0.24419 |    3.848 |    0.260 |   14.390
#         64 |    0.36836 |    5.804 |    0.172 |   22.216
#        128 |    0.65484 |   10.318 |    0.097 |   40.270
#        256 |    1.17356 |   18.491 |    0.054 |   72.962
# n_chunks_optimal = 1, t_min = 0.10501 s, LIF_min = 5.61783

# Resolution: 2048
# Serial Numba (T1): 0.25280 s
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    0.31849 |    1.260 |    0.794 |    4.039
#          2 |    0.31684 |    1.253 |    0.798 |    4.013
#          4 |    0.32605 |    1.290 |    0.775 |    4.159
#          8 |    0.34328 |    1.358 |    0.736 |    4.432
#         16 |    0.37278 |    1.475 |    0.678 |    4.898
#         32 |    0.43639 |    1.726 |    0.579 |    5.905
#         64 |    0.58615 |    2.319 |    0.431 |    8.275
#        128 |    0.84258 |    3.333 |    0.300 |   12.332
#        256 |    1.40244 |    5.548 |    0.180 |   21.191
# n_chunks_optimal = 2, t_min = 0.31684 s, LIF_min = 4.01332

# Resolution: 4096
# Serial Numba (T1): 1.01897 s
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    1.15121 |    1.130 |    0.885 |    3.519
#          2 |    1.18214 |    1.160 |    0.862 |    3.641
#          4 |    1.19096 |    1.169 |    0.856 |    3.675
#          8 |    1.19536 |    1.173 |    0.852 |    3.692
#         16 |    1.20690 |    1.184 |    0.844 |    3.738
#         32 |    1.26639 |    1.243 |    0.805 |    3.971
#         64 |    1.42855 |    1.402 |    0.713 |    4.608
#        128 |    1.76832 |    1.735 |    0.576 |    5.942
#        256 |    2.26827 |    2.226 |    0.449 |    7.904
# n_chunks_optimal = 1, t_min = 1.15121 s, LIF_min = 3.51913

# Resolution: 8192
# Serial Numba (T1): 4.05701 s
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    4.51469 |    1.113 |    0.899 |    3.451
#          2 |    4.56160 |    1.124 |    0.889 |    3.497
#          4 |    4.52963 |    1.116 |    0.896 |    3.466
#          8 |    4.53825 |    1.119 |    0.894 |    3.474
#         16 |    4.58195 |    1.129 |    0.885 |    3.518
#         32 |    4.62034 |    1.139 |    0.878 |    3.555
#         64 |    4.74772 |    1.170 |    0.855 |    3.681
#        128 |    4.92723 |    1.214 |    0.823 |    3.858
#        256 |    5.49496 |    1.354 |    0.738 |    4.418
# n_chunks_optimal = 1, t_min = 4.51469 s, LIF_min = 3.45124