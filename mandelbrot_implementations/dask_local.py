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
    resolution = 8192
    n_workers = 12
    n_runs = 3

    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
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
    plt.savefig('dask_chunk_sweep_8192.png')
    plt.show()

    client.close()
    cluster.close()
    # plt.imshow(result, cmap="cool", vmin=0, vmax=100)
    # plt.show()

# 1024
# Serial Numba (T1): 0.04065 s
# 
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    0.05885 |    1.448 |    0.691 |   16.371
#          2 |    0.04989 |    1.227 |    0.815 |   13.727
#          4 |    0.04326 |    1.064 |    0.940 |   11.768
#          8 |    0.03721 |    0.915 |    1.093 |    9.982
#         16 |    0.03588 |    0.883 |    1.133 |    9.591
#         32 |    0.04395 |    1.081 |    0.925 |   11.973
#         64 |    0.06407 |    1.576 |    0.634 |   17.913
#        128 |    0.09941 |    2.445 |    0.409 |   28.344
#        256 |    0.20117 |    4.948 |    0.202 |   58.380
# 
# n_chunks_optimal = 16, t_min = 0.03588 s, LIF_min = 9.59070
# 
# 2048
# Serial Numba (T1): 0.16203 s
# 
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    0.18874 |    1.165 |    0.858 |   12.978
#          2 |    0.16980 |    1.048 |    0.954 |   11.575
#          4 |    0.13060 |    0.806 |    1.241 |    8.672
#          8 |    0.09633 |    0.595 |    1.682 |    6.134
#         16 |    0.07948 |    0.491 |    2.039 |    4.886
#         32 |    0.06430 |    0.397 |    2.520 |    3.762
#         64 |    0.07548 |    0.466 |    2.147 |    4.590
#        128 |    0.12331 |    0.761 |    1.314 |    8.132
#        256 |    0.21970 |    1.356 |    0.738 |   15.271
# 
# n_chunks_optimal = 32, t_min = 0.06430 s, LIF_min = 3.76216
# 
# 4096
# Serial Numba (T1): 0.64611 s
# 
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    0.68577 |    1.061 |    0.942 |   11.737
#          2 |    0.56833 |    0.880 |    1.137 |    9.555
#          4 |    0.45447 |    0.703 |    1.422 |    7.441
#          8 |    0.28370 |    0.439 |    2.277 |    4.269
#         16 |    0.24498 |    0.379 |    2.637 |    3.550
#         32 |    0.16452 |    0.255 |    3.927 |    2.056
#         64 |    0.15752 |    0.244 |    4.102 |    1.926
#        128 |    0.18121 |    0.280 |    3.565 |    2.366
#        256 |    0.25775 |    0.399 |    2.507 |    3.787
# 
# n_chunks_optimal = 64, t_min = 0.15752 s, LIF_min = 1.92554
#
# 8192
# Serial Numba (T1): 2.57586 s
# 
#   n_chunks |   time (s) |    vs 1x |  speedup |      LIF
# --------------------------------------------------------
#          1 |    2.70085 |    1.049 |    0.954 |   11.582
#          2 |    2.22606 |    0.864 |    1.157 |    9.370
#          4 |    1.76307 |    0.684 |    1.461 |    7.214
#          8 |    1.10507 |    0.429 |    2.331 |    4.148
#         16 |    0.96803 |    0.376 |    2.661 |    3.510
#         32 |    0.61769 |    0.240 |    4.170 |    1.878
#         64 |    0.49249 |    0.191 |    5.230 |    1.294
#        128 |    0.48978 |    0.190 |    5.259 |    1.282
#        256 |    0.52728 |    0.205 |    4.885 |    1.456
# 
# n_chunks_optimal = 128, t_min = 0.48978 s, LIF_min = 1.28173
