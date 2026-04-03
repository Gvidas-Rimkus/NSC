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
        z = z**2 + c
        if abs(z) > 2: 
            n = i
            break
        else: n = max_iter
    return n

if __name__ == '__main__':
    resoultion = 1024
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    cluster = LocalCluster(n_workers=12, threads_per_worker=1)
    client = Client(cluster)
    client.run(lambda: generate_subset(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX)) #warm-up
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        result = mandelbrot_dask(resoultion, X_MIN, X_MAX, Y_MIN, Y_MAX)
        times.append(time.perf_counter() - t0)
    print(f"Dask local (n_chunks=32): {statistics.median(times):.5f} ± {statistics.stdev(times):.5f}")
    client.close(); cluster.close()

    # plt.imshow(result, cmap="cool", vmin=0, vmax=100)
    # plt.show()