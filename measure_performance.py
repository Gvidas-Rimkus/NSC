# from mandelbrot_naive import generate_set
from mandelbrot_vectorized import generate_set
import numpy as np
import time
import math

def measure_performance(resolution: int, max_iter: int, runs: int):
    elapsed_times = np.zeros((100), dtype=np.float16)
    for i in range(0,runs):
        start = time.perf_counter()
        generate_set(resolution=resolution, max_iter=max_iter)
        end = time.perf_counter()
        elapsed = end - start
        elapsed_times[i] = elapsed
        print(f"Run {i}: Time {elapsed:.3f}")
    mean = np.mean(elapsed_times)
    std = np.std(elapsed_times)
    ci = 1.96 * std / math.sqrt(runs)
    print("-------------------------------")
    print(f"95% CI: {mean:.5f} ± {ci:.5f}")
    return elapsed_times

times = measure_performance(resolution=1024, max_iter=100, runs=100)