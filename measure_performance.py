import numpy as np
import time
import math
import argparse

def get_generate_fn(implementation: str):
    if implementation == "naive": from mandelbrot_implementations.naive import generate_set
    elif implementation == "vectorized": from mandelbrot_implementations.vectorized import generate_set
    elif implementation == "numba": from mandelbrot_implementations.numba import generate_set
    elif implementation == "numba32": from mandelbrot_implementations.numba32 import generate_set
    elif implementation == "numba_parallel": from mandelbrot_implementations.numba_parallel import generate_set
    elif implementation == "numba32_parallel": from mandelbrot_implementations.numba32_parallel import generate_set
    return generate_set

def measure_performance(resolution: int):
    elapsed_times = np.zeros((100), dtype=np.float16)
    for i in range(0,100):
        start = time.perf_counter()
        generate_set(resolution=resolution)
        end = time.perf_counter()
        elapsed = end - start
        elapsed_times[i] = elapsed
        print(f"Run {i}: Time {elapsed:.3f}")
    mean = np.mean(elapsed_times)
    std = np.std(elapsed_times)
    ci = 1.96 * std / math.sqrt(100)
    print("-------------------------------")
    print(f"95% CI: {mean:.5f} ± {ci:.5f}")
    return elapsed_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=1024, choices=[1024, 2048, 4096, 8192])
    parser.add_argument("--implementation", type=str, default="numba", choices=["naive", 
                                                                                "vectorized", 
                                                                                "numba", 
                                                                                "numba32",
                                                                                "numba_parallel",
                                                                                "numba32_parallel"])
    args = parser.parse_args()

    generate_set = get_generate_fn(args.implementation)

    _ = generate_set(resolution=64) #warm-up
    measure_performance(resolution=args.resolution)