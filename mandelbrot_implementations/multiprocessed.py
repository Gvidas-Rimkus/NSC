from multiprocessing import Pool
from numba import njit
import numpy as np
import psutil
import os
import time

@njit
def generate_set(resolution=1024, x_min=-2, x_max=1, y_min=-1.5, y_max=1.5):
    return generate_subset(row_start=0, row_end=resolution, resolution=resolution, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

@njit
def generate_subset(row_start, row_end, resolution, x_min, x_max, y_min, y_max):
    x_region = np.linspace(start=x_min, stop=x_max, num=resolution)
    y_region = np.linspace(start=y_min, stop=y_max, num=resolution)
    grid = np.empty((row_end - row_start, resolution), dtype=np.int32)
    for i in range(row_end - row_start):
        for j in range(resolution):
            n = evaluate_point(x=x_region[i], y=y_region[j], max_iter=100)
            grid[i][j] = n
    return grid

@njit
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
    generate_set(resolution=1024, x_min=-2, x_max=1, y_min=-1.5, y_max=1.5)