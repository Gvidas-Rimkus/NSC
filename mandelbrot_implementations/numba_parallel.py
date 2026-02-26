from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def generate_set(resolution:int = 1024):
    x_region = np.linspace(start=-2, stop=1, num=resolution)
    y_region = np.linspace(start=-1.5, stop=1.5, num=resolution)
    grid = np.zeros((resolution, resolution))
    for i in prange(0, resolution):
        for j in range(0, resolution):
            c = x_region[i] + 1j*y_region[j]
            z = 0j
            for k in range(0, 100):
                z = z**2 + c
                if z.real*z.real + z.imag*z.imag > 4.0: 
                    n = k
                    break
                else: n = 100
            grid[i][j] = n
    return grid

if __name__ == "__main__":
    generate_set(resolution=1024)