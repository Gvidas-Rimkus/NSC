from numba import njit, complex64, prange
import numpy as np

@njit(parallel=True, fastmath=True)
def generate_set(resolution:int = 1024):
    x_region = np.linspace(-2, 1, resolution).astype(np.float32)
    y_region = np.linspace(-1.5, 1.5, resolution).astype(np.float32)
    grid = np.zeros((resolution, resolution), dtype=np.float32)
    for i in prange(0, resolution):
        for j in range(0, resolution):
            c = complex64(x_region[i] + 1j*y_region[j])
            z = complex64(0j)
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