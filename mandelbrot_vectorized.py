import numpy as np

def generate_set(resolution, max_iter):
    x_region = np.linspace(start=-2, stop=1, num=resolution)
    y_region = np.linspace(start=-1.5, stop=1.5, num=resolution)
    X, Y = np.meshgrid(x_region, y_region)
    C = X + 1j*Y
    Z = np.zeros(shape=C.shape, dtype=complex)
    M = np.zeros(shape=C.shape, dtype=int)
    for _ in range(0, max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M