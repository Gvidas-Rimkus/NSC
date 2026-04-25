import numpy as np


def generate_set(resolution: int) -> np.ndarray:
    """Generate a Mandelbrot set iteration-count grid.

    Parameters
    ----------
    resolution : int
        Number of pixels along each axis of the square output grid.

    Returns
    -------
    np.ndarray
        2-D array of shape (resolution, resolution) where each value is the
        iteration count returned by evaluate_point for that grid coordinate.
    """
    x_region = np.linspace(start=-2, stop=1, num=resolution)
    y_region = np.linspace(start=-1.5, stop=1.5, num=resolution)
    grid = np.zeros((resolution, resolution))
    for i in range(0, resolution):
        for j in range(0, resolution):
            n = evaluate_point(x=x_region[i], y=y_region[j], max_iter=100)
            grid[i][j] = n
    return grid


def evaluate_point(x: float, y: float, max_iter: int) -> int:
    """Count iterations before the Mandelbrot sequence escapes for point c = x + yj.

    Parameters
    ----------
    x : float
        Real part of the complex coordinate c.
    y : float
        Imaginary part of the complex coordinate c.
    max_iter : int
        Maximum number of iterations before declaring the point in-set.

    Returns
    -------
    int
        Iteration index at which |z| > 2, or max_iter if the point did not escape.
    """
    c = x + 1j*y
    z = 0
    for i in range(0, max_iter):
        z = z**2 + c
        if abs(z) > 2:
            n = i
            break
        else:
            n = max_iter
    return n

if __name__ == "__main__":
    generate_set(resolution=1024)