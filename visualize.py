from mandelbrot_implementations.numba import generate_set
import matplotlib.pyplot as plt

grid = generate_set(resolution=1024)
plt.imshow(grid, cmap="cool", vmin=0, vmax=100)
plt.show()