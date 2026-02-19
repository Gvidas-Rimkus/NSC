from mandelbrot_implementations.mandelbrot_vectorized import generate_set
import matplotlib.pyplot as plt

grid = generate_set(resolution=8192)
plt.imshow(grid, cmap="cool", vmin=0, vmax=100)
plt.show()