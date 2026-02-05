from mandelbrot import generate_set
import matplotlib.pyplot as plt

grid = generate_set(resolution=2048, max_iter=200)
plt.imshow(grid, cmap="hot", vmin=0, vmax=100)
plt.show()