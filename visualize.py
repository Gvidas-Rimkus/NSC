from mandelbrot_naive import generate_set
import matplotlib.pyplot as plt

grid = generate_set(resolution=1024, max_iter=200)
plt.imshow(grid, cmap="Accent", vmin=0, vmax=100)
plt.show()