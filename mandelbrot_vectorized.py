import numpy as np

resolution = 1024

x_region = np.linspace(start=-2, stop=1, num=resolution)
y_region = np.linspace(start=-1.5, stop=1.5, num=resolution)

X, Y = np.meshgrid(x_region, y_region)
C = X + 1j*Y
