import numpy as np

def generate_set(resolution:int):
    x_region = np.linspace(start=-2, stop=1, num=resolution)
    y_region = np.linspace(start=-1.5, stop=1.5, num=resolution)
    grid = np.zeros((resolution, resolution))
    for i in range(0, resolution):
        for j in range(0, resolution):
            n = evaluate_point(x=x_region[i], y=y_region[j], max_iter=100)
            grid[i][j] = n
    return grid

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