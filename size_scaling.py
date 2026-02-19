from measure_performance import measure_performance
import matplotlib.pyplot as plt
import numpy as np

sizes = [256, 512, 1024, 2048, 4096, 8192]
mean_times = []
for size in sizes:
    time_list = measure_performance(resolution=size, runs=1)
    mean_time = np.mean(time_list)
    mean_times.append(mean_time)

plt.figure()
plt.plot(sizes, mean_times, marker='o')
plt.title('Size vs Mean Time')
plt.tight_layout()
plt.show()