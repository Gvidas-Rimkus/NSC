import matplotlib.pyplot as plt
import numpy as np

labels = [
    "Naive\nPython",
    "NumPy",
    "Numba\nf64",
    "Numba\nf32",
    "Multi-\nprocessing",
    "Dask\nLocal",
    "Dask\nCluster",
    "GPU\nf32",
    "GPU\nf64",
]

times_1024 = [
    3.18359,   # Naive
    0.32275,   # NumPy (vectorized)
    0.00931,   # Numba f64 parallel+fastmath
    0.00957,   # Numba f32 parallel+fastmath
    0.02296,   # Multiprocessing optimal chunks
    0.03588,   # Dask local optimal chunks
    0.10501,   # Dask cluster optimal chunks
    0.00020,   # GPU f32
    0.00120,   # GPU f64
]

times_2048 = [
    19.70312,  # Naive
    1.59082,   # NumPy (vectorized)
    0.04132,   # Numba f64 parallel+fastmath
    0.03842,   # Numba f32 parallel+fastmath
    0.08784,   # Multiprocessing optimal chunks
    0.06430,   # Dask local optimal chunks
    0.31684,   # Dask cluster optimal chunks
    0.00031,   # GPU f32
    0.00367,   # GPU f64
]

colors = [
    "#4e79a7",  # Naive
    "#f28e2b",  # NumPy
    "#59a14f",  # Numba f64
    "#76b7b2",  # Numba f32
    "#e15759",  # Multiprocessing
    "#edc948",  # Dask local
    "#b07aa1",  # Dask cluster
    "#ff9da7",  # GPU f32
    "#9c755f",  # GPU f64
]

x = np.arange(len(labels))
bar_width = 0.6

for res, times, fname in [
    (1024, times_1024, "performance_1024.png"),
    (2048, times_2048, "performance_2048.png"),
]:
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(x, times, width=bar_width, color=colors, edgecolor="white", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Wall time (s, log scale)", fontsize=12)
    ax.set_title(f"Mandelbrot Performance — Desktop, {res}×{res}, max_iter=100", fontsize=13)
    ax.yaxis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            t * 1.4,
            f"{t:.5f}s",
            ha="center", va="bottom", fontsize=8, rotation=45,
        )

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()
