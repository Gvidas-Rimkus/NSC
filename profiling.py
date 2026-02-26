import cProfile , pstats
from mandelbrot_implementations import mandelbrot_naive, mandelbrot_vectorized

cProfile.run("mandelbrot_naive.generate_set(resolution=1024)", "profiles/naive_profile.prof")
cProfile.run("mandelbrot_vectorized.generate_set(resolution=1024)", "profiles/vectorized_profile.prof")

for name in ("profiles/naive_profile.prof", "profiles/vectorized_profile.prof"):
    stats = pstats.Stats(name)
    stats.sort_stats("cumulative")
    stats.print_stats(10) 