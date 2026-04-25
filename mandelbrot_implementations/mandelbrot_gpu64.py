import pyopencl as cl
import numpy as np

KERNEL_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot(
    __global int *result,
    const double x_min,
    const double x_max,

    const double y_min,
    const double y_max,

    const int N,
    const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return; // original comment says 'guard against over-launch' but i dont understand what this means?

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * N + col] = count;
}

"""

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
prog = cl.Program(ctx, KERNEL_SRC).build()

N, MAX_ITER = 2048, 100

X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

prog.mandelbrot(
    queue, (N, N), None,      # global size (N, N); let OpenCL pick local
    image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(N), np.int32(MAX_ITER),
)

cl.enqueue_copy(queue, image, image_dev)
queue.finish()

import time, math, matplotlib.pyplot as plt

# --- Warm up (first launch triggers a kernel compile) ---
prog.mandelbrot(queue, (64, 64), None, image_dev,
                np.float64(X_MIN), np.float64(X_MAX),
                np.float64(Y_MIN), np.float64(Y_MAX),
                np.int32(64), np.int32(MAX_ITER))
queue.finish()

# --- Timed runs ---
N_RUNS = 20
times = []
for i in range(N_RUNS):
    t0 = time.perf_counter()
    prog.mandelbrot(queue, (N, N), None, image_dev,
                    np.float64(X_MIN), np.float64(X_MAX),
                    np.float64(Y_MIN), np.float64(Y_MAX),
                    np.int32(N), np.int32(MAX_ITER))
    queue.finish()
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"Run {i}: Time {elapsed:.5f}")

mean = np.mean(times)
std  = np.std(times)
ci   = 1.96 * std / math.sqrt(N_RUNS)
print("-------------------------------")
print(f"GPU {N}x{N}  95% CI: {mean:.5f} ± {ci:.5f}")

cl.enqueue_copy(queue, image, image_dev)
queue.finish()
plt.imshow(image, cmap='hot', origin='lower'); plt.axis('off')
plt.savefig("pictures/mandelbrot_gpu64.png", dpi=150, bbox_inches='tight')