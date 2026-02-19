import numpy as np
import time

N = 10_000
A = np.random.rand(N, N)

print("NUMPY ARRAY")
start = time.perf_counter()
for i in range(N): s = np.sum(A[i, :])
end = time.perf_counter()
elapsed = end - start
print(f"ROWS: {elapsed}")

start = time.perf_counter()
for i in range(N): s = np.sum(A[:, i])
end = time.perf_counter()
elapsed = end - start
print(f"COLUMNS: {elapsed}")

print("FORTRAN ARRAY")
A_f = np.asfortranarray(A)
start = time.perf_counter()
for i in range(N): s = np.sum(A_f[i, :])
end = time.perf_counter()
elapsed = end - start
print(f"ROWS: {elapsed}")

start = time.perf_counter()
for i in range(N): s = np.sum(A_f[:, i])
end = time.perf_counter()
elapsed = end - start
print(f"COLUMNS: {elapsed}")

# NUMPY ARRAY
# ROWS: 0.06472887500422075
# COLUMNS: 0.2513156670029275
# FORTRAN ARRAY
# ROWS: 0.412986166018527
# COLUMNS: 0.039516541990451515