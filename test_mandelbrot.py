import numpy as np
import pytest

import mandelbrot_implementations.naive as naive
import mandelbrot_implementations.multiprocessed as mp_impl
import mandelbrot_implementations.dask_local as dask_impl


# ---------------------------------------------------------------------------
# Test 1 — analytically provable single-point values (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, expected", [
    (0.0, 0.0, 100),  # c=0: z stays 0 forever, never escapes
    (3.0, 0.0, 0),    # c=3: z1=3, |z1|>2, escapes on iteration 0
    (0.0, 3.0, 0),    # c=3j: z1=3j, |z1|=3>2, escapes on iteration 0
])
def test_evaluate_point_analytically_known(x, y, expected):
    assert naive.evaluate_point(x=x, y=y, max_iter=100) == expected


# ---------------------------------------------------------------------------
# Test 2 — cross-validation: naive oracle vs multiprocessing on a 32x32 grid
# ---------------------------------------------------------------------------

def test_generate_set_cross_validates_naive_vs_multiprocessed():
    resolution = 32
    naive_result = naive.generate_set(resolution=resolution)
    mp_result = mp_impl.generate_set(
        resolution=resolution, n_workers=1, n_chunks=4
    )
    assert np.array_equal(naive_result, mp_result)


# ---------------------------------------------------------------------------
# Test 3 — dask compute function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, expected", [
    (0.0, 0.0, 100),
    (3.0, 0.0, 0),
    (0.0, 3.0, 0),
])
def test_dask_evaluate_point_matches_known_values(x, y, expected):
    assert dask_impl.evaluate_point(x=x, y=y, max_iter=100) == expected
