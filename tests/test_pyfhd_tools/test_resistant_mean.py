from PyFHD.pyfhd_tools.pyfhd_utils import resistant_mean
import numpy as np
from numpy import testing as npt
import pytest

def test_res_mean_int():
    input = np.concatenate([np.arange(20), np.array([100,200,300,400])])
    assert resistant_mean(input, 2) == 9.5

def test_res_mean_float():
    input = np.concatenate([np.arange(0, 20, 0.75), np.array([25.0, -10.75, -30.0, 50])])
    assert resistant_mean(input, 2) == 9.75

def test_res_mean_complex_int():
    input = np.linspace(0 + 2j, 20 + 42j, 21)
    npt.assert_allclose(resistant_mean(input, 2), 7 + 16j)

def test_res_mean_complex_float():
    input = np.linspace(0 + 10j, 10 + 30j, 20)
    npt.assert_allclose(resistant_mean(input, 3), 2.6315789872949775 + 15.263158017938787j)

def test_res_mean_complex_large_i():
    input = np.concatenate([np.linspace(0, 19+19j,20), np.array([1 + 100j, 3 + 400j, 5 + 500j])])
    npt.assert_allclose(resistant_mean(input, 3), 9.5 + 9.5j)

def test_res_mean_random_large():
    input = np.concatenate([np.linspace(0, 10, 100_000), np.arange(-1_000_000, 1_000_000, 1000)])
    npt.assert_allclose(resistant_mean(input, 4), 4.9998998746918923, atol=1e-4)