from PyFHD.pyfhd_tools.pyfhd_utils import resistant_mean
import numpy as np
from numpy import testing as npt
import pytest

def test_res_mean_int():
    input = np.concatenate(np.arange(20), np.array([100,200,300,400]))
    assert resistant_mean(input, 2) == 9.5

def test_res_mean_float():
    input = np.arange(0, 20, 0.75), np.array([25.0, -10.75, -30.0, 50])
    assert resistant_mean(input, 2) == 9.75

def test_res_mean_complex():
    input = np.arange(0 + 10j, 10+ 30j, 0.75 + 0.1j)
    assert resistant_mean(input, 3) == 4.5 + 10.6j

def test_res_mean_random_large():
    input = np.concatenate([np.linspace(0, 10, 100_000), np.arange(-1_000_000, 1_000_000, 1000)])
    npt.assert_allclose(resistant_mean(input, 4), 4.9998998746918923)

