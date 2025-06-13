import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items
from PyFHD.pyfhd_tools.pyfhd_utils import deriv_coefficients


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "pyfhd_tools", "deriv_coefficients")


def test_deriv_coefficients_one(data_dir):
    n, divide_fact, expected = get_data_items(
        data_dir, "input_n_1.npy", "input_divide_factorial_1.npy", "output_coeff_1.npy"
    )
    result = deriv_coefficients(n, divide_factorial=divide_fact)
    # Precision Errors strike back, maximum difference was 2.2e-5, some were less difference than what single precision can do.
    assert np.max(result - expected) < 1e-5


def test_deriv_coefficients_two(data_dir):
    n, expected = get_data_items(data_dir, "input_n_2.npy", "output_coeff_2.npy")
    result = deriv_coefficients(n)
    assert np.array_equal(result, expected)


def test_deriv_coefficients_three(data_dir):
    n, expected = get_data_items(data_dir, "input_n_3.npy", "output_coeff_3.npy")
    result = deriv_coefficients(n)
    assert np.array_equal(result, expected)
