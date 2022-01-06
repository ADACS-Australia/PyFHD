import pytest
import numpy as np
from pathlib import Path
from tests.test_utils import get_data_items
from fhd_utils.FFT.deriv_coefficients import deriv_coefficients

@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return list(Path.glob(Path.cwd(), '**/deriv_coefficients/'))[0]

def test_deriv_coefficients_one(data_dir):
    n, divide_fact, expected = get_data_items(
                                                data_dir,
                                                'input_n_1.npy',
                                                'input_divide_factorial_1.npy',
                                                'output_coeff_1.npy'
                                             )
    result = deriv_coefficients(n, divide_factorial = divide_fact)
    # Precision Errors strike back, maximum difference was 2.2e-5, some were less difference than what single precision can do.
    assert np.max(result - expected) < 1e-5

def test_deriv_coefficients_two(data_dir):
    n, expected = get_data_items(
                                                data_dir,
                                                'input_n_2.npy',
                                                'output_coeff_2.npy'
                                             )
    result = deriv_coefficients(n)
    assert np.array_equal(result, expected)

def test_deriv_coefficients_three(data_dir):
    n, expected = get_data_items(
                                                data_dir,
                                                'input_n_3.npy',
                                                'output_coeff_3.npy'
                                             )
    result = deriv_coefficients(n)
    assert np.array_equal(result, expected)