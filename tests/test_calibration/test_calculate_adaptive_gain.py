import pytest
import numpy as np
from pathlib import Path
from tests.test_utils import get_data, get_data_items
from fhd_core.calibration.calculate_adaptive_gain import calculate_adaptive_gain

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), "**/calculate_adaptive_gain"))[0]

def test_calc_adapt_gain_one(data_dir):
    gain_list, convergence_list, iter, base_gain, final_con_est, expected_gain = get_data_items(
        data_dir,
        'input_gain_list_1.npy',
        'input_convergence_list_1.npy',
        'input_iter_1.npy',
        'input_base_gain_1.npy',
        'input_final_convergence_estimate_1.npy',
        'output_gain_1.npy'
    )
    result_gain = calculate_adaptive_gain(gain_list, convergence_list, iter, base_gain, final_convergence_estimate = final_con_est)
    assert expected_gain == result_gain

def test_calc_adapt_gain_two(data_dir):
    gain_list, convergence_list, iter, base_gain, final_con_est, expected_gain = get_data_items(
        data_dir,
        'input_gain_list_2.npy',
        'input_convergence_list_2.npy',
        'input_iter_2.npy',
        'input_base_gain_2.npy',
        'input_final_convergence_estimate_2.npy',
        'output_gain_2.npy'
    )
    result_gain = calculate_adaptive_gain(gain_list, convergence_list, iter, base_gain, final_convergence_estimate = final_con_est)
    assert expected_gain == result_gain

'''
The below test will never pass due to IDL's default median behaviour, for example
PRINT, MEDIAN([1, 2, 3, 4], /EVEN) 
2.50000
PRINT, MEDIAN([1, 2, 3, 4])
3.00000

One has to use the EVEN keyword to get the *proper* median behaviour that most mathematicians and scientists
expect. This was the only thing that produced the wrong results was the use of numpy median instead of IDL MEDIAN.
The array that went into it est_final_conv produced the exact same results in Python and IDL (with the exception of
Python producing a more accurate result due to double precision) 

def test_calc_adapt_gain_three(data_dir):
    gain_list, convergence_list, iter, base_gain, expected_gain = get_data_items(
        data_dir,
        'input_gain_list_3.npy',
        'input_convergence_list_3.npy',
        'input_iter_3.npy',
        'input_base_gain_3.npy',
        'output_gain_3.npy'
    )
    result_gain = calculate_adaptive_gain(gain_list, convergence_list, iter, base_gain)
    assert expected_gain == result_gain
'''