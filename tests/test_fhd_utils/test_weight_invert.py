import pytest
import numpy as np
from pathlib import Path
from tests.test_utils import get_data_items
from fhd_utils.weight_invert import weight_invert

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), "**/weight_invert"))[0]

def test_weight_invert_one(data_dir):
    threshold, weights, expected_result = get_data_items(data_dir, 'visibility_grid_input_threshold_1.npy', 
                                                                   'visibility_grid_input_weights_1.npy', 
                                                                   'visibility_grid_output_result_1.npy')
    result = weight_invert(weights, threshold = threshold)
    assert np.array_equal(result, expected_result)

def test_weight_invert_two(data_dir):
    threshold, weights, expected_result = get_data_items(data_dir, 
                                                         'visibility_grid_input_threshold_2.npy', 
                                                         'visibility_grid_input_weights_2.npy', 
                                                         'visibility_grid_output_result_2.npy')
    result = weight_invert(weights, threshold = threshold)
    assert np.array_equal(result, expected_result)

def test_weight_invert_three(data_dir):
    threshold, weights, expected_result, abs = get_data_items(data_dir, 
                                                              'visibility_grid_input_threshold_3.npy', 
                                                              'visibility_grid_input_weights_3.npy', 
                                                              'visibility_grid_output_result_3.npy',
                                                              'visibility_grid_input_abs_3.npy')
    result = weight_invert(weights, threshold = threshold)
    assert np.array_equal(result, expected_result)

def test_weight_invert_four(data_dir):
    weights, expected_result = get_data_items(data_dir,  
                                              'visibility_grid_input_weights_4.npy', 
                                              'visibility_grid_output_result_4.npy')
    result = weight_invert(weights)
    assert np.array_equal(result, expected_result)