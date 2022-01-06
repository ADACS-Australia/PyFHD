import numpy as np
from pathlib import Path
from tests.test_utils import get_data_items
from fhd_utils.idl_tools.array_match import array_match
import pytest

@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return list(Path.glob(Path.cwd(), '**/array_match/'))[0]

def test_array_match_1(data_dir):
    array1, array2, value_match, expected_indices, expected_n_match =  get_data_items(
                                                                                        data_dir, 
                                                                                        'vis_weights_update_input_array1.npy', 
                                                                                        'vis_weights_update_input_array2.npy', 
                                                                                        'vis_weights_update_input_value_match.npy',
                                                                                        'vis_weights_update_output_match_indices.npy',
                                                                                        'vis_weights_update_output_n_match.npy'
                                                                                     )

    # Get the result and see if they match.
    indices, n_match = array_match(array1, value_match, array_2 = array2)
    assert np.array_equal(indices, expected_indices)
    assert n_match == expected_n_match

def test_array_match_2(data_dir):
    array1, value_match, expected_indices, expected_n_match =  get_data_items(
        data_dir, 
        'input_array1_2.npy', 
        'input_value_match_2.npy',
        'output_match_indices_2.npy',
        'output_n_match_2.npy'
    )
    # Get the result and see if they match.
    indices, n_match = array_match(array1, value_match)
    print(indices.shape)
    print(expected_indices.shape)
    print(indices - expected_indices)
    assert np.array_equal(indices, expected_indices)
    assert n_match == expected_n_match

def test_array_match_3(data_dir):
    array1, array2, value_match, expected_indices, expected_n_match =  get_data_items(
        data_dir, 
        'input_array1_3.npy', 
        'input_array2_3.npy', 
        'input_value_match_3.npy',
        'output_match_indices_3.npy',
        'output_n_match_3.npy'
    )
    # Get the result and see if they match.
    indices, n_match = array_match(array1, value_match, array_2 = array2)
    assert np.array_equal(indices, expected_indices)
    assert n_match == expected_n_match