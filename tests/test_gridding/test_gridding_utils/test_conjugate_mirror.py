import pytest
import numpy as np
from glob import glob
from PyFHD.gridding.gridding_utils import conjugate_mirror
from PyFHD.pyfhd_tools.test_utils import get_data_items

@pytest.fixture
def data_dir():
    return glob("**/conjugate_mirror", recursive = True)[0]

def test_conj_mirror_one(data_dir):
    input, expected_image = get_data_items(data_dir, 
                                     'visibility_grid_input_1.npy',
                                     'visibility_grid_output_1.npy')
    image = conjugate_mirror(input)
    assert np.array_equal(image, expected_image)

def test_conj_mirror_two(data_dir):
    input, expected_image = get_data_items(data_dir, 
                                     'visibility_grid_input_2.npy',
                                     'visibility_grid_output_2.npy')
    image = conjugate_mirror(input)
    assert np.array_equal(image, expected_image)

def test_conj_mirror_three(data_dir):
    input, expected_image = get_data_items(data_dir, 
                                     'visibility_grid_input_3.npy',
                                     'visibility_grid_output_3.npy')
    image = conjugate_mirror(input)
    assert np.array_equal(image, expected_image)