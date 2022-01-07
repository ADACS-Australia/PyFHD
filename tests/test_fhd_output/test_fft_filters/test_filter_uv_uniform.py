import pytest
import numpy as np
from glob import glob
from fhd_output.fft_filters.filter_uv_uniform import filter_uv_uniform
from tests.test_utils import get_data_items

@pytest.fixture
def data_dir():
    return glob('**/filter_uv_uniform/', recursive = True)[0]

def test_filter_uni_one(data_dir):
    image_uv, return_name_only, expected_image_uv, expected_name = get_data_items(
        data_dir,
        'input_image_uv_1.npy',
        'input_return_name_only_1.npy',
        'output_image_uv_filtered_1.npy',
        'output_name_1.npy'
    )
    image_uv_filtered, name = filter_uv_uniform(image_uv, return_name_only = return_name_only)
    assert np.array_equal(image_uv_filtered, expected_image_uv)
    assert expected_name.decode("utf-8") == name

def test_filter_uni_two(data_dir):
    image_uv, vis_count, expected_image_uv, expected_filter = get_data_items(
        data_dir,
        'input_image_uv_2.npy',
        'restored_vis_count_2.npy',
        'output_image_uv_filtered_2.npy',
        'output_filter_2.npy'
    )
    image_uv_filtered, filter = filter_uv_uniform(image_uv, vis_count = vis_count)
    # Check the result precision error is beyond single precision
    assert  1e-8 > np.max(np.abs(expected_image_uv.real - image_uv_filtered.real))
    assert  1e-8 > np.max(np.abs(expected_image_uv.imag - image_uv_filtered.imag))
    # Check the result precision error is beyond single precision
    assert 1e-8 > np.max(np.abs(filter - expected_filter))

def test_filter_uni_three(data_dir):
    image_uv, vis_count, weights, expected_image_uv, expected_filter = get_data_items(
        data_dir,
        'input_image_uv_3.npy',
        'restored_vis_count_3.npy',
        'input_weights_3.npy',
        'output_image_uv_filtered_3.npy',
        'output_filter_3.npy'
    )
    image_uv_filtered, filter = filter_uv_uniform(image_uv, vis_count = vis_count, weights = weights)
    # Check the result precision error is beyond a threshold (single precision rounding errors have occurred)
    assert  1e-8 > np.max(np.abs(expected_image_uv.real - image_uv_filtered.real))
    assert  1e-8 > np.max(np.abs(expected_image_uv.imag - image_uv_filtered.imag))
    # Check the result precision error is beyond a threshold (single precision rounding errors have occurred)
    assert 1e-8 > np.max(np.abs(filter - expected_filter))