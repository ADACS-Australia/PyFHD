import numpy as np
from pathlib import Path
import pytest
from fhd_core.gridding.interpolate_kernel import interpolate_kernel
from tests.test_utils import get_data_items

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), '**/interpolate_kernel/'))[0]

def test_interpolate_kernel_one(data_dir):
    kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1, expected_kernel = get_data_items(
        data_dir,
        'visibility_grid_input_kernel_arr_1.npy',
        'visibility_grid_input_x_offset_1.npy',
        'visibility_grid_input_y_offset_1.npy',
        'visibility_grid_input_dx0dy0_1.npy',
        'visibility_grid_input_dx1dy0_1.npy',
        'visibility_grid_input_dx0dy1_1.npy',
        'visibility_grid_input_dx1dy1_1.npy',
        'visibility_grid_output_kernel_1.npy'
    )
    kernel = interpolate_kernel(kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1)
    assert np.array_equal(kernel, expected_kernel)

def test_interpolate_kernel_two(data_dir):
    kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1, expected_kernel = get_data_items(
        data_dir,
        'visibility_grid_input_kernel_arr_2.npy',
        'visibility_grid_input_x_offset_2.npy',
        'visibility_grid_input_y_offset_2.npy',
        'visibility_grid_input_dx0dy0_2.npy',
        'visibility_grid_input_dx1dy0_2.npy',
        'visibility_grid_input_dx0dy1_2.npy',
        'visibility_grid_input_dx1dy1_2.npy',
        'visibility_grid_output_kernel_2.npy'
    )
    kernel = interpolate_kernel(kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1)
    assert np.array_equal(kernel, expected_kernel)
    
def test_interpolate_kernel_three(data_dir):
    # TODO: Its failing, precision errors? Its such a simple function, I'm not sure what else.
    kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1, expected_kernel = get_data_items(
        data_dir,
        'visibility_grid_input_kernel_arr_3.npy',
        'visibility_grid_input_x_offset_3.npy',
        'visibility_grid_input_y_offset_3.npy',
        'visibility_grid_input_dx0dy0_3.npy',
        'visibility_grid_input_dx1dy0_3.npy',
        'visibility_grid_input_dx0dy1_3.npy',
        'visibility_grid_input_dx1dy1_3.npy',
        'visibility_grid_output_kernel_3.npy'
    )
    print(kernel_arr)
    kernel = interpolate_kernel(kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1)
    errors = kernel - expected_kernel
    errors = errors[np.nonzero(errors)]
    print(np.mean(errors))
    percent = np.round((np.size(np.nonzero(kernel - expected_kernel)) / np.size(kernel)) * 100, 2)
    print("{}% of the values were wrong".format(percent))
    print("kernel dtype: {} expected_kernel dtype: {}".format(kernel.dtype, expected_kernel.dtype))
    assert np.array_equal(kernel, expected_kernel)
