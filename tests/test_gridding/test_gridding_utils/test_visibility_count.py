import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data, get_data_items
from PyFHD.gridding.gridding_utils import visibility_count

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'visibility_count')

@pytest.fixture(scope="function", params=[1, 2, 3])
def number(request):
    return request.param

@pytest.fixture
def vis_count_before(data_dir, number):
    vis_count_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if vis_count_before.exists():
        return vis_count_before
    
@pytest.fixture
def vis_count_after(data_dir, number):
    vis_count_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if vis_count_after.exists():
        return vis_count_after

def test_vis_count_one(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
        'input_psf_1.npy',
    )
    file_path_fhd, obs, params, vis_weights, expected_uniform_filter = get_data_items(
        data_dir,
        'input_file_path_fhd_1.npy',
        'input_obs_1.npy',
        'input_params_1.npy',
        'input_vis_weight_ptr_1.npy',
        'output_uniform_filter_1.npy',
    )
    uniform_filter = visibility_count(
        obs, 
        psf, 
        params, 
        vis_weights, 
        file_path_fhd = file_path_fhd
    )
    # Due to precision errors from baseline_grid_locations we also have to check 
    # that uniform_filter meets a threshold rather than exactly. 
    # The threshold is 0.0015% in this instance, as only one ymin is "wrong" here.
    # ymin is not "wrong", its more like IDL's floor function was "wrong" due to single precision.
    uniform_wrong =  np.nonzero(np.abs(expected_uniform_filter - uniform_filter))[0].size
    assert (uniform_wrong / uniform_filter.size) < 0.0015

def test_vis_count_two(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
        'input_psf_2.npy',
    )
    file_path_fhd, obs, params, vis_weights, fill_model_vis, no_conj, expected_uniform_filter = get_data_items(
        data_dir,
        'input_file_path_fhd_2.npy',
        'input_obs_2.npy',
        'input_params_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_fill_model_vis_2.npy',
        'input_no_conjugate_2.npy',
        'output_uniform_filter_2.npy',
    )
    uniform_filter = visibility_count(
        obs, 
        psf, 
        params, 
        vis_weights[0], 
        file_path_fhd = file_path_fhd, 
        fill_model_visibilities = fill_model_vis,
        no_conjugate = no_conj,
    )
    # Due to precision errors from baseline_grid_locations we also have to check 
    # that uniform_filter meets a threshold rather than exactly. 
    # The threshold is 0.0015% in this instance, as only one ymin is "wrong" here.
    # ymin is not "wrong", its more like IDL's floor function was "wrong" due to single precision.
    uniform_wrong =  np.nonzero(np.abs(expected_uniform_filter - uniform_filter))[0].size
    assert (uniform_wrong / uniform_filter.size) < 0.0015

def test_vis_count_three(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
        'input_psf_3.npy',
    )
    file_path_fhd, obs, params, vis_weights, mm_inds, expected_uniform_filter = get_data_items(
        data_dir,
        'input_file_path_fhd_3.npy',
        'input_obs_3.npy',
        'input_params_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_mask_mirror_indices_3.npy',
        'output_uniform_filter_3.npy',
    )
    uniform_filter = visibility_count(
        obs, 
        psf, 
        params, 
        vis_weights, 
        file_path_fhd = file_path_fhd,
        mask_mirror_indices = mm_inds,
    )
    # Due to precision errors from baseline_grid_locations we also have to check 
    # that uniform_filter meets a threshold rather than exactly. 
    # The threshold is 1.75% in this instance.
    # This all spawned from a maximum difference of 6.1035e-05 between any element of xcen and ycen
    # That led to a 49 difference between any element in x_offset and y_offset
    # This led to changes in xmin and ymin in many elements, only ~0.00067% of elements to be exact
    # That change in xmin and ymin led to ~65% of bin_i being wrong, and 0.12% of the histogram being wrong
    # This affects the creation of the uniform_filter vastly as bin_i feeds the values for uniform_filter
    # All of that from precision errors in xcen and ycen, kind of amazing.
    uniform_wrong =  np.nonzero(np.abs(expected_uniform_filter - uniform_filter))[0].size
    assert (uniform_wrong / uniform_filter.size) < 0.0175
