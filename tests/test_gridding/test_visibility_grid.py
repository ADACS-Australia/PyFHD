import pytest
import numpy as np
from glob import glob
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.pyfhd_tools.test_utils import get_data, get_data_items

@pytest.fixture
def data_dir():
    return glob('../**/visibility_grid/', recursive = True)[0]

@pytest.fixture
def full_data_dir():
    return glob('../**/full_size_visibility_grid/', recursive = True)[0]

def test_visibility_grid_one(data_dir):
    psf = get_data(
        data_dir,
        'input_psf_1.npy',
    )

    map_flag, model, no_conjugate, params, polarization, return_mapfn, status_str,\
    uniform_flag, visibility, vis_weights, weights_flag, obs, \
    expected_image_uv, expected_uniform_filter, expected_weights = get_data_items(
        data_dir,
        'input_mapfn_recalculate_1.npy',
        'input_model_ptr_1.npy',
        'input_no_conjugate_1.npy',
        'input_params_1.npy',
        'input_polarization_1.npy',
        'input_return_mapfn_1.npy',
        'input_status_str_1.npy',
        'input_uniform_filter_1.npy',
        'input_visibility_ptr_1.npy',
        'input_vis_weight_ptr_1.npy',
        'input_weights_1.npy',
        'output_image_uv_1.npy',
        'output_uniform_filter_1.npy',
        'output_weights_1.npy',
    )

    image_uv, weights, _, uniform_filter = visibility_grid(
        visibility, 
        vis_weights, 
        obs, 
        status_str, 
        psf, 
        params,
        weights_flag = weights_flag,
        map_flag = map_flag,
        model = model,
        no_conjugate = no_conjugate,
        polarization = polarization,
        return_mapfn = return_mapfn,
        uniform_flag = uniform_flag,
    )

    assert np.array_equal(image_uv, expected_image_uv)
    assert np.array_equal(uniform_filter, expected_uniform_filter)
    assert np.array_equal(weights, expected_weights)

def test_visibility_grid_two(data_dir):
    psf = get_data(
        data_dir,
        'input_psf_2.npy',
    )

    map_flag, model, params, polarization, status_str, visibility, vis_weights,\
    weights_flag, obs, bi_use, fi_use,  variance_flag, \
    expected_image_uv, expected_variance, expected_weights = get_data_items(
        data_dir,
        'input_mapfn_recalculate_2.npy',
        'input_model_ptr_2.npy',
        'input_params_2.npy',
        'input_polarization_2.npy',
        'input_status_str_2.npy',
        'input_visibility_ptr_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_weights_holo_2.npy',
        'input_obs_2.npy',
        'input_bi_use_2.npy',
        'input_fi_use_2.npy',
        'input_variance_holo_2.npy',
        'output_image_uv_2.npy',
        'output_variance_2.npy',
        'output_weights_2.npy',
    )

    image_uv, weights, variance, _ = visibility_grid(
        visibility, 
        vis_weights, 
        obs, 
        status_str, 
        psf, 
        params,
        weights_flag = weights_flag,
        map_flag = map_flag,
        model = model,
        polarization = polarization,
        variance_flag = variance_flag,
        bi_use = bi_use,
        fi_use = fi_use,
    )

    assert np.array_equal(image_uv, expected_image_uv)
    assert np.array_equal(variance, expected_variance)
    assert np.array_equal(weights, expected_weights)

def test_visibility_grid_three(data_dir):
    psf = get_data(
        data_dir,
        'input_psf_3.npy',
    )

    map_flag, model, no_conjugate, params, polarization, return_mapfn, status_str,\
    uniform_flag, visibility, vis_weights, weights_flag, obs, \
    expected_image_uv, expected_uniform_filter, expected_weights = get_data_items(
        data_dir,
        'input_mapfn_recalculate_3.npy',
        'input_model_ptr_3.npy',
        'input_no_conjugate_3.npy',
        'input_params_3.npy',
        'input_polarization_3.npy',
        'input_return_mapfn_3.npy',
        'input_status_str_3.npy',
        'input_uniform_filter_3.npy',
        'input_visibility_ptr_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_weights_3.npy',
        'input_obs_3.npy',
        'output_image_uv_3.npy',
        'output_timing_3.npy',
        'output_uniform_filter_3.npy',
        'output_weights_3.npy',
    )

    image_uv, weights, _, uniform_filter = visibility_grid(
        visibility, 
        vis_weights, 
        obs, 
        status_str, 
        psf, 
        params,
        weights_flag = weights_flag,
        map_flag = map_flag,
        model = model,
        no_conjugate = no_conjugate,
        polarization = polarization,
        return_mapfn = return_mapfn,
        uniform_flag = uniform_flag,
    )

    assert np.array_equal(image_uv, expected_image_uv)
    assert np.array_equal(uniform_filter, expected_uniform_filter)
    assert np.array_equal(weights, expected_weights)
    
