import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from glob import glob
from fhd_core.gridding.visibility_degrid import visibility_degrid
from tests.test_utils import get_data, get_data_items

@pytest.fixture
def data_dir():
    return glob('**/visibility_degrid/', recursive = True)[0]

@pytest.fixture
def full_data_dir():
    return glob('**/full_size_visibility_degrid/', recursive = True)[0]

def test_visibility_degrid_one(data_dir):
    psf = get_data(
        data_dir,
        'input_psf_1.npy',
    )
    image_uv, vis_weights, obs, params, polarization,\
    fill_model_visibilities, spectral_model_uv_arr, conserve_memory,\
    expected_vis_return = get_data_items(
        data_dir,
        'image_uv_1.npy',
        'input_vis_weight_ptr_1.npy',
        'input_obs_1.npy',
        'input_params_1.npy',
        'input_polarization_1.npy',
        'input_fill_model_visibilities_1.npy',
        'input_spectral_model_uv_arr_1.npy',
        'input_conserve_memory_1.npy',
        'output_vis_return_1.npy'
    )

    vis_return = visibility_degrid(
        image_uv, 
        vis_weights, 
        obs, 
        psf,
        params,
        polarization = polarization,
        fill_model_visibilities = fill_model_visibilities,
        spectral_model_uv_arr = spectral_model_uv_arr,
        conserve_memory = conserve_memory
    )

    assert assert_almost_equal(vis_return, expected_vis_return, decimal = 7)

def test_visibility_degrid_two(data_dir):

    psf = get_data(
        data_dir,
        'input_psf_2.npy',
    )
    image_uv, vis_weights, obs, params, polarization,\
    fill_model_visibilities, spectral_model_uv_arr, conserve_memory,\
    expected_vis_return = get_data_items(
        data_dir,
        'image_uv_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_obs_2.npy',
        'input_params_2.npy',
        'input_polarization_2.npy',
        'input_fill_model_visibilities_2.npy',
        'input_spectral_model_uv_arr_2.npy',
        'input_conserve_memory_2.npy',
        'output_vis_return_2.npy'
    )

    vis_return = visibility_degrid(
        image_uv, 
        vis_weights, 
        obs, 
        psf, 
        params,
        polarization = polarization,
        fill_model_visibilities = fill_model_visibilities,
        spectral_model_uv_arr = spectral_model_uv_arr,
        conserve_memory = conserve_memory
    )

    assert assert_almost_equal(vis_return, expected_vis_return, decimal = 7)


def test_visibility_degrid_three(data_dir):
    psf = get_data(
        data_dir,
        'input_psf_3.npy',
    )
    image_uv, vis_weights, obs, params, polarization,\
    fill_model_visibilities, spectral_model_uv_arr, beam_per_baseline, \
    uv_grid_phase_only, conserve_memory, expected_vis_return = get_data_items(
        data_dir,
        'image_uv_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_obs_3.npy',
        'input_params_3.npy',
        'input_polarization_3.npy',
        'input_fill_model_visibilities_3.npy',
        'input_spectral_model_uv_arr_3.npy',
        'input_beam_per_baseline_3.npy',
        'input_uv_grid_phase_only_3.npy',
        'input_conserve_memory_3.npy',
        'output_vis_return_3.npy'
    )

    vis_return = visibility_degrid(
        image_uv, 
        vis_weights, 
        obs, 
        psf, 
        params,
        polarization = polarization,
        fill_model_visibilities = fill_model_visibilities,
        spectral_model_uv_arr = spectral_model_uv_arr,
        beam_per_baseline = beam_per_baseline,
        uv_grid_phase_only = uv_grid_phase_only,
        conserve_memory = conserve_memory
    )

    assert np.array_equal(vis_return, expected_vis_return)