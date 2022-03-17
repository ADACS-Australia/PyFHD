import pytest
import numpy as np
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from PyFHD.gridding.visibility_degrid import visibility_degrid
from PyFHD.pyfhd_tools.test_utils import get_savs

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'visibility_degrid/')

@pytest.fixture
def full_data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'full_size_visibility_degrid/')

def test_visibility_degrid_one(data_dir):
    inputs = get_savs(data_dir,'input_1.sav')
    image_uv = inputs['image_uv']
    vis_weights = inputs['vis_weight_ptr'] 
    obs = inputs['obs'] 
    psf = inputs['psf']
    params = inputs['params']
    polarization = inputs['polarization']
    fill_model_visibilities = inputs['fill_model_visibilities']
    vis_input = None
    spectral_model_uv_arr = inputs['spectral_model_uv_arr']
    beam_per_baseline = False
    uv_grid_phase_only = True
    conserve_memory = inputs['conserve_memory']
    memory_threshold = 1e8

    vis_return = visibility_degrid(
        image_uv,
        vis_weights, 
        obs,
        psf,
        params,
        polarization = polarization,
        fill_model_visibilities = fill_model_visibilities,
        vis_input = vis_input,
        spectral_model_uv_arr = spectral_model_uv_arr,
        beam_per_baseline = beam_per_baseline,
        uv_grid_phase_only = uv_grid_phase_only,
        conserve_memory = conserve_memory,
        memory_threshold = memory_threshold
    )

    outputs = get_savs(data_dir,'output_1.sav')

    npt.assert_allclose(vis_return.real, outputs['vis_return'].real, atol = 1e-3)
    npt.assert_allclose(vis_return.imag, outputs['vis_return'].imag, atol = 1e-3)

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

    npt.assert_allclose(vis_return.real, expected_vis_return.real, atol = 1e-3)
    npt.assert_allclose(vis_return.imag, expected_vis_return.imag, atol = 1e-3)


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

    vis_return, obs = visibility_degrid(
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

    npt.assert_allclose(vis_return.real, expected_vis_return.real, atol = 1e-3)
    npt.assert_allclose(vis_return.imag, expected_vis_return.imag, atol = 1e-3)