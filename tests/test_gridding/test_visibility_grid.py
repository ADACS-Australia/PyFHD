import pytest
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.pyfhd_tools.test_utils import get_savs

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'visibility_grid')

@pytest.fixture
def full_data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'full_size_visibility_grid/')

def test_visibility_grid_one(data_dir):
    # This is the way
    inputs = get_savs(data_dir,'input_1.sav')
    visibility = inputs['visibility_ptr']
    vis_weights = inputs['vis_weight_ptr']
    obs = inputs['obs']
    status_str = inputs['status_str']
    psf = inputs['psf']
    params = inputs['params']
    weights_flag = inputs['weights']
    variance_flag = inputs['variance']
    polarization = inputs['polarization']
    map_flag = inputs['mapfn_recalculate']
    uniform_flag = False
    fi_use = inputs['fi_use']
    bi_use = inputs['bi_use']
    no_conjugate = False
    mask_mirror_indices = False
    model = inputs['model_ptr']
    grid_spectral = False
    beam_per_baseline = False
    uv_grid_phase_only = True

    outputs = get_savs(data_dir, 'output_1.sav')

    image_uv, weights, variance, uniform_filter, obs = visibility_grid(
        visibility,
        vis_weights,
        obs, 
        status_str,
        psf, 
        params,
        weights_flag = weights_flag,
        variance_flag = variance_flag,
        polarization = polarization,
        map_flag = map_flag,
        uniform_flag = uniform_flag,
        fi_use = fi_use,
        bi_use = bi_use,
        no_conjugate = no_conjugate,
        mask_mirror_indices = mask_mirror_indices,
        model = model,
        grid_spectral = grid_spectral,
        beam_per_baseline = beam_per_baseline,
        uv_grid_phase_only = uv_grid_phase_only,
    )

    npt.assert_allclose(image_uv, outputs['image_uv'], atol = 1e-8)
    npt.assert_allclose(weights, outputs['weights'], atol = 1e-8)
    npt.assert_allclose(variance, outputs['variance'], atol = 1e-8)
    npt.assert_allclose(uniform_filter, outputs['uniform_filter'], atol = 1e-8)
    npt.assert_allclose(obs['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)