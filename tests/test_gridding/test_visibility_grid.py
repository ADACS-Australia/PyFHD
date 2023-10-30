from PyFHD.io.pyfhd_io import recarray_to_dict
import pytest
import numpy.testing as npt
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.pyfhd_tools.test_utils import get_savs
from PyFHD.io.pyfhd_io import save, load
from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'visibility_grid')

@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6, 7])
def number(request: pytest.FixtureRequest):
    return request.param

@pytest.fixture
def before_gridding(data_dir: Path, number: int):
    before_gridding = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if before_gridding.exists():
        return before_gridding
    
    h5_save_dict = get_savs(data_dir,f'input_{number}.sav')
    # Take out and copy the beam_ptr to keep its structure
    # Going to leave as an object array due to the size of it being
    # 2 * 8128 * 51 * 51 * 196, which is 123GiB for storing it as a
    # np.complex128 array. Too big for most machines except for HPC.
    beam_ptr = np.copy(h5_save_dict["psf"]["beam_ptr"][0].T)
    h5_save_dict = recarray_to_dict(h5_save_dict)
    h5_save_dict["psf"]["beam_ptr"] = beam_ptr
    h5_save_dict["uniform_flag"] = True if ("uniform_filter" in h5_save_dict and h5_save_dict["uniform_filter"]) else False
    h5_save_dict["no_conjugate"] = True if ("no_conjugate" in h5_save_dict and h5_save_dict["no_conjugate"]) else False
    h5_save_dict["obs"]["n_baselines"] = h5_save_dict["obs"]["nbaselines"]
    # Transpose the model if it exists
    if "model_ptr" in h5_save_dict and h5_save_dict["model_ptr"] is not None:
        h5_save_dict["model_ptr"] = h5_save_dict['model_ptr'].T
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": h5_save_dict["psf"]["interpolate_kernel"],
        "psf_dim": h5_save_dict["psf"]["dim"],
        "psf_resolution": h5_save_dict["psf"]["resolution"],
        "beam_mask_threshold": h5_save_dict["psf"]["beam_mask_threshold"],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "image_filter": h5_save_dict["extra"]["image_filter_fn"],
        "mask_mirror_indices": False,
        "beam_per_baseline": True if ("beam_per_baseline" in h5_save_dict and h5_save_dict['beam_per_baseline']) else False,
        "grid_spectral": True if ("grid_spectral" in h5_save_dict and h5_save_dict['grid_spectral']) else False,
        "grid_weights": True if h5_save_dict['weights'] else False,
        "grid_variance": True if ("variance" in h5_save_dict and h5_save_dict['variance']) else False,
        "grid_uniform": True if ("grid_uniform" in h5_save_dict and h5_save_dict["grid_uniform"]) else False
    }
    h5_save_dict['visibility_ptr'] = h5_save_dict['visibility_ptr'].T
    h5_save_dict["vis_weight_ptr"] = h5_save_dict["vis_weight_ptr"].T
    if 'fi_use' not in h5_save_dict:
        h5_save_dict['fi_use'] = None
    else:
        if not isinstance(h5_save_dict['fi_use'], np.ndarray):
            # Assume fi_use is an integer, make it an array
            h5_save_dict['fi_use'] = np.array([h5_save_dict['fi_use']], dtype = np.int64)
    if 'bi_use' not in h5_save_dict:
        h5_save_dict['bi_use'] = None
    else:
        if not isinstance(h5_save_dict['bi_use'], np.ndarray):
            h5_save_dict['bi_use'] = np.array([h5_save_dict['bi_use']], dtype = np.int64)
    dd.io.save(before_gridding, h5_save_dict)
    
    return before_gridding

@pytest.fixture
def after_gridding(data_dir: Path, number: int):
    after_gridding = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if after_gridding.exists():
        return after_gridding
    
    outputs = recarray_to_dict(get_savs(data_dir, f'output_{number}.sav'))

    h5_save_dict = {
        'image_uv': outputs['image_uv'],
        'weights': outputs['weights'],
        'variance': outputs['variance'],
        'uniform_filter': outputs['uniform_filter'],
        'nf_vis': outputs['obs']['nf_vis']
    }

    if 'model_return' in outputs:
        h5_save_dict["model_return"] = outputs["model_return"]

    dd.io.save(after_gridding, h5_save_dict)
    
    return after_gridding

def test_visibility_grid(before_gridding: Path, after_gridding: Path):
    h5_before = load(before_gridding)
    h5_after = load(after_gridding)

    h5_before["psf"]["id"] = h5_before["psf"]["id"].T

    gridding_dict = visibility_grid(
        h5_before["visibility_ptr"],
        h5_before['vis_weight_ptr'],
        h5_before['obs'],
        h5_before['psf'],
        h5_before['params'],
        h5_before['polarization'],
        h5_before["pyfhd_config"],
        RootLogger(1),
        uniform_flag = h5_before["uniform_flag"],
        no_conjugate = h5_before["no_conjugate"],
        model = h5_before["model_ptr"],
        fi_use = h5_before['fi_use'],
        bi_use = h5_before['bi_use']
    )
    # All atols are done by the lowest precision that passed for ALL tests
    npt.assert_allclose(gridding_dict['image_uv'], h5_after['image_uv'], atol = 1.5e-7)
    npt.assert_allclose(gridding_dict['weights'], h5_after['weights'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['variance'], h5_after['variance'], atol = 1e-8)
    # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
    # The minor difference in bin_n affected the uniform filter. The precision difference could cause errors upto 1
    # This doesn't occur for every test.
    npt.assert_allclose(gridding_dict['obs']['nf_vis'], h5_after['nf_vis'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['uniform_filter'], h5_after['uniform_filter'], atol = 0.5)

    if "model_return" in gridding_dict:
        npt.assert_allclose(gridding_dict['model_return'], h5_after['model_return'], atol = 1e-7)

# FULL SIZE TESTS BELOW

@pytest.fixture(scope="function", params=[1])
def full_number(request: pytest.FixtureRequest):
    return request.param

@pytest.fixture
def full_before_gridding(data_dir: Path, full_number: int):
    before_gridding = Path(data_dir, f"test_full_size_{full_number}_before_{data_dir.name}.h5")

    if before_gridding.exists():
        return before_gridding
    
    h5_save_dict = get_savs(data_dir,f'full_size_input_{full_number}.sav')
    # Take out and copy the beam_ptr to keep its structure
    # Going to leave as an object array due to the size of it being
    # 2 * 8128 * 51 * 51 * 196, which is 123GiB for storing it as a
    # np.complex128 array. Too big for most machines except for HPC.
    beam_ptr = np.copy(h5_save_dict["psf"]["beam_ptr"][0].T)
    h5_save_dict = recarray_to_dict(h5_save_dict)
    h5_save_dict["psf"]["beam_ptr"] = beam_ptr
    h5_save_dict["uniform_flag"] = True if ("uniform_filter" in h5_save_dict and h5_save_dict["uniform_filter"]) else False
    h5_save_dict["no_conjugate"] = True if ("no_conjugate" in h5_save_dict and h5_save_dict["no_conjugate"]) else False
    h5_save_dict["obs"]["n_baselines"] = h5_save_dict["obs"]["nbaselines"]
    # Transpose the model if it exists
    if "model_ptr" in h5_save_dict and h5_save_dict["model_ptr"] is not None:
        h5_save_dict["model_ptr"] = h5_save_dict['model_ptr'].T
    else: 
        h5_save_dict["model_ptr"] = None
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": h5_save_dict["psf"]["interpolate_kernel"],
        "psf_dim": h5_save_dict["psf"]["dim"],
        "psf_resolution": h5_save_dict["psf"]["resolution"],
        "beam_mask_threshold": h5_save_dict["psf"]["beam_mask_threshold"],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "image_filter": h5_save_dict["extra"]["image_filter_fn"],
        "mask_mirror_indices": False,
        "beam_per_baseline": True if ("beam_per_baseline" in h5_save_dict and h5_save_dict['beam_per_baseline']) else False,
        "grid_spectral": True if ("grid_spectral" in h5_save_dict and h5_save_dict['grid_spectral']) else False,
        "grid_weights": True if h5_save_dict['weights'] else False,
        "grid_variance": True if ("variance" in h5_save_dict and h5_save_dict['variance']) else False,
        "grid_uniform": True if ("grid_uniform" in h5_save_dict and h5_save_dict["grid_uniform"]) else False
    }
    h5_save_dict['visibility_ptr'] = h5_save_dict['visibility_ptr'].T
    h5_save_dict["vis_weight_ptr"] = h5_save_dict["vis_weight_ptr"].T
    if 'fi_use' not in h5_save_dict:
        h5_save_dict['fi_use'] = None
    else:
        if not isinstance(h5_save_dict['fi_use'], np.ndarray):
            # Assume fi_use is an integer, make it an array
            h5_save_dict['fi_use'] = np.array([h5_save_dict['fi_use']], dtype = np.int64)
    if 'bi_use' not in h5_save_dict:
        h5_save_dict['bi_use'] = None
    else:
        if not isinstance(h5_save_dict['bi_use'], np.ndarray):
            h5_save_dict['bi_use'] = np.array([h5_save_dict['bi_use']], dtype = np.int64)
    dd.io.save(before_gridding, h5_save_dict)
    
    return before_gridding

@pytest.fixture
def full_after_gridding(data_dir: Path, full_number: int):
    after_gridding = Path(data_dir, f"test_full_size_{full_number}_after_{data_dir.name}.h5")

    if after_gridding.exists():
        return after_gridding
    
    outputs = recarray_to_dict(get_savs(data_dir, f'full_size_output_{full_number}.sav'))

    h5_save_dict = {
        'image_uv': outputs['image_uv'],
        'weights': outputs['weights'],
        'variance': outputs['variance'],
        # 'uniform_filter': outputs['uniform_filter'],
        'nf_vis': outputs['obs']['nf_vis']
    }

    if 'model_return' in outputs:
        h5_save_dict["model_return"] = outputs["model_return"]

    dd.io.save(after_gridding, h5_save_dict)
    
    return after_gridding

def test_full_visibility_grid(full_before_gridding: Path, full_after_gridding: Path):
    h5_before = load(full_before_gridding)
    h5_after = load(full_after_gridding)

    gridding_dict = visibility_grid(
        h5_before["visibility_ptr"],
        h5_before['vis_weight_ptr'],
        h5_before['obs'],
        h5_before['psf'],
        h5_before['params'],
        h5_before['polarization'],
        h5_before["pyfhd_config"],
        RootLogger(1),
        uniform_flag = h5_before["uniform_flag"],
        no_conjugate = h5_before["no_conjugate"],
        model = h5_before["model_ptr"],
        fi_use = h5_before['fi_use'],
        bi_use = h5_before['bi_use']
    )
    # All atols are done by the lowest precision that passed for ALL tests
    npt.assert_allclose(gridding_dict['image_uv'], h5_after['image_uv'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['weights'], h5_after['weights'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['variance'], h5_after['variance'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['obs']['nf_vis'], h5_after['nf_vis'], atol = 1e-8)
    # npt.assert_allclose(gridding_dict['uniform_filter'], h5_after['uniform_filter'], atol = 1e-8)

    if "model_return" in gridding_dict:
        npt.assert_allclose(gridding_dict['model_return'], h5_after['model_return'], atol = 1e-8)