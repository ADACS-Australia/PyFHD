import pytest
import numpy as np
from pathlib import Path
from fhd_core.gridding.dirty_image_generate import dirty_image_generate
from tests.test_utils import get_data, get_data_items

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), '**/dirty_image_generate/'))[0]

def test_dirty_one(data_dir):
    dirty_image_uv, no_real, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_1.npy',
        'input_no_real_1.npy',
        'output_dirty_image_1.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real
    )
    assert 1e-16 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-16 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_two(data_dir):
    dirty_image_uv, no_real, degpix, expected_dirty_image, expected_normalization = get_data_items(
        data_dir,
        'input_dirty_image_uv_2.npy',
        'input_no_real_2.npy',
        'input_degpix_2.npy',
        'output_dirty_image_2.npy',
        'output_normalization_2.npy'
    )
    dirty_image, normalization = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        normalization = expected_normalization
    )
    assert 1e-14 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-14 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))
    assert np.array_equal(expected_normalization, normalization)

def test_dirty_three(data_dir):
    dirty_image_uv, no_real, degpix, baseline_threshold, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_3.npy',
        'input_no_real_3.npy',
        'input_degpix_3.npy',
        'input_baseline_threshold_3.npy',
        'output_dirty_image_3.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        baseline_threshold = baseline_threshold
    )
    assert 1e-10 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-14 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_four(data_dir):
    dirty_image_uv, no_real, degpix, baseline_threshold, width_smooth, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_4.npy',
        'input_no_real_4.npy',
        'input_degpix_4.npy',
        'input_baseline_threshold_4.npy',
        'input_width_smooth_4.npy',
        'output_dirty_image_4.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        baseline_threshold = baseline_threshold,
        width_smooth = width_smooth
    )
    assert 1e-10 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-14 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_five(data_dir):
    dirty_image_uv, no_real, degpix, mask, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_5.npy',
        'input_no_real_5.npy',
        'input_degpix_5.npy',
        'input_mask_5.npy',
        'output_dirty_image_5.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        mask = mask
    )
    assert 1e-14 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-14 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_six(data_dir):
    dirty_image_uv, no_real, degpix, resize, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_6.npy',
        'input_no_real_6.npy',
        'input_degpix_6.npy',
        'input_resize_6.npy',
        'output_dirty_image_6.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        resize = int(resize)
    )
    # REBIN in IDL is single precision so see if it matches single precision
    assert 1e-8 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-8 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_seven(data_dir):
    dirty_image_uv, no_real, degpix, pad_uv_image, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_7.npy',
        'input_no_real_7.npy',
        'input_degpix_7.npy',
        'input_pad_uv_image_7.npy',
        'output_dirty_image_7.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        pad_uv_image = pad_uv_image
    )
    # Set to single precision threshold as the division by radians will be single precision.
    assert 1e-8 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-8 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_eight(data_dir):
    dirty_image_uv, no_real, degpix, filter, expected_dirty_image = get_data_items(
        data_dir,
        'input_dirty_image_uv_8.npy',
        'input_no_real_8.npy',
        'input_degpix_8.npy',
        'input_filter_8.npy',
        'output_dirty_image_8.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv, 
        not_real = no_real, 
        degpix = degpix, 
        filter = filter
    )
    assert 1e-16 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-16 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))

def test_dirty_nine(data_dir):
    dirty_image_uv, filter, image_filter_fn, obs, params, weights, expected_dirty_image = get_data_items(
        data_dir,
        'newbranch_input_dirty_image_uv_9.npy',
        'newbranch_input_filter_9.npy',
        'newbranch_input_image_filter_fn_9.npy',
        'newbranch_input_obs_9.npy',
        'newbranch_input_params_9.npy',
        'newbranch_input_weights_9.npy',
        'newbranch_output_dirty_image_9.npy'
    )
    psf = get_data(
        data_dir,
        'newbranch_input_psf_9.npy'
    )
    dirty_image = dirty_image_generate(
        dirty_image_uv,  
        filter = filter,
        image_filter_fn = image_filter_fn.decode("utf-8"),
        obs = obs,
        psf = psf,
        params = params,
        weights = weights
    )
    assert 1e-16 > np.max(np.abs(expected_dirty_image.real - dirty_image.real))
    assert 1e-16 > np.max(np.abs(expected_dirty_image.imag - dirty_image.imag))
