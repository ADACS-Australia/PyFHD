import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.gridding.gridding_utils import dirty_image_generate
from PyFHD.pyfhd_tools.test_utils import get_data, get_data_items
from numpy.testing import assert_allclose
import deepdish as dd
from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'dirty_image_generate')

@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6, 7, 8, 9])
def number(request):
    return request.param

def get_file(data_dir, file_name):
    if Path(data_dir, file_name).exists():
        item = get_data_items(data_dir, file_name)
        return item
    else:
        return None

@pytest.fixture
def dirty_before(data_dir, number):
    dirty_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if dirty_before.exists():
        return dirty_before
    
    pyfhd_config = {
        "image_filter": "filter_uv_uniform"
    }
    baseline_threshold = get_file(data_dir, f'input_baseline_threshold_{number}.npy')
    h5_save_dict = {
        "dirty_image_uv": get_file(data_dir, f'input_dirty_image_uv_{number}.npy'),
        "pyfhd_config": pyfhd_config,
        "mask": get_file(data_dir, f'input_mask_{number}.npy'),
        "baseline_threshold": baseline_threshold if baseline_threshold else 0,
        "normalization": get_file(data_dir, f'output_normalization_{number}.npy'),
        "resize": get_file(data_dir, f'input_resize_{number}.npy'),
        "width_smooth": get_file(data_dir, f'input_width_smooth_{number}.npy'),
        "degpix": get_file(data_dir, f'input_degpix_{number}.npy'),
        "not_real": True if get_file(data_dir, f'input_no_real_{number}.npy') else False,
        "pad_uv_image": get_file(data_dir, f'input_pad_uv_image_{number}.npy'),
        "weights": get_file(data_dir, f'input_weights_{number}.npy'),
        "filter": get_file(data_dir, f'input_filter_{number}.npy'),
        "beam_ptr": get_file(data_dir, f'input_beam_ptr_{number}.npy'),
    }

    dd.io.save(dirty_before, h5_save_dict)
    
    return dirty_before

@pytest.fixture
def dirty_after(data_dir, number):
    dirty_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if dirty_after.exists():
        return dirty_after

    dirty_image = get_data_items(data_dir, f'output_dirty_image_{number}.npy')
    normalization = get_file(data_dir, f'output_normalization_{number}.npy')
    h5_save_dict = {
        "dirty_image": dirty_image,
        "normalization": normalization
    }

    dd.io.save(dirty_after, h5_save_dict)
    
    return dirty_after

def test_dirty_image_generate(dirty_before: Path, dirty_after: Path):
    h5_before = dd.io.load(dirty_before)
    h5_after = dd.io.load(dirty_after)

    if h5_before["normalization"] == None:
        dirty_image = dirty_image_generate(
            h5_before["dirty_image_uv"],
            h5_before["pyfhd_config"],
            RootLogger(1),
            mask = h5_before["mask"],
            baseline_threshold = h5_before["baseline_threshold"],
            normalization = h5_before["normalization"],
            resize = int(h5_before["resize"]) if h5_before["resize"] else None,
            width_smooth = h5_before["width_smooth"],
            degpix = h5_before["degpix"],
            not_real = h5_before["not_real"],
            pad_uv_image = h5_before["pad_uv_image"],
            weights = h5_before["weights"],
            filter = h5_before["filter"],
            beam_ptr = h5_before["beam_ptr"],
        )
    else:
        dirty_image, normalization = dirty_image_generate(
            h5_before["dirty_image_uv"],
            h5_before["pyfhd_config"],
            RootLogger(1),
            mask = h5_before["mask"],
            baseline_threshold = h5_before["baseline_threshold"],
            normalization = h5_before["normalization"],
            resize = int(h5_before["resize"]) if h5_before["resize"] else None,
            width_smooth = h5_before["width_smooth"],
            degpix = h5_before["degpix"],
            not_real = h5_before["not_real"],
            pad_uv_image = h5_before["pad_uv_image"],
            weights = h5_before["weights"],
            filter = h5_before["filter"],
            beam_ptr = h5_before["beam_ptr"],
        )
    
    assert_allclose(dirty_image, h5_after["dirty_image"], atol=1e-8)
    if h5_after["normalization"] != None:
        assert_allclose(normalization, h5_after["normalization"], atol=1e-8)

# def test_dirty_one(data_dir):
#     dirty_image_uv, no_real, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_1.npy',
#         'input_no_real_1.npy',
#         'output_dirty_image_1.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-16)

# def test_dirty_two(data_dir):
#     dirty_image_uv, no_real, degpix, expected_dirty_image, expected_normalization = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_2.npy',
#         'input_no_real_2.npy',
#         'input_degpix_2.npy',
#         'output_dirty_image_2.npy',
#         'output_normalization_2.npy'
#     )
#     dirty_image, normalization = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         normalization = expected_normalization
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-14)
#     assert np.array_equal(expected_normalization, normalization)

# def test_dirty_three(data_dir):
#     dirty_image_uv, no_real, degpix, baseline_threshold, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_3.npy',
#         'input_no_real_3.npy',
#         'input_degpix_3.npy',
#         'input_baseline_threshold_3.npy',
#         'output_dirty_image_3.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         baseline_threshold = baseline_threshold
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-10)

# def test_dirty_four(data_dir):
#     dirty_image_uv, no_real, degpix, baseline_threshold, width_smooth, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_4.npy',
#         'input_no_real_4.npy',
#         'input_degpix_4.npy',
#         'input_baseline_threshold_4.npy',
#         'input_width_smooth_4.npy',
#         'output_dirty_image_4.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         baseline_threshold = baseline_threshold,
#         width_smooth = width_smooth
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-10)

# def test_dirty_five(data_dir):
#     dirty_image_uv, no_real, degpix, mask, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_5.npy',
#         'input_no_real_5.npy',
#         'input_degpix_5.npy',
#         'input_mask_5.npy',
#         'output_dirty_image_5.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         mask = mask
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-14)

# def test_dirty_six(data_dir):
#     dirty_image_uv, no_real, degpix, resize, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_6.npy',
#         'input_no_real_6.npy',
#         'input_degpix_6.npy',
#         'input_resize_6.npy',
#         'output_dirty_image_6.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         resize = int(resize)
#     )
#     # REBIN in IDL is single precision so see if it matches single precision
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-8)

# def test_dirty_seven(data_dir):
#     dirty_image_uv, no_real, degpix, pad_uv_image, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_7.npy',
#         'input_no_real_7.npy',
#         'input_degpix_7.npy',
#         'input_pad_uv_image_7.npy',
#         'output_dirty_image_7.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         pad_uv_image = pad_uv_image
#     )
#     # Set to single precision threshold as the division by radians will be single precision.
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-8)

# def test_dirty_eight(data_dir):
#     dirty_image_uv, no_real, degpix, filter, expected_dirty_image = get_data_items(
#         data_dir,
#         'input_dirty_image_uv_8.npy',
#         'input_no_real_8.npy',
#         'input_degpix_8.npy',
#         'input_filter_8.npy',
#         'output_dirty_image_8.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv, 
#         not_real = no_real, 
#         degpix = degpix, 
#         filter = filter
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-16)

# def test_dirty_nine(data_dir):
#     dirty_image_uv, filter, image_filter_fn, expected_dirty_image = get_data_items(
#         data_dir,
#         'newbranch_input_dirty_image_uv_9.npy',
#         'newbranch_input_filter_9.npy',
#         'newbranch_input_image_filter_fn_9.npy',
#         'newbranch_output_dirty_image_9.npy'
#     )
#     psf = get_data(
#         data_dir,
#         'newbranch_input_psf_9.npy'
#     )
#     dirty_image = dirty_image_generate(
#         dirty_image_uv,  
#         filter = filter,
#         image_filter_fn = image_filter_fn.decode("utf-8"),
#     )
#     assert_allclose(dirty_image, expected_dirty_image, atol=1e-16)
