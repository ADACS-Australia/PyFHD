import pytest
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from PyFHD.gridding.filters import filter_uv_uniform
from PyFHD.pyfhd_tools.test_utils import get_data_items
import deepdish as dd

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'filter_uv_uniform')

@pytest.fixture(scope="function", params=[2,3])
def number(request):
    return request.param

def get_file(data_dir, file_name):
    if Path(data_dir, file_name).exists():
        item = get_data_items(data_dir, file_name)
        return item
    else:
        return None

@pytest.fixture
def filter_uni_before(data_dir, number):
    filter_uni_before = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if filter_uni_before.exists():
        return filter_uni_before

    image_uv, vis_count = get_data_items(
        data_dir,
        f'input_image_uv_{number}.npy',
        f'input_vis_count_{number}.npy',
    )

    weights = get_file(data_dir, f'input_weights_{number}.npy')

    h5_save_dict = {}
    h5_save_dict["image_uv"] = image_uv
    h5_save_dict["vis_count"] = vis_count
    h5_save_dict["weights"] = weights

    dd.io.save(filter_uni_before, h5_save_dict)
    
    return filter_uni_before

@pytest.fixture
def filter_uni_after(data_dir, number):
    filter_uni_after = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if filter_uni_after.exists():
        return filter_uni_after

    expected_image_uv, expected_filter = get_data_items(
        data_dir,
        f'output_image_uv_filtered_{number}.npy',
        f'output_filter_{number}.npy'
    )

    h5_save_dict = {}
    h5_save_dict["image_uv"] = expected_image_uv
    h5_save_dict["filter"] = expected_filter

    dd.io.save(filter_uni_after, h5_save_dict)
    
    return filter_uni_after

def test_filter_uv_uniform(filter_uni_before: Path, filter_uni_after: Path):
    h5_before = dd.io.load(filter_uni_before)
    h5_after = dd.io.load(filter_uni_after)

    image_uv_filtered, filter = filter_uv_uniform(
        h5_before["image_uv"], 
        h5_before["vis_count"],
        weights = h5_before["weights"]
    )

    npt.assert_allclose(image_uv_filtered, h5_after["image_uv"], atol=2e-5)
    npt.assert_allclose(filter, h5_after["filter"], atol=1e-8)

# def test_filter_uni_two(data_dir):
#     image_uv, vis_count, expected_image_uv, expected_filter = get_data_items(
#         data_dir,
#         'input_image_uv_2.npy',
#         'restored_vis_count_2.npy',
#         'output_image_uv_filtered_2.npy',
#         'output_filter_2.npy'
#     )
#     image_uv_filtered, filter = filter_uv_uniform(image_uv, vis_count)
#     # Check the result precision error is beyond single precision
#     assert  1e-8 > np.max(np.abs(expected_image_uv.real - image_uv_filtered.real))
#     assert  1e-8 > np.max(np.abs(expected_image_uv.imag - image_uv_filtered.imag))
#     # Check the result precision error is beyond single precision
#     assert 1e-8 > np.max(np.abs(filter - expected_filter))

# def test_filter_uni_three(data_dir):
#     image_uv, vis_count, weights, expected_image_uv, expected_filter = get_data_items(
#         data_dir,
#         'input_image_uv_3.npy',
#         'restored_vis_count_3.npy',
#         'input_weights_3.npy',
#         'output_image_uv_filtered_3.npy',
#         'output_filter_3.npy'
#     )
#     image_uv_filtered, filter = filter_uv_uniform(image_uv, vis_count = vis_count, weights = weights)
#     # Check the result precision error is beyond a threshold (single precision rounding errors have occurred)
#     assert  1e-8 > np.max(np.abs(expected_image_uv.real - image_uv_filtered.real))
#     assert  1e-8 > np.max(np.abs(expected_image_uv.imag - image_uv_filtered.imag))
#     # Check the result precision error is beyond a threshold (single precision rounding errors have occurred)
#     assert 1e-8 > np.max(np.abs(filter - expected_filter))