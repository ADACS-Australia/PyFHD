import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from glob import glob
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, get_data_items, sav_file_vis_arr_swap_axes
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from numpy.testing import assert_allclose
import deepdish as dd

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'vis_calibrate_subroutine')

@pytest.fixture
def full_data_dir():
    return glob('../**/full_size_vis_calibrate_subroutine/', recursive = True)[0]

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run2', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"]]

@pytest.fixture()
def before_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = {}
    h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
    h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])
    h5_save_dict['vis_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_ptr'])
    h5_save_dict['vis_model_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_ptr'])
    h5_save_dict['vis_weight_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weight_ptr'])

    dd.io.save(before_file, h5_save_dict)

    return before_file

@pytest.fixture()
def after_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    
    sav_file = after_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = {}
    h5_save_dict['cal_return'] = recarray_to_dict(sav_dict['cal_return'])

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_points_around_zenith_and_1088716296(before_file, after_file):
    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}")

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    vis_ptr = h5_before['vis_ptr']
    vis_model_ptr = h5_before['vis_model_ptr']
    vis_weight_ptr = h5_before['vis_weight_ptr']
    obs = h5_before['obs']
    cal = h5_before['cal']

    expected_cal = h5_after['cal_return']

    cal_return = vis_calibrate_subroutine(
        vis_ptr, 
        vis_model_ptr, 
        vis_weight_ptr, 
        obs, 
        cal,
        calibration_weights=True
    )

    assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal'] 
    assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1e-05)

@pytest.fixture(scope="function", params=[1, 2, 3])
def subroutine_test(request):
    return request.param

@pytest.fixture
def subroutine_before(data_dir, subroutine_test):
    subroutine_before = Path(data_dir, f"test_{subroutine_test}_before_{data_dir.name}.h5")

    if subroutine_before.exists():
        return subroutine_before

    vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal = get_data_items(
        data_dir,
        f"input_vis_ptr_{subroutine_test}.npy",
        f"input_vis_model_ptr_{subroutine_test}.npy",
        f"input_vis_weight_ptr_{subroutine_test}.npy",
        f"input_obs_{subroutine_test}.npy",
        f"input_cal_{subroutine_test}.npy",
    )

    h5_save_dict = {}
    h5_save_dict["vis_ptr"] = sav_file_vis_arr_swap_axes(vis_ptr)
    h5_save_dict["vis_model_ptr"] = sav_file_vis_arr_swap_axes(vis_model_ptr)
    h5_save_dict["vis_weight_ptr"] = sav_file_vis_arr_swap_axes(vis_weight_ptr)
    h5_save_dict["obs"] = recarray_to_dict(obs)
    h5_save_dict["cal"] = recarray_to_dict(cal)
    h5_save_dict['calibration_weights'] = 0
    if (subroutine_test == 2):
        h5_save_dict['calibration_weights'] = 1
   
    dd.io.save(subroutine_before, h5_save_dict)

    return subroutine_before

@pytest.fixture
def subroutine_after(data_dir, subroutine_test):
    subroutine_after = Path(data_dir, f"test_{subroutine_test}_after_{data_dir.name}.h5")

    if subroutine_after.exists():
        return subroutine_after
    
    h5_save_dict = {}
    h5_save_dict['cal_return'] = recarray_to_dict(get_data_items(data_dir, f"output_cal_return_{subroutine_test}.npy"))

    dd.io.save(subroutine_after, h5_save_dict)

    return subroutine_after

def test_vis_calibration_x(subroutine_before, subroutine_after):

    h5_before = dd.io.load(subroutine_before)
    h5_after = dd.io.load(subroutine_after)

    vis_ptr = h5_before['vis_ptr']
    vis_model_ptr = h5_before['vis_model_ptr']
    vis_weight_ptr = h5_before['vis_weight_ptr']
    obs = h5_before['obs']
    cal = h5_before['cal']
    calibration_weights = h5_before['calibration_weights']

    expected_cal = h5_after['cal_return']

    cal_return = vis_calibrate_subroutine(
        vis_ptr, 
        vis_model_ptr, 
        vis_weight_ptr, 
        obs, 
        cal,
        calibration_weights = calibration_weights
    )
    
    cal_return = recarray_to_dict(cal_return)
    assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal'] 
    expected_cal['gain'] = np.vstack(expected_cal['gain']).astype(np.complex128)
    cal_return['gain'] = np.vstack(cal_return['gain']).astype(np.complex128)
    assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1.5e-05)

# def test_vis_calibration_two(data_dir):
#     vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, expected_cal = get_data_items(
#         data_dir,
#         'input_vis_ptr_2.npy',
#         'input_vis_model_ptr_2.npy',
#         'input_vis_weight_ptr_2.npy',
#         'input_obs_2.npy',
#         'input_cal_2.npy',
#         'output_cal_return_2.npy'
#     )
#     cal_return = vis_calibrate_subroutine(
#         vis_ptr, 
#         vis_model_ptr, 
#         vis_weight_ptr, 
#         obs, 
#         cal
#     )
#     expected_cal = recarray_to_dict(expected_cal)
#     cal_return = recarray_to_dict(cal_return)
#     # assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal']
#     expected_cal['gain'] = np.vstack(expected_cal['gain']).astype(np.complex128)
#     cal_return['gain'] = np.vstack(cal_return['gain']).astype(np.complex128)
#     assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1e-05)

# def test_vis_calibration_three(data_dir):
#     vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, expected_cal = get_data_items(
#         data_dir,
#         'input_vis_ptr_3.npy',
#         'input_vis_model_ptr_3.npy',
#         'input_vis_weight_ptr_3.npy',
#         'input_obs_3.npy',
#         'input_cal_3.npy',
#         'output_cal_return_3.npy'
#     )
#     cal_return = vis_calibrate_subroutine(
#         vis_ptr, 
#         vis_model_ptr, 
#         vis_weight_ptr, 
#         obs, 
#         cal
#     )
#     expected_cal = recarray_to_dict(expected_cal)
#     cal_return = recarray_to_dict(cal_return)
#     assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal']
#     expected_cal['gain'] = np.vstack(expected_cal['gain']).astype(np.complex128)
#     cal_return['gain'] = np.vstack(cal_return['gain']).astype(np.complex128)
#     assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1e-05)