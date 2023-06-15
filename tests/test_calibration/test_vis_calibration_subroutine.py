import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from glob import glob
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, get_data_items
from numpy.testing import assert_allclose

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'vis_calibrate_subroutine')

@pytest.fixture
def full_data_dir():
    return glob('../**/full_size_vis_calibrate_subroutine/', recursive = True)[0]

def test_vis_calibration_one(data_dir):
    vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, expected_cal = get_data_items(
        data_dir,
        'input_vis_ptr_1.npy',
        'input_vis_model_ptr_1.npy',
        'input_vis_weight_ptr_1.npy',
        'input_obs_1.npy',
        'input_cal_1.npy',
        'output_cal_return_1.npy'
    )
    cal_return = vis_calibrate_subroutine(
        vis_ptr, 
        vis_model_ptr, 
        vis_weight_ptr, 
        obs, 
        cal
    )
    expected_cal = recarray_to_dict(expected_cal)
    cal_return = recarray_to_dict(cal_return)
    assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal'] 
    expected_cal['gain'] = np.vstack(expected_cal['gain']).astype(np.complex128)
    cal_return['gain'] = np.vstack(cal_return['gain']).astype(np.complex128)
    assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1e-05)

def test_vis_calibration_two(data_dir):
    vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, expected_cal = get_data_items(
        data_dir,
        'input_vis_ptr_2.npy',
        'input_vis_model_ptr_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_obs_2.npy',
        'input_cal_2.npy',
        'output_cal_return_2.npy'
    )
    cal_return = vis_calibrate_subroutine(
        vis_ptr, 
        vis_model_ptr, 
        vis_weight_ptr, 
        obs, 
        cal
    )
    expected_cal = recarray_to_dict(expected_cal)
    cal_return = recarray_to_dict(cal_return)
    # assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal']
    expected_cal['gain'] = np.vstack(expected_cal['gain']).astype(np.complex128)
    cal_return['gain'] = np.vstack(cal_return['gain']).astype(np.complex128)
    assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1e-05)

def test_vis_calibration_three(data_dir):
    vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, expected_cal = get_data_items(
        data_dir,
        'input_vis_ptr_3.npy',
        'input_vis_model_ptr_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_obs_3.npy',
        'input_cal_3.npy',
        'output_cal_return_3.npy'
    )
    cal_return = vis_calibrate_subroutine(
        vis_ptr, 
        vis_model_ptr, 
        vis_weight_ptr, 
        obs, 
        cal
    )
    expected_cal = recarray_to_dict(expected_cal)
    cal_return = recarray_to_dict(cal_return)
    assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal']
    expected_cal['gain'] = np.vstack(expected_cal['gain']).astype(np.complex128)
    cal_return['gain'] = np.vstack(cal_return['gain']).astype(np.complex128)
    assert_allclose(cal_return['gain'], expected_cal['gain'], atol = 1e-05)