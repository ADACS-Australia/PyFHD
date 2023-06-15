import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from PyFHD.pyfhd_tools.test_utils import get_data_items
from glob import glob

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
    assert expected_cal['n_vis_cal'] == cal_return['n_vis_cal']