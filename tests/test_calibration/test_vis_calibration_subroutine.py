import pytest
import numpy as np
from pathlib import Path
from fhd_core.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from tests.test_utils import get_data_items

@pytest.fixture
def data_dir():
    return list(Path.glob(Path.cwd(), '**/vis_calibrate_subroutine/'))[0]

@pytest.fixture
def full_data_dir():
    return list(Path.glob(Path.cwd(), '**/full_size_vis_calibrate_subroutine/'))[0]

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