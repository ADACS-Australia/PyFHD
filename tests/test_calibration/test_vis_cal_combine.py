import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_combine
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import importlib_resources

from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_combine")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_cal_combine` - reads in the data in `data_loc`,
    and then calls `vis_cal_combine`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_cal_combine.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_cal_combine.h5"))

    cal1 = h5_before['cal1']
    cal2 = h5_before['cal2']
    
    expected_cal_out = h5_after['cal_out']
    
    logger = RootLogger(1)

    ##TODO make this work depending on how the function is translated
    # return_cal_out = vis_cal_combine(cal1, cal2, logger)
    # assert np.allclose(return_cal_out['gain'], expected_cal_out['gain'], 1e-5)

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_cal_combine`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_combine.sav", "meh")

        cal1 = recarray_to_dict(sav_dict['cal1'][0])
        cal2 = recarray_to_dict(sav_dict['cal2'][0])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal1['gain'] = sav_file_vis_arr_swap_axes(cal1['gain'])
        cal2['gain'] = sav_file_vis_arr_swap_axes(cal2['gain'])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['cal1'] = cal1
        h5_save_dict['cal2'] = cal2
        
        dd.io.save(Path(test_dir, "before_vis_cal_combine.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_cal_combine`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_combine.sav", "meh")
        
        cal_out = recarray_to_dict(sav_dict['cal_out'][0])
        
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal_out['gain'] = sav_file_vis_arr_swap_axes(cal_out['gain'])
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_out'] = cal_out
        dd.io.save(Path(test_dir, "after_vis_cal_combine.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_cal_combine`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))