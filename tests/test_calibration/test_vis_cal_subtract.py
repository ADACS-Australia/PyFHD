import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_subtract
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import importlib_resources

from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_subtract")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_cal_subtract` - reads in the data in `data_loc`,
    and then calls `vis_cal_subtract`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_cal_subtract.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_cal_subtract.h5"))

    cal_base = h5_before['cal_base']
    cal = h5_before['cal']
    absolute_value = h5_before['absolute_value']
    
    exptected_cal_residual = h5_after['cal_residual']
    
    ##TODO make this work depending on how the function is translated
    # return_cal_residual = vis_cal_subtract(cal_base, cal, absolute_cal)
    # assert np.allclose(return_cal_residual['gain'], exptected_cal_residual['gain'], 1e-5)

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_cal_subtract`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_subtract.sav", "meh")

        cal_base = recarray_to_dict(sav_dict['cal_base'][0])
        cal = recarray_to_dict(sav_dict['cal'][0])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal_base['gain'] = sav_file_vis_arr_swap_axes(cal_base['gain'])
        cal['gain'] = sav_file_vis_arr_swap_axes(cal['gain'])

        ##When keys are unset in IDL, they just don't save to a .sav
        ##file. So try accessing with an exception and set to None
        ##if they don't exists
        try:
            absolute_value = sav_dict['absolute_value']
        except KeyError:
            absolute_value = None

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['cal_base'] = cal_base
        h5_save_dict['cal'] = cal
        h5_save_dict['absolute_value'] = absolute_value
        
        dd.io.save(Path(test_dir, "before_vis_cal_subtract.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_cal_subtract`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_subtract.sav", "meh")
        
        cal_residual = recarray_to_dict(sav_dict['cal_residual'][0])
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal_residual['gain'] = sav_file_vis_arr_swap_axes(cal_residual['gain'])
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_residual'] = cal_residual
        dd.io.save(Path(test_dir, "after_vis_cal_subtract.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_cal_subtract`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))