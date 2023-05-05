import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_polyfit
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
import importlib_resources

from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_polyfit")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_cal_polyfit` - reads in the data in `data_loc`,
    and then calls `vis_cal_polyfit`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_cal_polyfit.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_cal_polyfit.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    pyfhd_config = h5_before['pyfhd_config']
    
    ##TODO for this to work, we have to move mwa_cable_reflection_coefficients.txt
    ##into PyFHD.templates as it needs to be in a PyFHD module
    pyfhd_config["cable_reflection_coefficients"] = importlib_resources.files('PyFHD.templates').joinpath('mwa_cable_reflection_coefficients.txt')
    
    exptected_cal_return = h5_after['cal_return']
    
    logger = RootLogger(1)

    print(obs['n_tile'])
    
    vis_cal_polyfit(obs, cal, pyfhd_config, logger)
    
    ##TODO actually test something!

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_cal_polyfit`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_polyfit.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'][0])
        cal = recarray_to_dict(sav_dict['cal'][0])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal['gain'])
        cal['gain'] = gain

        ##TODO these probably need reshaping
        # cal['amp_params']
        # cal['phase_params']

        fhd_keys = ["cal_reflection_mode_theory",
                    "cal_reflection_mode_file",
                    "cal_reflection_mode_delay",
                    "cal_reflection_hyperresolve",
                    "amp_degree",
                    "phase_degree"]
        config_keys = ["cal_reflection_mode_theory",
                      "cal_reflection_mode_file",
                      "cal_reflection_mode_delay",
                      "cal_reflection_hyperresolve",
                      "cal_amp_degree_fit",
                      "cal_phase_degree_fit"]
        
        ##make a slimmed down version of pyfhd_config
        pyfhd_config = {}
        
        ##When keys are unset in IDL, they just don't save to a .sav
        ##file. So try accessing with an exception and set to None
        ##if they don't exists
        for fhd_key, config_key in zip(fhd_keys, config_keys):
            try:
                pyfhd_config[config_key] = sav_dict[fhd_key]
            except KeyError:
                pyfhd_config[config_key] = None
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['pyfhd_config'] = pyfhd_config
        
        dd.io.save(Path(test_dir, "before_vis_cal_polyfit.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_cal_polyfit`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_polyfit.sav", "meh")
        
        cal_return = recarray_to_dict(sav_dict['cal_return'][0])
        
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal_return['gain'])
        cal_return['gain'] = gain
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_return'] = cal_return
        dd.io.save(Path(test_dir, "after_vis_cal_polyfit.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_cal_polyfit`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))