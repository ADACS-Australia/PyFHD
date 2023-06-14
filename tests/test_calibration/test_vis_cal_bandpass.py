import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_bandpass
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import importlib_resources

from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_bandpass")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_cal_bandpass` - reads in the data in `data_loc`,
    and then calls `vis_cal_bandpass`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_cal_bandpass.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_cal_bandpass.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    params = h5_before['params']
    pyfhd_config = h5_before['pyfhd_config']
    
    ##TODO if cable_bandpass_fit=True we need to setup:
    '''
    pyfhd_config["input"]
    pyfhd_config["cable-reflection-coefficients"]
    to make
    cable_len = np.loadtxt(Path(pyfhd_config["input"], pyfhd_config["cable-reflection-coefficients"]), skiprows=1)[:, 2].flatten()
    
    
    need cable len file to point to
    
    PyFHD/input/instrument_config/mwa_cable_reflection_coefficients.txt
    
    work'''
    
    ##TODO for this to work, we have to move mwa_cable_reflection_coefficients.txt
    ##into PyFHD.templates as it needs to be in a PyFHD module
    pyfhd_config["cable_reflection_coefficients"] = importlib_resources.files('PyFHD.templates').joinpath('mwa_cable_reflection_coefficients.txt')
    
    exptected_cal_bandpass = h5_after['cal_bandpass']
    exptected_cal_remainder = h5_after['cal_remainder']
    
    logger = RootLogger(1)
    
    result_cal_bandpass, result_cal_remainder = vis_cal_bandpass(obs, cal, params, pyfhd_config, logger)
    
    # print(result_res_mean, expected_res_mean)

    # assert np.allclose(result_res_mean, expected_res_mean, atol=1e-4)

    # ##Check that the gain value is inserted in `gain_list` correctly
    # assert np.allclose(gain_list[iter], expected_gain, atol=1e-8)

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_cal_bandpass`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_bandpass.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'][0])
        cal = recarray_to_dict(sav_dict['cal'][0])
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal['gain'])
        cal['gain'] = gain
        
        params = recarray_to_dict(sav_dict['params'][0])
        
        ##make a slimmed down version of pyfhd_config
        pyfhd_config = {}
        
        ##If keyword cal_bp_transfer is not set, IDL won't save it,
        ##so catch non-existent key and set to zero if this is the case
        try:
            pyfhd_config['cal_bp_transfer'] = sav_dict['cal_bp_transfer']
        except KeyError:
            pyfhd_config['cal_bp_transfer'] = None
            
        try:
            pyfhd_config['cable_bandpass_fit'] = sav_dict['cable_bandpass_fit']
        except KeyError:
            pyfhd_config['cable_bandpass_fit'] = None
            
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['params'] = params
        h5_save_dict['pyfhd_config'] = pyfhd_config
        
        dd.io.save(Path(test_dir, "before_vis_cal_bandpass.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_cal_bandpass`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_bandpass.sav", "meh")
        
        cal_bandpass = recarray_to_dict(sav_dict['cal_bandpass'][0])
        cal_remainder = recarray_to_dict(sav_dict['cal_remainder'][0])
        
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal_bandpass['gain'])
        cal_bandpass['gain'] = gain
        
        gain = sav_file_vis_arr_swap_axes(cal_remainder['gain'])
        cal_remainder['gain'] = gain

        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_bandpass'] = cal_bandpass
        h5_save_dict['cal_remainder'] = cal_remainder
        
        dd.io.save(Path(test_dir, "after_vis_cal_bandpass.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_cal_bandpass`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        run_test(Path(base_dir, test_set))