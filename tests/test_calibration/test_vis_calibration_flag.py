import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_calibration_flag, vis_calibration_flag
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd

from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_calibration_flag")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_calibration_flag` - reads in the data in `data_loc`,
    and then calls `vis_calibration_flag`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_calibration_flag.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_calibration_flag.h5"))

    obs_in = h5_before['obs']
    cal = h5_before['cal']
    pyfhd_config = h5_before['pyfhd_config']
    
    expected_obs = h5_after['obs']
    
    ##TODO doesn't seem to be used in function, but is an argument
    params = {}
    logger = RootLogger(1)
    
    result_obs = vis_calibration_flag(obs_in, cal, params, pyfhd_config, logger)
    
    # print(result_res_mean, expected_res_mean)

    # assert np.allclose(result_res_mean, expected_res_mean, atol=1e-4)

    # ##Check that the gain value is inserted in `gain_list` correctly
    # assert np.allclose(gain_list[iter], expected_gain, atol=1e-8)

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_calibration_flag`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_calibration_flag.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'][0])
        cal = recarray_to_dict(sav_dict['cal'][0])
        
        ##Swap the freq and baseline dimensions
        gain = sav_file_vis_arr_swap_axes(cal['gain'])
        cal['gain'] = gain
        
        ##Make a small pyfhd_config with just the variables needed for this func
        pyfhd_config = {}
        pyfhd_config['amp_degree'] = sav_dict['amp_degree']
        pyfhd_config['phase_degree'] = sav_dict['phase_degree']
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['pyfhd_config'] = pyfhd_config
        
        dd.io.save(Path(test_dir, "before_vis_calibration_flag.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_calibration_flag`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_calibration_flag.sav", "meh")
        
        obs = recarray_to_dict(sav_dict['obs'][0])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        
        dd.io.save(Path(test_dir, "after_vis_calibration_flag.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_calibration_flag`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))