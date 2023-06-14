import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_auto_fit
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_fit")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def test_pointsource1_vary(data_dir):
    """Runs the test on `vis_cal_auto_fit` - reads in the data in `data_loc`,
    and then calls `vis_cal_auto_fit`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, "before_vis_cal_auto_fit.h5"))
    h5_after = dd.io.load(Path(data_dir, "after_vis_cal_auto_fit.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    vis_auto = h5_before['vis_auto']
    vis_model_auto = h5_before['vis_model_auto']
    auto_tile_i = h5_before['auto_tile_i']

    expected_cal_fit = h5_after['cal_fit']

    return_cal_fit = vis_cal_auto_fit(obs, cal, vis_auto, vis_model_auto, auto_tile_i)
    # auto_scale is nan, nan from FHD and Python, can't compare them due to the nans
    # assert np.array_equal(return_cal_fit['auto_scale'], expected_cal_fit['auto_scale'])
    # cal_fit['auto_params'] came in as an object array
    auto_params = np.empty([2, cal["n_pol"], cal['n_tile']], dtype = np.float64)
    auto_params[0] = expected_cal_fit['auto_params'][0].transpose()
    auto_params[1] = expected_cal_fit['auto_params'][1].transpose()
    assert np.allclose(return_cal_fit['auto_params'], auto_params)
    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_cal_auto_fit`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_auto_fit.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'][0])
        cal = recarray_to_dict(sav_dict['cal'][0])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal['gain'] = sav_file_vis_arr_swap_axes(cal['gain'])
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['vis_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_auto'])
        h5_save_dict['vis_model_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_auto'])
        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']
        
        dd.io.save(Path(test_dir, "before_vis_cal_auto_fit.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_cal_auto_fit`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_auto_fit.sav", "meh")
        
        cal_fit = recarray_to_dict(sav_dict['cal_fit'][0])
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal_fit['gain'] = sav_file_vis_arr_swap_axes(cal_fit['gain'])
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_fit'] = cal_fit
        dd.io.save(Path(test_dir, "after_vis_cal_auto_fit.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_cal_auto_fit`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))