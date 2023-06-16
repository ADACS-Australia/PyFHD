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
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_bandpass")

def run_test(data_dir, tag_name):
    """Runs the test on `vis_cal_bandpass` - reads in the data in `data_loc`,
    and then calls `vis_cal_bandpass`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_vis_cal_bandpass.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_vis_cal_bandpass.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    params = h5_before['params']
    pyfhd_config = h5_before['pyfhd_config']

    exptected_cal_bandpass = h5_after['cal_bandpass']
    exptected_cal_remainder = h5_after['cal_remainder']
    
    logger = RootLogger(1)
    
    result_cal_bandpass, result_cal_remainder = vis_cal_bandpass(obs, cal, params, pyfhd_config, logger)

    ##The FHD function does some dividing by zeros, so we end up with NaNs
    ##in both the expected and result data. To check we are replicating
    ##the NaNs correctly, check both results for NaNs and assert they are
    ##in the same place

    expec_nan_inds = np.where(np.isnan(exptected_cal_remainder['gain']) == True)
    result_nan_inds = np.where(np.isnan(exptected_cal_remainder['gain']) == True)

    npt.assert_array_equal(expec_nan_inds, result_nan_inds)

    ##find where things are not NaN and check they are close
    test_inds = np.where(np.isnan(exptected_cal_remainder['gain']) == False)

    rtol = 5e-5
    atol = 1e-8

    npt.assert_allclose(exptected_cal_remainder['gain'][test_inds],
                        result_cal_remainder['gain'][test_inds],
                        rtol=rtol, atol=atol)

    ##shouoldn't be NaNs in this, so just check all the outputs
    npt.assert_allclose(exptected_cal_bandpass['gain'],
                       result_cal_bandpass['gain'], rtol=rtol, atol=atol)


def test_pointsource1_vary(data_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(data_dir, "pointsource1_vary1")

def test_pointsource2_vary(data_dir):
    """Test using the `pointsource1_vary2` set of inputs"""

    run_test(data_dir, "pointsource1_vary2")

    
if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_cal_bandpass`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_vis_cal_bandpass.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        cal = recarray_to_dict(sav_dict['cal'])
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal['gain'])
        cal['gain'] = gain
        
        params = recarray_to_dict(sav_dict['params'])
        
        ##make a slimmed down version of pyfhd_config
        pyfhd_config = {}
        
        ##If keywords cal_bp_transfer, cable_bandpass_fit are not set, IDL won't save it,
        ##so catch non-existent key and set to zero if this is the case
        ##Also, pyfhd uses None instead of 0 for False, so swap to none if
        ##needed
        try:
            pyfhd_config['cal_bp_transfer'] = sav_dict['cal_bp_transfer']
            if pyfhd_config['cal_bp_transfer'] == 0:
                pyfhd_config['cal_bp_transfer'] = None
        except KeyError:
            pyfhd_config['cal_bp_transfer'] = None
            
        try:
            pyfhd_config['cable_bandpass_fit'] = sav_dict['cable_bandpass_fit']
        except KeyError:
            pyfhd_config['cable_bandpass_fit'] = None

        try:
            pyfhd_config['auto_ratio_calibration'] = sav_dict['auto_ratio_calibration']
        except KeyError:
            pyfhd_config['auto_ratio_calibration'] = None


        ##need the instrument as that's needed for a file path
        pyfhd_config['instrument'] = 'mwa'
            
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['params'] = params
        h5_save_dict['pyfhd_config'] = pyfhd_config
        
        dd.io.save(Path(data_dir, f"{tag_name}_before_vis_cal_bandpass.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_cal_bandpass`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_vis_cal_bandpass.sav", "meh")
        
        cal_bandpass = recarray_to_dict(sav_dict['cal_bandpass'])
        cal_remainder = recarray_to_dict(sav_dict['cal_remainder'])
        
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
        
        dd.io.save(Path(data_dir, f"{tag_name}_after_vis_cal_bandpass.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_cal_bandpass`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_cal_bandpass')

     ##TODO get the tag_names from some kind of glob on the relevant dir
    tag_names = ['pointsource1_vary1', 'pointsource1_vary2']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)