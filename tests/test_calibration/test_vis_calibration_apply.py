import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_calibration_apply
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_calibration_apply")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_calibration_apply` - reads in the data in `data_loc`,
    and then calls `vis_calibration_apply`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_calibration_apply.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_calibration_apply.h5"))

    vis_ptr = h5_before['vis_ptr']
    cal = h5_before['cal']
    preserve_original = h5_before['preserve_original']
    invert_gain = h5_before['invert_gain']
    vis_model_ptr = h5_before['vis_model_ptr']
    vis_weight_ptr = h5_before['vis_weight_ptr']

    exptected_vis_cal_ptr = h5_after['vis_cal_ptr']
    
    ##TODO make this work depending on how the function is translated
    # return_vis_cal_ptr = vis_calibration_apply(vis_ptr, cal, preserve_original, invert_gain,
    #                                            vis_model_ptr, vis_weight_ptr)
    # assert np.allclose(return_vis_cal_ptr, exptected_vis_cal_ptr, 1e-5)

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_calibration_apply`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_calibration_apply.sav", "meh")

        # obs = recarray_to_dict(sav_dict['obs'][0])
        cal = recarray_to_dict(sav_dict['cal'][0])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal['gain'] = sav_file_vis_arr_swap_axes(cal['gain'])

        # vis_ptr,cal,preserve_original,invert_gain,vis_model_ptr,vis_weight_ptr

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['cal'] = cal
        h5_save_dict['vis_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_ptr'])

        ##TODO these probably need reshaping
        # cal['amp_params']
        # cal['phase_params']

        ##When keys are unset in IDL, they just don't save to a .sav
        ##file. So try accessing with an exception and set to None
        ##if they don't exists
        for key in ['vis_model_ptr', 'vis_weight_ptr']:
            iskey = False
            try:
                iskey = sav_dict[key]
            except KeyError:
                h5_save_dict[key] = None

            ##You can get a IDL pointer to two empty arrays here, so check if
            ##anything exists inside the array as well as it being an array
            if type(iskey) == np.ndarray and type(iskey[0]) == np.ndarray:
                h5_save_dict[key] = sav_file_vis_arr_swap_axes(sav_dict[key])
            else:
                h5_save_dict[key] = None

        for key in ['invert_gain', 'preserve_original']:
            try:
                h5_save_dict[key] = sav_dict[key]
                print("yeah boi?", key, h5_save_dict[key])
            except KeyError:
                h5_save_dict[key]  = None

        dd.io.save(Path(test_dir, "before_vis_calibration_apply.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_calibration_apply`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_calibration_apply.sav", "meh")
        
        h5_save_dict = {}

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        h5_save_dict['vis_cal_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_cal_ptr'])
        
        dd.io.save(Path(test_dir, "after_vis_calibration_apply.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_calibration_apply`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))