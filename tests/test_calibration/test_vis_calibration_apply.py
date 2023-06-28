import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_calibration_apply
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import numpy.testing as npt
from logging import RootLogger
# import matplotlib.pyplot as plt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_calibration_apply")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_dir, tag_name):
    """Runs the test on `vis_calibration_apply` - reads in the data in `data_loc`,
    and then calls `vis_calibration_apply`, checking the outputs match """
    
    func_name = "vis_calibration_apply"

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_{func_name}.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_{func_name}.h5"))

    vis_ptr = h5_before['vis_ptr']
    cal = h5_before['cal']

    ##The FHD code has made copies of things from `obs` into `cal`. In PyFHD,
    ##we just supply the `obs`. Means we need to make a mini `obs` for testing
    ##here
    obs = {}
    obs['baseline_info'] = {}
    obs['baseline_info']['tile_a'] = cal['tile_a']
    obs['baseline_info']['tile_b'] = cal['tile_b']
    obs['n_freq'] = cal['n_freq']
    obs['n_baselines'] = len(cal['tile_a'])
    obs['n_times'] = cal['n_time']

    preserve_original = h5_before['preserve_original']
    invert_gain = h5_before['invert_gain']
    vis_model_ptr = h5_before['vis_model_ptr']
    vis_weight_ptr = h5_before['vis_weight_ptr']

    print("Are we invert_gain ing?", invert_gain)

    exptected_vis_cal_ptr = h5_after['vis_cal_ptr']

    logger = RootLogger(1)

    return_vis_cal_ptr, return_cal = vis_calibration_apply(vis_ptr, obs, cal,
                                                           vis_model_ptr,
                                                           vis_weight_ptr,
                                                           logger)
    npt.assert_allclose(return_vis_cal_ptr, exptected_vis_cal_ptr, atol=1e-6,
                        equal_nan=True)
    
    ##TODO if we have 4 pols, we need to test against
    ##h5_after['vis_cal_ptr']['cross_phase]

def test_pointsource1_vary1(data_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(data_dir, "pointsource1_vary1")

# def test_pointsource2_vary1(data_dir):
#     """Test using the `pointsource2_vary1` set of inputs"""

#     run_test(data_dir, "pointsource2_vary1")

    
if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_calibration_apply`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_vis_calibration_apply.sav", "meh")

        cal = recarray_to_dict(sav_dict['cal'])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal['gain'] = sav_file_vis_arr_swap_axes(cal['gain'])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['cal'] = cal
        h5_save_dict['vis_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_ptr'])

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
                # print("yeah boi?", key, h5_save_dict[key])
            except KeyError:
                h5_save_dict[key]  = None

        dd.io.save(Path(data_dir, f"{tag_name}_before_vis_calibration_apply.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_calibration_apply`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_vis_calibration_apply.sav", "meh")
        
        h5_save_dict = {}

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        h5_save_dict['vis_cal_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_cal_ptr'])
        h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])
        
        # dd.io.save(Path(test_dir, "after_vis_calibration_apply.h5"), h5_save_dict)
        dd.io.save(Path(data_dir, f"{tag_name}_after_vis_calibration_apply.h5"), h5_save_dict)
        
    def convert_sav(data_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_cal_auto_fit`"""
        convert_before_sav(data_dir, tag_name)
        convert_after_sav(data_dir, tag_name)

    ##Where be all of our data
    data_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_calibration_apply')

    ##Each test_set contains a run with a different set of inputs/options
    # for test_set in ['pointsource1_vary1']:
    # tag_names = ['pointsource1_vary1', 'pointsource2_vary1']
    tag_names = ['pointsource1_vary2']
    # tag_names = ['pointsource1_vary1']

    for tag_name in tag_names:
        convert_sav(data_dir, tag_name)
        # run_test(data_dir, tag_name)