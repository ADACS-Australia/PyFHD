import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_auto_init
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes

import numpy as np
import deepdish as dd
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_init")

def run_test(data_dir, tag_name):
    """Runs the test on `vis_cal_auto_init` - reads in the data in `data_loc`,
    and then calls `vis_cal_auto_init`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_vis_cal_auto_init.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_vis_cal_auto_init.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    vis_arr = h5_before['vis_arr']
    vis_model_arr = h5_before['vis_model_arr']
    vis_auto = h5_before['vis_auto']
    vis_model_auto = h5_before['vis_model_auto']
    auto_tile_i = h5_before['auto_tile_i']

    expected_auto_gain = h5_after['auto_gain']

    result_auto_gain = vis_cal_auto_init(obs, cal, vis_arr, vis_model_arr, vis_auto, vis_model_auto, auto_tile_i)

    ##Check returned gain is as expected
    npt.assert_allclose(result_auto_gain, expected_auto_gain, atol=1e-4)

# def test_pointsource1_standard(data_dir):
#     """Test using the `pointsource1_standard` set of inputs"""

#     run_test(data_dir, "pointsource1_standard")
    
def test_pointsource2_vary1(data_dir):
    """Test using the `pointsource1_standard` set of inputs"""

    run_test(data_dir, "pointsource2_vary1")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_cal_auto_init`
        and converts into an hdf5 format"""

        func_name = 'vis_cal_auto_init'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        cal = recarray_to_dict(sav_dict['cal'])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['vis_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
        h5_save_dict['vis_model_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_arr'])

        h5_save_dict['vis_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_auto'])
        h5_save_dict['vis_model_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_auto'])

        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']
        
        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_cal_auto_init`
        and converts into an hdf5 format"""

        func_name = 'vis_cal_auto_init'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['auto_gain'] = sav_file_vis_arr_swap_axes(sav_dict["auto_gain"])
        
        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_cal_auto_init`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_cal_auto_init')

    ##Each test_set contains a run with a different set of inputs/options
    tag_names = ['pointsource2_vary1']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)