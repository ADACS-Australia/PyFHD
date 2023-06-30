import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibrate import calibrate_qu_mixing
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
from logging import RootLogger
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_calibrate_qu_mixing")

def run_test(data_dir, tag_name):
    """Runs the test on `calibrate_qu_mixing` - reads in the data in `data_loc`,
    and then calls `calibrate_qu_mixing`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_vis_calibrate_qu_mixing.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_vis_calibrate_qu_mixing.h5"))

    obs = h5_before['obs']
    vis_ptr = h5_before['vis_ptr']
    vis_model_ptr = h5_before['vis_model_ptr']
    vis_weight_ptr = h5_before['vis_weight_ptr']

    expected_calc_phase = h5_after['calc_phase']

    result_cal_phase = calibrate_qu_mixing(vis_ptr, vis_model_ptr,
                                          vis_weight_ptr, obs)
    
    atol = 1e-5

    npt.assert_allclose(expected_calc_phase, result_cal_phase, atol=atol)


def test_pointsource2_vary3(data_dir):
    """Test using the `pointsource2_vary3` set of inputs"""

    run_test(data_dir, "pointsource2_vary3")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `calibrate_qu_mixing`
        and converts into an hdf5 format"""

        func_name = 'vis_calibrate_qu_mixing'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
        h5_save_dict['vis_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_ptr'])
        h5_save_dict['vis_model_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_ptr'])
        h5_save_dict['vis_weight_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weight_ptr'])

        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `calibrate_qu_mixing`
        and converts into an hdf5 format"""

        func_name = 'vis_calibrate_qu_mixing'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['calc_phase'] = sav_dict['calc_phase']

        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `calibrate_qu_mixing`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_calibrate_qu_mixing')

    tag_names = ['pointsource2_vary3']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)