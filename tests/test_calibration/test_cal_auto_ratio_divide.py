import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import cal_auto_ratio_divide
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
from logging import RootLogger
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "cal_auto_ratio_divide")

def run_test(data_dir, tag_name):
    """Runs the test on `cal_auto_ratio_divide` - reads in the data in `data_loc`,
    and then calls `cal_auto_ratio_divide`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_cal_auto_ratio_divide.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_cal_auto_ratio_divide.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    vis_auto = h5_before['vis_auto']
    auto_tile_i = h5_before['auto_tile_i']

    expected_cal = h5_after['cal']
    expected_auto_ratio = h5_after['auto_ratio']

    result_cal, result_auto_ratio = cal_auto_ratio_divide(obs, cal, vis_auto, auto_tile_i)

    atol = 1e-6
    npt.assert_allclose(expected_auto_ratio, result_auto_ratio, atol=atol)

    ##check the gains have been updated
    npt.assert_allclose(expected_cal['gain'], result_cal['gain'], atol=atol)


def test_pointsource1_vary1(data_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(data_dir, "pointsource1_vary1")

def test_pointsource2_vary1(data_dir):
    """Test using the `pointsource2_vary1` set of inputs"""

    run_test(data_dir, "pointsource2_vary1")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `cal_auto_ratio_divide`
        and converts into an hdf5 format"""

        func_name = 'cal_auto_ratio_divide'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        cal = recarray_to_dict(sav_dict['cal'])
        vis_auto = sav_file_vis_arr_swap_axes(sav_dict['vis_auto'])
            
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
        h5_save_dict['vis_auto'] = vis_auto
        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']

        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `cal_auto_ratio_divide`
        and converts into an hdf5 format"""

        func_name = 'cal_auto_ratio_divide'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])
        h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
        h5_save_dict['auto_ratio'] = sav_file_vis_arr_swap_axes(sav_dict['auto_ratio'])

        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `cal_auto_ratio_divide`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'cal_auto_ratio_divide')

    tag_names = ['pointsource1_vary1', 'pointsource2_vary1']
    # tag_names = ['pointsource2_vary1']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)