import pytest
from os import environ as env
from pathlib import Path
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

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run3'])
def run(request):
    return request.param

@pytest.fixture
def before_file(tag, run, data_dir):
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
    h5_save_dict['vis_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_ptr'])
    h5_save_dict['vis_model_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_ptr'])
    h5_save_dict['vis_weight_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weight_ptr'])

    dd.io.save(before_file, h5_save_dict)

    return before_file

@pytest.fixture
def after_file(tag, run, data_dir):
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    
    sav_file = after_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['calc_phase'] = sav_dict['calc_phase']

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_qu_mixing(before_file, after_file):
    """Runs the test on `calibrate_qu_mixing` - reads in the data in `data_loc`,
    and then calls `calibrate_qu_mixing`, checking the outputs match expectations"""

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    obs = h5_before['obs']
    vis_ptr = h5_before['vis_ptr']
    vis_model_ptr = h5_before['vis_model_ptr']
    vis_weight_ptr = h5_before['vis_weight_ptr']

    expected_calc_phase = h5_after['calc_phase']

    result_cal_phase = calibrate_qu_mixing(vis_ptr, vis_model_ptr,
                                          vis_weight_ptr, obs)
    
    atol = 1e-5

    npt.assert_allclose(expected_calc_phase, result_cal_phase, atol=atol)

# if __name__ == "__main__":

#     def convert_before_sav(data_dir, tag_name):
#         """Takes the before .sav file out of FHD function `calibrate_qu_mixing`
#         and converts into an hdf5 format"""

#         func_name = 'vis_calibrate_qu_mixing'

#         sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

#         ##super dictionary to save everything in
#         h5_save_dict = {}
#         h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
#         h5_save_dict['vis_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_ptr'])
#         h5_save_dict['vis_model_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_ptr'])
#         h5_save_dict['vis_weight_ptr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weight_ptr'])

#         dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
#     def convert_after_sav(data_dir, tag_name):
#         """Takes the after .sav file out of FHD function `calibrate_qu_mixing`
#         and converts into an hdf5 format"""

#         func_name = 'vis_calibrate_qu_mixing'

#         sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
#         ##super dictionary to save everything in
#         h5_save_dict = {}
        
#         h5_save_dict['calc_phase'] = sav_dict['calc_phase']

#         dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
#     def convert_sav(base_dir, tag_name):
#         """Load the inputs and outputs needed for testing `calibrate_qu_mixing`"""
#         convert_before_sav(base_dir, tag_name)
#         convert_after_sav(base_dir, tag_name)

#     ##Where be all of our data
#     base_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_calibrate_qu_mixing')

#     tag_names = ['pointsource2_vary3']

#     for tag_name in tag_names:
#         convert_sav(base_dir, tag_name)
#         # run_test(base_dir, tag_name)