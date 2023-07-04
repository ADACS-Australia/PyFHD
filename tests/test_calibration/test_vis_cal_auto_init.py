import pytest
from os import environ as env
from pathlib import Path
from PyFHD.calibration.calibration_utils import vis_cal_auto_init
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import deepdish as dd
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_init")

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run2', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"]]

@pytest.fixture()
def before_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

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

    dd.io.save(before_file, h5_save_dict)

    return before_file

# Same as the before_file fixture, except we're taking the the after files
@pytest.fixture()
def after_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    
    sav_file = after_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['auto_gain'] = sav_file_vis_arr_swap_axes(sav_dict["auto_gain"])

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_vis_cal_auto_init(before_file, after_file):
    """Runs the test on `vis_cal_auto_init` - reads in the data in before_file & after_file,
    and then calls `vis_cal_auto_init`, checking the outputs match expectations"""
    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outpoutting them: {skip_tests}")

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

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
    npt.assert_allclose(result_auto_gain, expected_auto_gain, atol=1e-7)