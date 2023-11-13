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

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run2', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"], ["point_offzenith", "run3"]]

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

    print(type(sav_dict['obs']))

    obs = recarray_to_dict(sav_dict['obs'])
    cal = recarray_to_dict(sav_dict['cal'])
    
    #Swap the freq and baseline dimensions
    gain = sav_file_vis_arr_swap_axes(cal['gain'])
    cal['gain'] = gain
    
    #Make a small pyfhd_config with just the variables needed for this func
    pyfhd_config = {}
    pyfhd_config['amp_degree'] = sav_dict['amp_degree']
    pyfhd_config['phase_degree'] = sav_dict['phase_degree']
    
    #super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = obs
    h5_save_dict['cal'] = cal
    h5_save_dict['pyfhd_config'] = pyfhd_config

    dd.io.save(before_file, h5_save_dict)

    return before_file

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

    obs = recarray_to_dict(sav_dict['obs'])

    #super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = obs

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_vis_calibration_flag(before_file, after_file):
    """Runs the test on `vis_calibration_flag` - reads in the data in before_file and after_file,
    and then calls `vis_calibration_flag`, checking the outputs match expectations"""
    if (before_file == None or after_file == None):
        pytest.skip(f"""This test has been skipped because the test was listed in the 
                    skipped tests due to FHD not outputting them: {skip_tests}. The reason
                    for skipping point_offzenith_run3 is due to the difference of NumPy median
                    and median in IDL. IDL doesn't take the average of the two middle numbers
                    when dealing with an an array of even values. It wasn't a problem in the other
                    cases, but not this one, and because we're dealing with a continuous set of data,
                    it doesn't feel appropriate to use the IDL median.
                    """)

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    obs_in = h5_before['obs']
    cal = h5_before['cal']
    pyfhd_config = h5_before['pyfhd_config']
    pyfhd_config['cal_amp_degree_fit'] = 2
    pyfhd_config['cal_phase_degree_fit'] = 1
    
    expected_obs = h5_after['obs']
    
    logger = RootLogger(1)
    
    result_obs = vis_calibration_flag(obs_in, cal, pyfhd_config, logger)

    # Check the values of tile_use and freq_use inside baseline_info
    assert np.array_equal(result_obs["baseline_info"]["freq_use"], expected_obs["baseline_info"]["freq_use"])
    differences = np.nonzero(expected_obs["baseline_info"]["tile_use"] - result_obs["baseline_info"]["tile_use"])
    assert np.array_equal(result_obs["baseline_info"]["tile_use"], expected_obs["baseline_info"]["tile_use"])