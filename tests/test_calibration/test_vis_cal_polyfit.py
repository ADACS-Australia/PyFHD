import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_polyfit
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy.testing as npt
import deepdish as dd
import importlib_resources
import numpy as np

from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_polyfit")

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run2', 'run3'])
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

    #Swap the freq and tile dimensions
    #this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(cal['gain'])
    cal['gain'] = gain

    fhd_keys = ["cal_reflection_mode_theory",
                "cal_reflection_mode_file",
                "cal_reflection_mode_delay",
                "cal_reflection_hyperresolve",
                "amp_degree",
                "phase_degree"]
    config_keys = ["cal_reflection_mode_theory",
                    "cal_reflection_mode_file",
                    "cal_reflection_mode_delay",
                    "cal_reflection_hyperresolve",
                    "cal_amp_degree_fit",
                    "cal_phase_degree_fit"]
    
    #make a slimmed down version of pyfhd_config
    pyfhd_config = {}
    
    #When keys are unset in IDL, they just don't save to a .sav
    #file. So try accessing with an exception and set to None
    #if they don't exists
    for fhd_key, config_key in zip(fhd_keys, config_keys):
        try:
            pyfhd_config[config_key] = sav_dict[fhd_key]
        except KeyError:
            pyfhd_config[config_key] = None

    pyfhd_config["cable_reflection_coefficients"] = importlib_resources.files('PyFHD.templates').joinpath('mwa_cable_reflection_coefficients.txt')
    pyfhd_config["cable_lengths"] = importlib_resources.files('PyFHD.templates').joinpath('mwa_cable_length.txt')
    pyfhd_config['digital_gain_jump_polyfit'] = True
    
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

    cal_return = recarray_to_dict(sav_dict['cal_return'])
        
        
    #Swap the freq and tile dimensions
    #this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(cal_return['gain'])
    cal_return['gain'] = gain
    
    #super dictionary to save everything in
    h5_save_dict = {}
    
    h5_save_dict['cal_return'] = cal_return

    dd.io.save(after_file, h5_save_dict)

    return after_file


def test_vis_cal_polyfit(before_file, after_file):
    """Runs the test on `vis_cal_polyfit` - reads in the data in before_file and after_file,
    and then calls `vis_cal_polyfit`, checking the outputs match expectations"""
    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}")

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    obs = h5_before['obs']
    cal = h5_before['cal']
    
    pyfhd_config = h5_before['pyfhd_config']
    
    expected_cal_return = h5_after['cal_return']

    logger = RootLogger(1)
    
    cal_polyfit = vis_cal_polyfit(obs, cal, None, pyfhd_config, logger)

    # cal_return keeps amp_params as a pointer array of shape (128, 2)
    # However because digital_gain_jump_polyfit was used each pointer contains a (2,2)
    # This means the shape should be (2, 128, 2, 2) or 
    # (n_pol, n_tile, cal_amp_degree_fit, cal_amp_degree_fit)
    # cal_return also keeps the phase_params as (128, 2) object array, each one containing two 1 element float arrays 
    # This should be (n_pol, n_tile, cal_phase_degree_fit + 1) or (2, 128, 2) so
    # We'll grab each one by tile and polarization, and stack and flatten.
    expected_amp_params = np.empty(
        (cal_polyfit['n_pol'], 
         obs['n_tile'], 
         pyfhd_config['cal_amp_degree_fit'], 
         pyfhd_config['cal_amp_degree_fit']
        )
    )
    expected_phase_params = np.empty((
        cal_polyfit['n_pol'],
        obs['n_tile'],
        pyfhd_config['cal_phase_degree_fit'] + 1
    ))
    for pol_i in range(cal_polyfit['n_pol']):
        for tile_i in range(obs['n_tile']):
            expected_amp_params[pol_i, tile_i] = np.transpose(cal_polyfit['amp_params'][tile_i, pol_i])
            expected_phase_params[pol_i, tile_i] = np.vstack(cal_polyfit['phase_params'][tile_i, pol_i]).flatten()
    
    # 6e-8 atol for amp_params due to precision errors, still really close to single_precision
    npt.assert_allclose(cal_polyfit['amp_params'], expected_amp_params, atol=6e-8)
    npt.assert_allclose(cal_polyfit['phase_params'], expected_phase_params, atol=1e-8)
    npt.assert_allclose(cal_polyfit['gain'], expected_cal_return['gain'], atol=2e-7)