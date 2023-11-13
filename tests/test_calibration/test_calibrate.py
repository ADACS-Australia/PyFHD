from PyFHD.io.pyfhd_io import recarray_to_dict
import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.calibration.calibrate import calibrate
from PyFHD.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from numpy.testing import assert_allclose
from PyFHD.source_modeling.vis_model_transfer import vis_model_transfer, flag_model_visibilities
from PyFHD.io.pyfhd_io import save, load
from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'vis_calibrate')

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run2', 'run3'])
def run(request):
    return request.param

@pytest.fixture()
def before_file(tag, run, data_dir):
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = {}
    h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
    h5_save_dict['obs']['n_baselines'] = h5_save_dict['obs']['nbaselines']
    h5_save_dict['params'] = recarray_to_dict(sav_dict['params'])
    h5_save_dict['vis_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
    h5_save_dict['vis_weights'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weights'])

    # Create slimmed down version of pyfhd_config
    pyfhd_config = {}
    pyfhd_config['model_file_type'] = 'uvfits'
    pyfhd_config['model_file_path'] = str(Path(data_dir, "point_zenith_8s_80kHz_analy_autos.uvfits"))
    pyfhd_config['obs_id'] = 1088176296
    pyfhd_config['n_pol'] = h5_save_dict['obs']['n_pol']
    pyfhd_config['cal_convergence_threshold'] = 1e-7
    pyfhd_config["bandpass_calibrate"] = False
    pyfhd_config["calibration_auto_initialize"] = False
    pyfhd_config["flag_calibration"] = True
    pyfhd_config["cal_gain_init"] = 1
    pyfhd_config["cal_base_gain"] = 0.75
    pyfhd_config["calibration_auto_fit"] = False
    if run in ["run1", "run3"]:
        pyfhd_config["bandpass_calibrate"] = True
    if run == "run2":
        pyfhd_config["calibration_auto_initialize"] = True
    pyfhd_config["calibration_polyfit"] = True

    if "1088716296" in tag:
        pyfhd_config['model_file_type'] = 'sav'
        pyfhd_config['model_file_path'] = str(data_dir)
        pyfhd_config['obs_id'] = 1088716296
    # This test will assume all other tests have been run before
    # Fill pyfhd_config from the bandpass before
    bandpass_dir = Path(data_dir.parent, 'vis_cal_bandpass')
    bandpass_file = Path(bandpass_dir, f"{tag}_{run}_before_{bandpass_dir.name}.h5")
    if bandpass_file.exists():
        bandpass = load(bandpass_file)
        bandpass['pyfhd_config']['cal_bp_transfer'] = None
        pyfhd_config.update(bandpass['pyfhd_config'])
    # Fill pyfhd_config from the polyfit before
    polyfit_dir = Path(data_dir.parent, 'vis_cal_polyfit')
    polyfit_file = Path(polyfit_dir, f"{tag}_{run}_before_{polyfit_dir.name}.h5")
    if polyfit_file.exists():
        polyfit = load(polyfit_file)
        pyfhd_config.update(polyfit['pyfhd_config'])
        pyfhd_config["cable_reflection_coefficients"] = str(pyfhd_config["cable_reflection_coefficients"])
        pyfhd_config["cable_lengths"] = str(pyfhd_config["cable_lengths"]) 
    # Fill pyfhd_config from the flag before
    flag_before = Path(data_dir.parent, 'vis_calibration_flag')
    flag_file = Path(flag_before, f"{tag}_{run}_before_{flag_before.name}.h5")
    if flag_file.exists():
        flag = load(flag_file)
        pyfhd_config.update(flag['pyfhd_config'])
    # Fill pyfhd_config from the vis_calibrate_subroutine
    subroutine_before = Path(data_dir.parent, 'vis_calibrate_subroutine')
    subroutine_file = Path(subroutine_before, f"{tag}_{run}_before_{subroutine_before.name}.h5")
    if subroutine_file.exists():
        subroutine = load(subroutine_file)
        pyfhd_config.update(subroutine['pyfhd_config'])

    h5_save_dict['pyfhd_config'] = pyfhd_config

    # Read the vis_model_arr
    vis_model_arr, params_model = vis_model_transfer(pyfhd_config, h5_save_dict['obs'], RootLogger(1))
    # No flagging was done for point zenith or point offzenith
    if tag == "1088716296":
        vis_model_arr = flag_model_visibilities(vis_model_arr, h5_save_dict['params'], params_model, h5_save_dict['obs'], pyfhd_config, RootLogger(1))

    h5_save_dict['vis_model_arr'] = vis_model_arr

    save(before_file, h5_save_dict, "before_file")

    return before_file

@pytest.fixture()
def after_file(tag, run, data_dir):
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    
    sav_file = after_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = {}
    h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
    h5_save_dict['vis_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
    h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])

    # Swap the freq and tile dimensions
    # this make shape (n_pol, n_freq, n_tile)
    gain = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
    h5_save_dict['cal']['gain'] = gain

    # Do the same formatting for the amp, phase and mode params as per the polyfit tests
    expected_amp_params = np.empty(
        (h5_save_dict['cal']['n_pol'], 
         h5_save_dict['obs']['n_tile'], 
         h5_save_dict['cal']['amp_degree'], 
         h5_save_dict['cal']['amp_degree']
        )
    )
    expected_phase_params = np.empty((
        h5_save_dict['cal']['n_pol'],
        h5_save_dict['obs']['n_tile'],
        h5_save_dict['cal']['phase_degree'] + 1
    ))
    expected_mode_params = np.empty((
        h5_save_dict['cal']['n_pol'],
        h5_save_dict['obs']['n_tile'],
        3
    ))
    # Recarray didn't know what to do with it, so turn it back into object array
    h5_save_dict['cal']['mode_params'] = np.array(h5_save_dict['cal']['mode_params'])
    # Convert the amp, phase and mode params to not object arrays, or put them in the right size
    for pol_i in range(h5_save_dict['cal']['n_pol']):
        for tile_i in range(h5_save_dict['obs']['n_tile']):
            expected_amp_params[pol_i, tile_i] = np.transpose(h5_save_dict['cal']['amp_params'][tile_i, pol_i])
            expected_phase_params[pol_i, tile_i] = np.vstack(h5_save_dict['cal']['phase_params'][tile_i, pol_i]).flatten()
            if h5_save_dict['cal']['phase_params'][tile_i, pol_i][0] is None:
                expected_mode_params[pol_i, tile_i] = np.full(3, np.nan, dtype = np.float64)
            else:
                expected_mode_params[pol_i, tile_i] = h5_save_dict['cal']['mode_params'][tile_i, pol_i]
    h5_save_dict['cal']['amp_params'] = expected_amp_params
    h5_save_dict['cal']['phase_params'] = expected_phase_params
    h5_save_dict['cal']['mode_params'] = expected_mode_params

    save(after_file, h5_save_dict, "after_file")

    return after_file

def test_calibrate(before_file, after_file):

    h5_before = load(before_file)
    h5_after = load(after_file)

    vis_cal, cal, obs = calibrate(
        h5_before['obs'],
        h5_before['params'],
        h5_before['vis_arr'],
        h5_before['vis_weights'],
        h5_before['vis_model_arr'],
        h5_before['pyfhd_config'],
        RootLogger(1)
    )
    actual_nan = np.nonzero(~np.isnan(cal['gain']))
    expected_nan = np.nonzero(np.isnan(h5_after['cal']['gain']))
    # The gain has been calculated
    assert_allclose(cal['gain'][actual_nan], h5_after['cal']['gain'][actual_nan], atol = 1.2e-6)
    # The visibilities should be changed, check them
    # assert_allclose(vis_cal, h5_after['vis_arr'], atol = 1e-8)
    # Not checking the amp, phase and mode_params as they shouldn't chnage after polyfit (nor are they used in PyFHD)
    # This test is focusing on the interaction between all the functions (i.e. integration test)
    # Check the changes to tile_use and freq_use
    # assert_allclose(obs['baseline_info']['tile_use'], h5_after['baseline_info']['tile_use'], atol = 1e-8)
    # assert_allclose(obs['baseline_info']['freq_use'], h5_after['baseline_info']['freq_use'], atol = 1e-8)
    