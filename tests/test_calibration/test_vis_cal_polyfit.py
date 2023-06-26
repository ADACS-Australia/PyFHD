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

def test_pointsource1_vary(data_dir):
    """Runs the test on `vis_cal_polyfit` - reads in the data in `data_loc`,
    and then calls `vis_cal_polyfit`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, "pointsource1_vary1_before_vis_cal_polyfit.h5"))
    h5_after = dd.io.load(Path(data_dir, "pointsource1_vary1_after_vis_cal_polyfit.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    
    pyfhd_config = h5_before['pyfhd_config']
    
    pyfhd_config["cable_reflection_coefficients"] = importlib_resources.files('PyFHD.templates').joinpath('mwa_cable_reflection_coefficients.txt')
    pyfhd_config["cable_lengths"] = importlib_resources.files('PyFHD.templates').joinpath('mwa_cable_length.txt')
    pyfhd_config['digital_gain_jump_polyfit'] = True
    
    expected_cal_return = h5_after['cal_return']
    # cal_return keeps amp_params as a pointer array of shape (128, 2)
    # However because digital_gain_jump_polyfit was used each pointer contains a (2,2)
    # This means the shape should be (2, 128, 2, 2) or 
    # (n_pol, n_tile, cal_amp_degree_fit, cal_amp_degree_fit)
    # cal_return also keeps the phase_params as (128, 2) object array, each one containing two 1 element float arrays 
    # This should be (n_pol, n_tile, cal_phase_degree_fit + 1) or (2, 128, 2) so
    # We'll grab each one by tile and polarization, and stack and flatten.
    expected_amp_params = np.empty(
        (expected_cal_return['n_pol'], 
         obs['n_tile'], 
         pyfhd_config['cal_amp_degree_fit'], 
         pyfhd_config['cal_amp_degree_fit']
        )
    )
    expected_phase_params = np.empty((
        expected_cal_return['n_pol'],
        obs['n_tile'],
        pyfhd_config['cal_phase_degree_fit'] + 1
    ))
    for pol_i in range(expected_cal_return['n_pol']):
        for tile_i in range(obs['n_tile']):
            expected_amp_params[pol_i, tile_i] = np.transpose(expected_cal_return['amp_params'][tile_i, pol_i])
            expected_phase_params[pol_i, tile_i] = np.vstack(expected_cal_return['phase_params'][tile_i, pol_i]).flatten()

    logger = RootLogger(1)
    
    cal_polyfit = vis_cal_polyfit(obs, cal, None, pyfhd_config, logger)
    
    # 6e-8 atol for amp_params due to precision errors, still really close to single_precision
    npt.assert_allclose(cal_polyfit['amp_params'], expected_amp_params, atol=6e-8)
    npt.assert_allclose(cal_polyfit['phase_params'], expected_phase_params)
    npt.assert_allclose(cal_polyfit['gain'], expected_cal_return['gain'])

    
if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_cal_polyfit`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_vis_cal_polyfit.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        cal = recarray_to_dict(sav_dict['cal'])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal['gain'])
        cal['gain'] = gain

        ##TODO these probably need reshaping
        # cal['amp_params']
        # cal['phase_params']

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
        
        ##make a slimmed down version of pyfhd_config
        pyfhd_config = {}
        
        ##When keys are unset in IDL, they just don't save to a .sav
        ##file. So try accessing with an exception and set to None
        ##if they don't exists
        for fhd_key, config_key in zip(fhd_keys, config_keys):
            try:
                pyfhd_config[config_key] = sav_dict[fhd_key]
            except KeyError:
                pyfhd_config[config_key] = None
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['pyfhd_config'] = pyfhd_config
        
        dd.io.save(Path(data_dir, f"{tag_name}_before_vis_cal_polyfit.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_cal_polyfit`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_vis_cal_polyfit.sav", "meh")
        
        cal_return = recarray_to_dict(sav_dict['cal_return'])
        
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        gain = sav_file_vis_arr_swap_axes(cal_return['gain'])
        cal_return['gain'] = gain
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_return'] = cal_return
        dd.io.save(Path(data_dir, f"{tag_name}_after_vis_cal_polyfit.h5"), h5_save_dict)
        
    def convert_sav(data_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_cal_polyfit`"""
        convert_before_sav(data_dir, tag_name)
        convert_after_sav(data_dir, tag_name)

    ##Where be all of our data
    data_dir = Path(env.get('PYFHD_TEST_PATH'), "vis_cal_polyfit")

    print("DATA DIR IS", data_dir)

    ##Each test_set contains a run with a different set of inputs/options
    ##TODO get the tag_names from some kind of glob on the relevant dir
    tag_names = ['pointsource1_vary1', 'pointsource1_standard',  'pointsource1_vary2']

    ##Each test_set contains a run with a different set of inputs/options
    for tag_name in tag_names:
        convert_sav(data_dir, tag_name)