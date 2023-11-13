import numpy as np
from typing import Tuple
from astropy.constants import c
import logging
from scipy.interpolate import interp1d
from PyFHD.beam_setup.beam_utils import mwa_beam_setup_init

def create_psf(pyfhd_config : dict, obs : dict) -> Tuple[dict, dict]:

    psf = {}

    # Add ability later here to restore an old psf
    
    freq_bin_i = obs['baseline_info']['fbin_i']
    nfreq_bin = np.max(freq_bin_i) + 1
    antenna = create_antenna(pyfhd_config, obs)
    
    return psf, antenna

def create_antenna(pyfhd_config : dict, obs : dict) -> dict:
    """_summary_

    Parameters
    ----------
    pyfhd_config : dict
        _description_
    obs : dict
        _description_

    Returns
    -------
    antenna : dict
        _description_
    """

    # Setup the constants and variables
    n_tiles = obs['n_tile']
    n_freq = obs['n_freq']
    n_pol = obs['n_pol']
    # Almost all instruments have two instrumental polarizations (either linear or circular)
    n_ant_pol = 2
    obsra = obs['obsra']
    obsdec = obs['obsdec']
    zenra = obs['zenra']
    zendec = obs['zendec']
    obsx = obs['obsx']
    obsy = obs['obsy']
    dimension = obs['dimension']
    elements = obs['elements']
    kbinsize = obs['kpix']
    degpix = obs['degpix']
    astr = obs['astr']
    psf_image_resolution = 10
    frequency_array = obs['baseline_info']['freq']
    freq_bin_i = obs['baseline_info']['fbin_i']
    nfreq_bin = int(np.max(freq_bin_i)) + 1
    tile_a = obs['baseline_info']['tile_a']
    tile_b = obs['baseline_info']['tile_b']
    ant_names = np.unique(tile_a[: obs['n_baselines']])
    if pyfhd_config['beam_offset_time'] is not None:
        jdate_use = obs['jd0'] + pyfhd_config['beam_offset_time'] / 24 / 3600
    else:
        jdate_use = obs['jd0']
    if pyfhd_config['psf_resolution'] is None:
        psf_resolution = 16
    
    freq_center = np.zeros(nfreq_bin)
    interp_func = interp1d(freq_bin_i, frequency_array)
    for fi in range(nfreq_bin):
        fi_i = np.where(freq_bin_i == fi)[0]
        if fi_i.size == 0:
            freq_center[fi] = interp_func(fi)
        else:
            freq_center[fi] = np.median(frequency_array[fi_i])
    
    # Create basic antenna dictionary
    antenna = {
        'n_pol' : n_ant_pol,
        'antenna_type' : pyfhd_config['instrument'],
        'names' : ant_names,
        'beam_model_version' : pyfhd_config['beam_model_version'],
        'freq' : freq_center,
        'nfreq_bin' : nfreq_bin,
        'n_ant_elements' : 0,
        # Anything that was pointer arrays in IDL will be None until assigned in Python 
        'jones' : None,
        'coupling' : None,
        'gain' : None,
        'coords' : None,
        'delays' : None,
        'size_meters' : 0,
        'height' : 0,
        'response' : None,
        'group_id' : np.full(n_ant_pol, -1, dtype = np.int64),
        'pix_window' : None,
        'pix_use' : None,
        'psf_image_dim' : 0,
        'psf_scale' : 0
    }

    # We are building PyFHD with only MWA in mind at the moment
    antenna = mwa_beam_setup_init(pyfhd_config, obs, antenna)

    return antenna