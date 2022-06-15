import numpy as np
from typing import Tuple
from astropy.constants import c
import logging

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

    antenna = {}
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
    tile_A = obs['baseline_info']['tile_A']
    tile_B = obs['baseline_info']['tile_B']
    ant_names = np.unique(tile_A[: obs['nbaselines']])
    if pyfhd_config['beam_offset_time'] is not None:
        jdate_use = obs['jd0'] + pyfhd_config['beam_offset_time'] / 24 / 3600
    else:
        jdate_use = obs['jd0']
    if pyfhd_config['psf_resolution'] is None:
        psf_resolution = 16
    
    


    return antenna