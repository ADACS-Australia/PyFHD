import numpy as np
from typing import Tuple
from astropy.constants import c
from PyFHD.pyfhd_tools.pyfhd_utils import meshgrid
import logging

def mwa_beam_setup_init(pyfhd_config : dict, obs : dict, antenna : dict) -> dict:
    """
    Initializes the antenna dictionary for a MWA observation.

    Parameters
    ----------
    pyfhd_config : dict
        PyFHD config
    obs : dict
        The observation dictionary containing observation metadata
    antenna : dict
        The basic antenna dictionary

    Returns
    -------
    antenna : dict
        The antenna dictionary setup for MWA
    """

    n_dipoles = 16
    n_ant_pol = antenna['n_ant_pol']
    n_tiles = obs['n_tiles']
    nfreq_bin = antenna['nfreq_bin']
    delay_settings = obs['delays']
    # meters (MWA groundscreen size)
    antenna_size = 5
    # meters (Same as M&C SetDelays script) Was 1.071 before? Verified in Tingay et al 2013
    antenna_spacing = 1.1
    # meters (June 2014 e-mail from Brian Crosse) Was 0.35 before
    antenna['height'] = 0.29
    # 435 picoseconds is base delay length unit [units in seconds]
    base_delay_unit = 4.35e-10

    antenna['coords'] = np.zeros((3, 16))
    # dipole east position (meters)
    xc_arr = (meshgrid(4,4,1) * antenna_spacing).flatten()
    antenna['coords'][0, :] = xc_arr - np.mean(xc_arr)
    # dipole north position (meters)
    yc_arr = np.flipud(meshgrid(4,4,2) * antenna_spacing).flatten()
    antenna['coords'][1, :] = yc_arr - np.mean(yc_arr)
    antenna['coords'][2, :] = np.zeros(16, dtype = np.float64)

    if delay_settings is None:
        D_d - xc_arr * np.sin(())

    return antenna