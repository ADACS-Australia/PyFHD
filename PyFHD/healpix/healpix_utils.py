import numpy as np
from logging import RootLogger
import deepdish as dd
from PyFHD.pyfhd_tools.pyfhd_utils import angle_difference
import importlib_resources

def healpix_cnv_generate(obs: dict, mask: np.ndarray, hpx_radius: float, pyfhd_config: dict, logger: RootLogger) -> dict:
    """
    TODO:_summary_

    All angles are in degrees
    Uses the RING index scheme

    Parameters
    ----------
    obs : dict
        _description_
    mask : np.ndarray
        _description_
    hpx_radius : float
        _description_
    pyfhd_config : dict
        _description_
    logger : RootLogger
        _description_

    Returns
    -------
    dict
        _description_
    """
    # TODO: Add ability to restore old healpix_inds from PyFHD runs

    # Have ignored the code that relates to hpx_radius being empty, in PyFHD's case
    # we will assume that a value for hpx_radius is supplied, if you want to add it
    # yourself you can do that here
    hpx_inds = None
    nside = None
    # Fill hpx_inds and nside with values from a file if restrict_healpix_inds has been activated
    if (pyfhd_config["restrict_healpix_inds"]):
        if pyfhd_config["healpix_inds"] is None:
            # Get the healpix indexes based off the observation, comes from observation_healpix_inds_select
            files = np.array([
                {
                    "name": "EoR0_high_healpix_inds.h5",
                    "ra": 0,
                    "dec": -30,
                    "freq": 182,
                },
                {
                    "name": "EoR0_low_healpix_inds.h5",
                    "ra": 0,
                    "dec": -30,
                    "freq": 151,
                },
                {
                    "name": "EoR1_high_healpix_inds.h5",
                    "ra": 60,
                    "dec": -30,
                    "freq": 182,
                },
                {
                    "name": "EoR1_low_healpix_inds.h5",
                    "ra": 60,
                    "dec": -30,
                    "freq": 151,
                },
            ])
            ang_dist = []
            for file in files:
                ang_dist.append(np.abs(angle_difference(obs["obsra"], obs["obsdec"], file["ra"], file["dec"],degree = True, nearest = True)))
            ang_dist = np.array(ang_dist)
            ang_use = np.min(ang_dist)
            i_use = np.where(np.abs(ang_dist - ang_use) <= 1)
            files = files[i_use]
            freq_dist = []
            for file in files:
                freq_dist.append(file["freq"])
            freq_dist = np.abs(np.array(freq_dist) - (obs["freq_center"]/1e6))
            min_i = np.argmin(freq_dist)
            pyfhd_config["healpix_inds"] = importlib_resources.files('PyFHD.templates').joinpath(files[min_i]["name"])
        hpx_inds = dd.io.load(pyfhd_config["healpix_inds"])
        nside = hpx_inds["nside"]
        hpx_inds = hpx_inds["hpx_inds"]
    if nside is None:
        pix_sky = 4 * np.pi * ((180 / np.pi) ** 2) /  np.prod(np.abs(obs["astr"]["cdelt"]))
        nside = 2 ** (np.ceil(np.log(np.sqrt(pix_sky/12)) / np.log(2)))
    
    # If you wish to implement the keyword divide_pixel_area implement it here and 
    # add it as an option inside pyfhd_config

    if hpx_inds is not None:
        

