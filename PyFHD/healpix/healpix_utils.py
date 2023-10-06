import numpy as np
from logging import RootLogger
import deepdish as dd
from PyFHD.pyfhd_tools.pyfhd_utils import angle_difference
import importlib_resources
from healpy.pixelfunc import pix2vec, vec2ang, ang2vec
from healpy import query_disc
from PyFHD.pyfhd_tools.unit_conv import radec_to_pixel, radec_to_altaz

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
        pix_coords = np.vstack(pix2vec(nside, hpx_inds)).T
        pix_dec, pix_ra = vec2ang(pix_coords, lonlat=True)
        xv_hpx, yv_hpx = radec_to_pixel(pix_ra, pix_dec, obs["astr"])
    else:
        cen_coords = ang2vec(obs["obsra"], obs["obsdec"], lonlat=True)
        hpx_inds = query_disc(nside, cen_coords, hpx_radius)
        pix_coords = np.vstack(pix2vec(nside, hpx_inds)).T
        pix_dec, pix_ra = vec2ang(pix_coords, lonlat=True)
        xv_hpx, yv_hpx = radec_to_pixel(pix_ra, pix_dec, obs["astr"])
        # slightly more restrictive boundary here ('LT' and 'GT' instead of 'LE' and 'GE') 
        pix_i_use = np.where((xv_hpx > 0) & (xv_hpx < obs['dimension'] - 1) & (yv_hpx > 0) & (yv_hpx < obs['elements'] - 1))
        xv_hpx = xv_hpx[pix_i_use]
        yv_hpx = yv_hpx[pix_i_use]
        hpx_mask00 = mask[np.floor(xv_hpx), np.floor(yv_hpx)]
        hpx_mask01 = mask[np.floor(xv_hpx), np.ceil(yv_hpx)]
        hpx_mask10 = mask[np.ceil(xv_hpx), np.floor(yv_hpx)]
        hpx_mask11 = mask[np.ceil(xv_hpx), np.ceil(yv_hpx)]
        hpx_mask = hpx_mask00 * hpx_mask01 * hpx_mask10 * hpx_mask11
        pix_i_use2 = np.nonzero(hpx_mask)
        xv_hpx = xv_hpx[pix_i_use2]
        yv_hpx = yv_hpx[pix_i_use2]
        pix_i_use = pix_i_use[pix_i_use2]
        hpx_inds = hpx_inds[pix_i_use]
    
    # Test for pixels past the horizon. We don't need to be precise with this, so turn off precession, etc..
    alt, _ = radec_to_altaz(pix_ra, pix_dec, obs["lat"], obs["lon"], obs["alt"], obs["jd0"])
    horizon_i = np.where(alt <= 0)
    if np.size(horizon_i) > 0:
        logger.info("Cutting the HEALPix pixels that were below the horizon")
        h_use = np.where(alt > 0)
        xv_hpx = xv_hpx[h_use]
        yv_hpx = yv_hpx[h_use]
        hpx_inds = hpx_inds[h_use]
    
    x_frac = 1 - (xv_hpx - np.floor(xv_hpx))
    y_frac = 1 - (yv_hpx - np.floor(yv_hpx))

    

