import sys
from logging import RootLogger
from pathlib import Path
import h5py
import importlib_resources
import numpy as np
from healpy import query_disc
from healpy.pixelfunc import ang2vec, pix2vec, vec2ang
from PyFHD.beam_setup.beam_utils import beam_image
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.gridding.gridding_utils import dirty_image_generate
from PyFHD.io.pyfhd_io import load, save
from PyFHD.pyfhd_tools.pyfhd_utils import (angle_difference, histogram,
                                           meshgrid, region_grow)
from PyFHD.pyfhd_tools.unit_conv import radec_to_altaz, radec_to_pixel


def healpix_cnv_apply(image: np.ndarray, hpx_cnv: dict, transpose = True, mask = False) -> np.ndarray:
    """
    healpix_cnv_apply creates a map based off the array/image and healpix convention dictionary given.
    In FHD the healpix_cnv_apply was mainly used as a wrapper for sprsax2, as such I will put the code
    for sprsax2 in here as PyFHD will only use sprsax2 here as we don't have the holographic mapping
    function at this time.

    Parameters
    ----------
    image : np.ndarray
        _description_
    hpx_cnv : dict
        The HEALPix convention dictionary
    transpose : bool, optional
        Use a transpose of the image instead, by default True
    mask : bool, optional
        TODO: _description_, by default False

    Returns
    -------
    hpx_map: np.ndarray
        TODO: _description_
    """
    # Is B in sprsax2
    hpx_map = np.zeros(np.size(hpx_cnv['inds']))
    # Is X in sprsax2 or X_use
    image_vector = image.flatten()
    for i in range(0, np.size(hpx_cnv['i_use'])):
        i_use = hpx_cnv['i_use'][i]
        if transpose:
            hpx_map[hpx_cnv['ija'][i]] += hpx_cnv['sa'][i] * image_vector[i_use]
        else:
            x1 = image_vector[hpx_cnv['ija'][i]]
            # TODO: Check if outer is the correct multiplication and the order of matrices is correct
            hpx_map[i] = np.outer(x1, np.reshape(hpx_cnv['sa'][i], [np.size(hpx_cnv['sa'][i]), 1]))
    return hpx_map

def healpix_cnv_generate(obs: dict, mask: np.ndarray, hpx_radius: float, pyfhd_config: dict, logger: RootLogger, nside: float = None) -> dict:
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
    nside : float
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
        hpx_inds = load(pyfhd_config["healpix_inds"], logger = logger)
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

    v_floor = np.floor(xv_hpx) + obs['dimension'] * np.floor(yv_hpx)
    v_ceil = np.ceil(xv_hpx) + obs['dimension'] * np.ceil(yv_hpx)
    min_bin = max(np.min(v_floor), 0)
    max_bin = min(np.max(v_ceil), obs['dimension'] * obs['elements'] - 1)
    h00, _, ri00 = histogram(v_floor, min = min_bin, max = max_bin)
    h01, _, ri01 = histogram(np.floor(xv_hpx) + obs['dimension'] * np.ceil(yv_hpx), min = min_bin, max = max_bin)
    h10, _, ri10 = histogram(np.ceil(xv_hpx) + obs['dimension'] * np.floor(yv_hpx), min = min_bin, max = max_bin)
    h11, _, ri11 = histogram(v_ceil, min = min_bin, max = max_bin)
    htot = h00 + h01 + h10 + h11
    inds = np.nonzero(htot)

    n_arr = htot[inds]
    i_use = inds + min_bin
    sa = np.empty(np.size(n_arr), dtype=object)
    ija = np.empty(np.size(n_arr), dtype=object)

    for i in range(np.size(n_arr)):
        ind0 = inds[i]
        sa0 = np.zeros(n_arr[ind0])
        ija0 = np.zeros(n_arr[ind0], dtype = np.int64)
        hist_arr = np.array([0, h00[ind0], h01[ind0], h10[ind0], h11[ind0]], dtype = np.int64)
        bin_i = np.cumsum(hist_arr)
        for j in range(1, hist_arr.size):
            bi = j - 1
            if (hist_arr[j] > 0):
                if (j == 1):
                    # h00 > 0
                    inds1 = ri00[ri00[ind0] : ri00[ind0 + 1]]
                elif (j == 2):
                    # h01 > 0
                    inds1 = ri01[ri01[ind0] : ri01[ind0 + 1]]
                elif (j == 3):
                    # h10 > 0
                    inds1 = ri10[ri10[ind0] : ri10[ind0 + 1]]
                elif (j == 4):
                    # h11 > 0
                    inds1 = ri11[ri11[ind0] : ri11[ind0 + 1]]
                y_frac_multi = y_frac[inds1]
                x_frac_multi = x_frac[inds1]
                if j % 2 == 0:
                    y_frac_multi = 1 - y_frac_multi
                if j > 2:
                    x_frac_multi = 1 - x_frac_multi
                sa0[bin_i[bi] : bin_i[bi + 1]] = x_frac_multi * y_frac_multi
                ija0[bin_i[bi] : bin_i[bi + 1]] = inds1
        sa[i] = sa0
        ija[i] = ija0

    hpx_cnv = {
        "nside": nside,
        "ija": ija,
        "sa": sa,
        "i_use": i_use,
        "inds": hpx_inds
    }
    obs['healpix']['nside'] = nside
    if pyfhd_config["restrict_healpix_inds"]:
        obs['healpix']['ind_list'] = None
    else:
        obs['healpix']['ind_list'] = hpx_inds
    obs['healpix']['n_pix'] = np.size(hpx_inds)
    mask_test = healpix_cnv_apply(mask, hpx_cnv)
    mask_test_i0 = np.where(mask_test == 0)
    obs['healpix']['n_zero'] = np.size(mask_test_i0[0])

    return hpx_cnv, obs

def beam_image_cube(
    obs: dict, 
    psf: dict | h5py.File, 
    logger: RootLogger,
    freq_i_arr: np.ndarray | None = None, 
    pol_i_arr: np.ndarray | None = None,
    n_freq_bin: float | None = None,
    beam_mask: np.ndarray | None = None,
    square: bool = True,
    beam_threshold: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO: _summary_

    Parameters
    ----------
    obs : dict
        _description_
    psf : dict | h5py.File
        _description_
    logger : RootLogger
        _description_
    freq_i_arr : np.ndarray | None, optional
        _description_, by default None
    pol_i_arr : np.ndarray | None, optional
        _description_, by default None
    n_freq_bin : float | None, optional
        _description_, by default None
    beam_mask : np.ndarray | None, optional
        _description_, by default None
    square : bool, optional
        _description_, by default True
    beam_threshold : float | None, optional
        _description_, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        _description_
    """

    if beam_threshold is None:
        beam_threshold = 0.05
    
    if pol_i_arr is None:
        pol_i_arr = np.arange(obs['n_pol'])
    
    if n_freq_bin is not None:
        freq_i_arr = np.floor(np.arange(n_freq_bin) * (obs['n_freq'] / n_freq_bin)).astype(np.int64)
    if freq_i_arr is None:
        logger.error("beam_image_cube requires n_freq_bin or freq_i_arr to be set")
        sys.exit(1)
    if n_freq_bin is None and freq_i_arr is not None:
        n_freq_bin = freq_i_arr.size

    if np.median(freq_i_arr) > obs['n_freq']:
        freq_i_use = np.interp(np.arange(obs['n_freq']), obs['baseline_info']['freq'], freq_i_arr)
    else:
        freq_i_use = freq_i_arr
    
    # TODO What is the shape of this eventually?
    # TODO: Change shape of this based on beam_gaussian_params
    beam_arr = np.zeros([obs['n_pol'], n_freq_bin])

    bin_arr = obs['baseline_info']['fbin_i'][freq_i_use]
    bin_hist, _, bri = histogram(bin_arr, min = 0)
    bin_use = np.nonzero(bin_hist)
    if np.size(bin_use) == 0:
        return beam_arr
    bin_n = bin_hist[bin_use]
    beam_mask = np.ones(obs['dimension'], obs['elements'])
    for pol_i in range(obs['n_pol']):
        for fb_i in np.size(bin_use):
            f_i_i = bri[bri[bin_use[fb_i] : bri[bin_use[fb_i]]]]
            f_i = freq_i_use[f_i_i[0]]
            beam_single = beam_image(psf, obs, pol_i, f_i,  square = square)
            for bi in range(bin_n[fb_i]):
                beam_arr[pol_i, f_i_i[bi]] = beam_single
            b_i = obs['obsx'] + obs['obsy'] * obs['dimension']
            beam_i = region_grow(beam_single, b_i, low = beam_threshold ** (square + 1), max = np.max(beam_single))
            beam_mask1 = np.zeros([obs['dimension'], obs['elements']])
            beam_mask1[beam_i] = 1
            beam_mask *= beam_mask1
    return beam_arr, beam_mask

def phase_shift_uv_image(obs: dict) -> np.ndarray:
    """
    TODO: _summary_

    Parameters
    ----------
    obs : dict
        Observation metadata dictionary

    Returns
    -------
    np.ndarray
        _description_
    """
    # Since we only use it once in PyFHD, assume we always want to do /to_orig_phase
    ra_use = obs['orig_phasera']
    dec_use = obs['orig_phasedec']
    
    # Skip calculations if phased correctly
    if obs['phasera'] == obs['orig_phasera'] and obs['phasedec'] == obs['orig_phasedec']:
        return np.ones([obs['dimension'], obs['elements']], dtype = np.complex128)

    x, y = radec_to_pixel(ra_use, dec_use, obs['astr'])

    # uv_mask is not applied in PyFHD decided not to translate it, if you want it put it here

    dx = (x - (obs['dimension'] / 2)) * (2 * np.pi / obs['dimension'])
    dy = (y - (obs['elements'] / 2)) * (2 * np.pi / obs['dimension'])

    xvals = meshgrid(obs['dimension'], obs['elements'], 1) - (obs['dimension'] / 2)
    yvals = meshgrid(obs['dimension'], obs['elements'], 2) - (obs['elements'] / 2)

    phase = xvals * dx + yvals * dy
    rephase_vals = np.cos(phase) + np.sin(phase) * 1j

    # Again no uv_mask therefore just return the rephase as is

    return rephase_vals

def vis_model_freq_split(
    obs: dict, 
    psf: dict, 
    params: dict, 
    vis_weights: np.ndarray,
    vis_model_arr: np.ndarray,
    vis_arr: np.ndarray,
    pyfhd_config: dict,
    logger: RootLogger,
    fft: bool = True,
    save_uvf = True,
    uvf_name = '',
    bi_use = None
) -> dict:
    """
    TODO: _summary_

    Parameters
    ----------
    obs : dict
        _description_
    psf : dict
        _description_
    params : dict
        _description_
    vis_weights : np.ndarray
        _description_
    vis_model_arr : np.ndarray
        _description_
    vis_arr : np.ndarray
        _description_
    pyfhd_config : dict
        _description_
    logger : RootLogger
        _description_
    fft : bool, optional
        _description_, by default True
    save_uvf : bool, optional
        _description_, by default True
    uvf_name : str, optional
        _description_, by default ''
    bi_use : _type_, optional
        _description_, by default None

    Returns
    -------
    model_split: dict
        _description_
    """

    freq_bin_i2 = np.arange(obs['n_freq']) // pyfhd_config['n_avg']
    nf = np.max(freq_bin_i2) + 1
    if save_uvf:
        dirty_uv_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']], dtype = np.complex128)
        weights_uv_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']], dtype = np.int64)
        variance_uv_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']])
        model_uv_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']], dtype = np.complex128)
    dirty_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']], dtype = np.complex128)
    weights_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']], dtype = np.int64)
    variance_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']])
    model_arr = np.zeros([obs['n_pol'], nf, obs['dimension'], obs['dimension']], dtype = np.complex128)
    vis_n_arr = np.zeros([obs['n_pol'], nf])
    if pyfhd_config["rephase_weights"]:
        rephase_use = phase_shift_uv_image(obs)
    else:
        rephase_use = 1
    # No x_range and y_range is used in PyFHD, if you wish to do that add that here

    flag_test = np.maximum(np.maximum(vis_weights[0], vis_weights[1]), 0)
    # Double check the axis used
    flag_test = np.sum(flag_test, axis = 1)
    bi_use = np.where(flag_test > 0)[0]
    
    for pol_i in range(obs['n_pol']):
        n_vis_use = 0
        for fi in range(nf):
            fi_use = np.where((freq_bin_i2 == fi) & (obs['baseline_info']['freq_use'] > 0))
            if np.size(fi_use) == 0:
                n_vis = 0
            else:
                gridding_dict = visibility_grid(vis_arr[pol_i], vis_weights[pol_i], obs, psf, params, pol_i,
                                                pyfhd_config, logger, model = vis_model_arr[pol_i], fi_use = fi_use, 
                                                bi_use = bi_use)
            n_vis_use += gridding_dict['n_vis']
            vis_n_arr[pol_i, fi] = gridding_dict['n_vis']

            if save_uvf:
                dirty_uv_arr[pol_i, fi] = gridding_dict['image_uv'] * gridding_dict['n_vis']
                weights_uv_arr[pol_i, fi] = gridding_dict['weights'] * rephase_use * gridding_dict['n_vis']
                variance_uv_arr[pol_i, fi] = gridding_dict['variance'] * rephase_use * gridding_dict['n_vis']
                model_uv_arr[pol_i, fi] = gridding_dict['model_return'] * gridding_dict['n_vis']
            
            if fft:
                # No x_range and y_range hence no check for it here
                dirty_arr[pol_i, fi] = dirty_image_generate(
                    gridding_dict['image_uv'], 
                    pyfhd_config, 
                    logger, 
                    degpix = obs['degpix']
                ) * gridding_dict['n_vis']
                weights_arr[pol_i, fi] = dirty_image_generate(
                    gridding_dict['weights'] * rephase_use,
                    pyfhd_config,
                    logger,
                    degpix = obs['degpix']
                ) * gridding_dict['n_vis']
                variance_arr[pol_i, fi] = dirty_image_generate(
                    gridding_dict['variance'] * rephase_use,
                    pyfhd_config,
                    logger,
                    degpix = obs['degpix']
                ) * gridding_dict['n_vis']
                model_arr[pol_i, fi] = dirty_image_generate(
                    gridding_dict['model_return'] * gridding_dict['n_vis'],
                    pyfhd_config,
                    logger,
                    degpix = obs['degpix']
                ) * gridding_dict['n_vis']
            else:
                dirty_arr[pol_i, fi] = gridding_dict['image_uv'] * gridding_dict['n_vis']
                weights_arr[pol_i, fi] = gridding_dict['weights'] * rephase_use * gridding_dict['n_vis']
                variance_arr[pol_i, fi] = gridding_dict['variance'] * rephase_use * gridding_dict['n_vis']
                model_arr[pol_i, fi] = gridding_dict['model_return'] * gridding_dict['n_vis']
        obs['n_vis'] = n_vis_use

    if save_uvf:
        h5_save_dict = {
            "dirty_uv": dirty_uv_arr,
            "weights_uv": weights_uv_arr,
            "variance_uv": variance_uv_arr,
            "model_uv": model_uv_arr,
        }
        save(Path(pyfhd_config['output_dir'], f'{uvf_name}_gridded_uvf.h5'), h5_save_dict, f'{uvf_name}_gridded_uvf.h5', logger = logger)
    
    model_split = {
        "obs": obs,
        "residual_arr" : dirty_arr,
        "weights_arr": weights_arr,
        "variance_arr": variance_arr,
        "model_arr": model_arr,
    }

    return model_split