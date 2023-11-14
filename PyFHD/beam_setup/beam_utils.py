import numpy as np
from astropy.constants import c
from PyFHD.pyfhd_tools.pyfhd_utils import histogram
import logging
import h5py

def beam_image(psf: dict | h5py.File, obs: dict, pol_i: int, freq_i: int, abs = False, square = False) -> np.ndarray:
    """
    TODO: _summary_

    Parameters
    ----------
    psf : dict
        _description_
    obs : dict
        _description_
    pol_i : int
        _description_
    freq_i : int
        _description_
    abs : bool, optional
        _description_, by default False
    square : bool, optional
        _description_, by default False

    Returns
    -------
    beam_base : np.ndarray
        _description_
    """

    psf_dim = psf['dim']
    freq_norm = psf['fnorm']
    pix_horizon = psf['pix_horizon']
    group_id = psf['id'][pol_i, 0, :]
    if "beam_gaussian_params" in psf:
        beam_gaussian_params = psf['beam_gaussian_params'][:]
    else: 
        beam_gaussian_params = None
    rbin = 0
    # If we lazy loaded psf, get actual numbers out of the datasets
    if isinstance(psf, h5py.File):
        psf_dim = psf_dim[0]
        freq_norm = freq_norm[0]
        pix_horizon = pix_horizon[0]
    dimension = elements = obs['dimension']
    xl = dimension / 2 - psf_dim / 2 + 1
    xh = dimension / 2 - psf_dim / 2 + psf_dim
    yl = elements / 2 - psf_dim / 2 + 1
    yh = elements / 2 - psf_dim / 2 + psf_dim

    group_n, _, ri_id = histogram(group_id, min = 0)
    gi_use = np.nonzero(group_n)
    gi_ref = ri_id[ri_id[gi_use]]

    if beam_gaussian_params is not None:
        # 1.3 is the padding factor for the gaussian fitting procedure
        # (2.*obs.kpix) is the ratio of full sky (2 in l,m) to the analysis range (1/obs.kpix)
        # (2.*obs.kpix*dimension/psf.pix_horizon) is the scale factor between the psf pixels-to-horizon and the 
        # analysis pixels-to-horizon 
        # (0.5/obs.kpix) is the resolution scaling of what the beam model was made at and the current res 
        model_npix = pix_horizon * 1.3
        model_res = (2 * obs['kpix'] * dimension) / pix_horizon * (0.5 / obs['kpix'])
    
    freq_bin_i = obs["baseline_info"]["fbin_i"]
    n_freq = obs["n_freq"]
    n_bin_use = 0

    if square:
        beam_base = np.zeros([dimension, elements])
        
    else:
        if beam_gaussian_params is not None:
            beam_base_uv = np.zeros([dimension, elements])
            beam_single = np.zeros([dimension, elements])
        else:
            beam_base_uv = np.zeros([psf_dim, psf_dim])
            beam_single = np.zeros([psf_dim, psf_dim])

    return beam_base



    
    
    