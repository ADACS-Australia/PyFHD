import numpy as np
from astropy.constants import c
from PyFHD.pyfhd_tools.pyfhd_utils import histogram
import h5py

def gaussian_decomp(
        x: np.ndarray, 
        y: np.ndarray, 
        p: np.ndarray, 
        ftransform: bool = False,  
        model_npix: float | None = None,
        model_res: float | None = None,
) -> tuple[np.ndarray, float, float]:
    decomp_beam = np.zeros([x.size, y.size])
    i = 1j
    # Expand the p vector into readable names
    var = np.reshape(p, [p.size // 5, 5])
    amp = var[:, 0]
    offset_x = var[:, 1]
    sigma_x = var[:, 2]
    offset_y = var[:, 3]
    sigma_y = var[:, 4]
    n_lobes = var[:, 0].size

    # If the parameters were built on a different grid, then put on new grid
    # Npix only affects the offset params
    if model_npix is not None:
        if model_npix < x.size:
            offset = np.abs(x.size / 2 - model_npix / 2)
            offset_x += offset
            offset_y += offset
        else:
            offset = np.abs(model_npix / 2 - x.size / 2)
            offset_x -= offset
            offset_y -= offset
    # Resolution affects gaussian sigma and offsets
    if model_res is not None:
        sigma_x *= model_res
        sigma_y *= model_res
        offset_x = ((offset_x - x.size / 2) * model_res) + x.size / 2
        offset_y = ((offset_y - y.size / 2) * model_res) + y.size / 2

    if not ftransform:
        for lobe in range(n_lobes):
            decomp_beam += amp[lobe] * np.outer(
                np.exp(-(y - offset_y[lobe]) ** 2 / (2 * sigma_y[lobe] ** 2)),
                np.exp(-(x - offset_x[lobe]) ** 2 / (2 * sigma_x[lobe] ** 2))
            )
        volume_beam = 0
        sq_volume_beam = 0
    else:
        # Full uv model with all the gaussian components
        decomp_beam = decomp_beam.astype(np.complex128)
        volume_beam = np.sum(amp)
        sq_volume_beam = np.pi * np.sum(sigma_x * sigma_y * amp ** 2) / (x.size * y.size)

        offset_x -= x.size / 2
        offset_y -= y.size / 2

        kx = np.outer(np.arange(x.size) - x.size / 2, np.ones(y.size))
        ky = np.outer(np.ones(x.size), np.arange(y.size) - y.size / 2)
        decomp_beam += amp ** 2 * \
            np.pi / (x.size * y.size) * \
            sigma_x * sigma_y * np.exp(
                (2 * np.pi ** 2 / (x.size * y.size) * sigma_x ** 2 * kx **2 + sigma_y ** 2 * ky ** 2) -  
                (2 * np.pi / x.size * 1j * (offset_x * kx + offset_y * ky))
            )

    return decomp_beam, volume_beam, sq_volume_beam

def beam_image(psf: dict | h5py.File, obs: dict, pol_i: int, freq_i: int | None = None, abs = False, square = False) -> np.ndarray:
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
        freq_norm = freq_norm[:]
        pix_horizon = pix_horizon[0]
    dimension = elements = obs['dimension']
    xl = dimension / 2 - psf_dim / 2 + 1
    xh = dimension / 2 - psf_dim / 2 + psf_dim
    yl = elements / 2 - psf_dim / 2 + 1
    yh = elements / 2 - psf_dim / 2 + psf_dim

    group_n, _, ri_id = histogram(group_id, min = 0)
    gi_use = np.nonzero(group_n)
    # Most likely going to be 1 as PyFHD does only one beam mostly
    n_groups = np.count_nonzero(group_n)

    if beam_gaussian_params is not None:
        # 1.3 is the padding factor for the gaussian fitting procedure
        # (2.*obs.kpix) is the ratio of full sky (2 in l,m) to the analysis range (1/obs.kpix)
        # (2.*obs.kpix*dimension/psf.pix_horizon) is the scale factor between the psf pixels-to-horizon and the 
        # analysis pixels-to-horizon 
        # (0.5/obs.kpix) is the resolution scaling of what the beam model was made at and the current res 
        model_npix = pix_horizon * 1.3
        model_res = (2 * obs['kpix'] * dimension) / pix_horizon * (0.5 / obs['kpix'])
    
    freq_bin_i = obs["baseline_info"]["fbin_i"]
    freq_i_use = np.nonzero(obs["baseline_info"]["freq_use"])[0]
    n_bin_use = 0
    # We assume freq_i is an int when provided (i.e. a single frequency index)
    if freq_i is not None:
        freq_i_use = freq_i

    if square:
        # Do note freq_i_use could be an integer or an array if freq_i is supplied or not
        beam_base = np.zeros([dimension, elements])
        freq_bin_use = freq_bin_i[freq_i_use]
        fbin_use = np.sort(np.unique(freq_bin_use))
        nbin = fbin_use.size

        if beam_gaussian_params is not None:
            beam_single = np.zeros([dimension, elements])
        else:
            beam_single = np.zeros([psf_dim, psf_dim], dtype = np.complex128)
        for bin_i in range(nbin):
            fbin = fbin_use[bin_i]
            nf_bin = np.count_nonzero(freq_bin_use == fbin)
            if beam_gaussian_params is not None:
                for gi in range(n_groups):    
                    # beam_gaussian_params needs to be copied here rather than
                    # a view as interestingly gaussian_decomp affects the values
                    # of the array used with the calculations done to var and no copies
                    # are made, only views are adjusted. so we explcitly call copy
                    params = beam_gaussian_params[pol_i, fbin, :].copy()
                    gaussian = gaussian_decomp(
                        np.arange(dimension), 
                        np.arange(elements),
                        params,
                        model_npix = model_npix,
                        model_res = model_res
                    )[0]
                    beam_single += gaussian * group_n[gi_use[gi]]                
                beam_single /= np.sum(group_n[gi_use])
                beam_base += nf_bin * beam_single ** 2  
            else:
                for gi in range(n_groups):
                    beam_single += (psf['beam_ptr'][0, fbin, rbin, rbin] * group_n[gi_use[gi]]).reshape([psf_dim, psf_dim])
                beam_single /= np.sum(group_n[gi_use])
                if abs:
                    beam_single = np.abs(beam_single)
                beam_base_uv1 = np.zeros([dimension, elements], np.complex128)
                beam_base_uv1[xl : xh + 1, yl : yh + 1] = beam_single
                beam_base_single = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(beam_base_uv1)))
                beam_base += nf_bin * (beam_base_single * np.conjugate(beam_base_single)).real
            n_bin_use += nf_bin * freq_norm[fbin]
    else:
        nf_use = freq_i_use.size
        if beam_gaussian_params is not None:
            beam_base_uv = np.zeros([dimension, elements])
            beam_single = np.zeros([dimension, elements])
        else:
            beam_base_uv = np.zeros([psf_dim, psf_dim], dtype = np.complex128)
            beam_single = np.zeros([psf_dim, psf_dim], dtype = np.complex128)
        for f_idx in range(nf_use):
            fi = freq_i_use[f_idx]
            if freq_i is not None:
                if freq_i != fi:
                    continue
            fbin=freq_bin_i[fi]
            beam_single[:, :] = 0
            if beam_gaussian_params is not None:
                for gi in range(n_groups):
                    params = beam_gaussian_params[pol_i, fbin, :].copy()
                    gaussian = gaussian_decomp(
                        np.arange(dimension), 
                        np.arange(elements),
                        params,
                        model_npix = model_npix,
                        model_res = model_res
                    )[0]
                    beam_single += gaussian * group_n[gi_use[gi]]
            else:
                for gi in range(n_groups):
                    beam_single += (psf['beam_ptr'][0, fbin, rbin, rbin] * group_n[gi_use[gi]]).reshape([psf_dim, psf_dim])
            beam_single /= np.sum(group_n[gi_use])
            beam_base_uv += beam_single
            n_bin_use += freq_norm[fbin]

        if beam_gaussian_params is None:
            beam_base_uv1 = np.zeros([dimension, elements], dtype = np.complex128)
            beam_base_uv1[xl : xh + 1, yl : yh + 1] = beam_base_uv
            beam_base = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(beam_base_uv1)))
        else:
            beam_base = beam_base_uv
    
    beam_base /= n_bin_use
    return beam_base.real



    
    
    