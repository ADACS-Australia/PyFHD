import numpy as np
from numpy.typing import NDArray
from logging import Logger

from ..pyfhd_tools.pyfhd_utils import weight_invert, spectral_window, rebin
from . import gridding_utils


def filter_uv_uniform(
    image_uv: NDArray[np.complex128],
    vis_count: NDArray[np.float64] | None,
    obs: dict | None = None,
    params: dict | None = None,
    pyfhd_config: dict | None = None,
    logger: Logger | None = None,
    weights: NDArray[np.float64] | None = None,
    fi_use: NDArray[np.integer] | None = None,
    bi_use: NDArray[np.integer] | None = None,
    mask_mirror_indices: bool = False,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    Perform uniform weighting in {u,v} space.

    Parameters
    ----------
    image_uv : NDArray[np.complex128]
        A 2D {u,v} gridded plane to be filtered
    vis_count : NDArray[np.float64] | None
        2D array of number of contributing visibilities per pixel on the {u,v} grid
    obs : dict | None, optional
        Observation metadata dictionary, by default None
    params : dict | None, optional
        Visibility metadata dictionary, by default None
    pyfhd_config : dict | None, optional
        Run option dictionary, by default None
    logger : Logger | None, optional
        pyfhd's logger, by default None
    weights : NDArray[np.float64] | None, optional
        The weights array (aka vis_weights), by default None
    fi_use : NDArray[np.integer] | None, optional
        Frequency index array for gridding, i.e. gridding all frequencies for continuum images, by default None
    bi_use : NDArray[np.integer] | None, optional
        Baseline index array for gridding, i.e even vs odd time stamps, by default None
    mask_mirror_indices : bool, optional
        Exclude baselines mirrored along the v-axis, by default False

    Returns
    -------
    image_uv_filtered : NDArray[np.complex128]
        The filtered 2D {u,v} plane
    filter_use : NDArray[np.float64]
        The filter used

    Raises
    ------
    TypeError
        In the case obs or params is None and vis_count is also None
    """

    # If you need the name, grab it from pyfhd_config where needed

    # This does not make use of fine-grained flagging, but relies on coarse flags from the obs structure
    # (i.e. a list of tiles completely flagged, and of frequencies completely flagged)
    if vis_count is None:
        if obs is not None or params is not None:
            vis_count = gridding_utils.visibility_count(
                obs,
                params,
                weights,
                pyfhd_config,
                logger,
                fi_use,
                bi_use,
                mask_mirror_indices,
            )
        elif weights is not None and np.size(weights) == np.size(image_uv):
            if np.max(weights) == 0:
                vis_count = weights
            else:
                vis_count = np.abs(weights) / np.min(np.abs(weights[weights > 0]))
        else:
            raise TypeError("obs and params must not be None when vis_count is None")

    # Get the parts of the filter we're using
    filter_use = weight_invert(vis_count, threshold=1)
    # Get the weights index as well
    if weights is not None and np.size(weights) == np.size(image_uv):
        wts_i = np.nonzero(weights)
    else:
        wts_i = np.nonzero(filter_use)
    # Apply a mean normalization
    if np.size(wts_i) > 0:
        filter_use /= np.mean(filter_use[wts_i])
    else:
        filter_use /= np.mean(filter_use)
    # Return the filtered
    return image_uv * filter_use, filter_use


def vis_delay_filter(
    vis_model_arr: NDArray[np.complex128], *, obs: dict, params: dict
) -> NDArray[np.complex128]:
    """
    Apply a delay space filter at the horizon to remove fft artifacts.

    Model visibilities are phased to zenith, windowed, transformed to delay space,
    cut at the horizon, transformed back to visibility space, unwindowed, unphased
    from zenith, and frequency cut to match the data.

    Parameters
    ----------
    vis_model_arr : ndarray of complex
        Input model visibilities to be filtered.
    obs : dict
        The observation metadata dictionary.
    params : dict
        Visibility metadata dictionary.

    Returns
    -------
    vis_model_arr : ndarray of complex
        Delay filtered visibilities with half the number of frequencies as the
        input visibilities.
    """
    # u,v,w are in light travel time in seconds
    freq_arr = obs["baseline_info"]["freq"]
    freq_res = obs["freq_res"]
    n_pol = obs["n_pol"]
    kbinsize = obs["kpix"]
    nfreq = freq_arr.size
    nbl = params["uu"].size

    data = vis_model_arr.transpose(1, 2, 0)

    # test with removing zeroed visibilities instead
    total_data = np.sum(np.abs(data), axis=(0, 1))
    bi_use = np.nonzero(total_data != 0)
    cross_inds = np.nonzero(params["antenna1"] != params["antenna2"])
    bi_use = np.intersect1d(bi_use, cross_inds)

    data = data[:, bi_use]
    uu = params["uu"][bi_use]
    vv = params["vv"][bi_use]
    ww = params["ww"][bi_use]
    bb = np.sqrt(uu**2.0 + vv**2.0 + ww**2.0)
    nbl = bi_use.size

    # Phase to zenith -- easier calculations of the location of the horizon when
    # phased to zenith
    dimension = obs["dimension"]
    dx = obs["obsx"] - obs["zenx"]
    dy = obs["obsy"] - obs["zeny"]
    dx *= 2.0 * np.pi / dimension
    dy *= 2.0 * np.pi / dimension
    phase = (
        np.outer(freq_arr, uu) * dx / kbinsize + np.outer(freq_arr, vv) * dy / kbinsize
    )
    rephase_vals = np.cos(phase) + 1j * np.sin(phase)
    rephase_vals = np.repeat((rephase_vals[:, :, np.newaxis]), n_pol, axis=2)

    data *= rephase_vals
    del uu, vv, ww

    # Apply window function
    window = spectral_window(nfreq, type="Blackman-Harris", periodic=True)
    norm_factor = np.sqrt(nfreq / np.sum(window**2.0))
    window = window * norm_factor
    window_expand = rebin(window, (nfreq, nbl, n_pol), sample=True)
    data = data * window_expand

    # FFT along freq axis
    spectra = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    del data, window

    # Cut at the horizon
    tau_cut = 1.0

    # Calculate upper and lower delay limits for each baseline
    lower_limit = nfreq * (0.5 - bb * tau_cut * freq_res)
    upper_limit = nfreq * (0.5 + bb * tau_cut * freq_res)

    for freq_i in range(nfreq):
        mask_high = freq_i / upper_limit
        mask_high_inds = np.nonzero(mask_high > 1.0)
        if mask_high_inds[0].size > 0:
            spectra[freq_i, mask_high_inds] = 0

        mask_low = freq_i / lower_limit
        mask_low_inds = np.nonzero(mask_low < 1.0)
        if mask_low_inds[0].size > 0:
            spectra[freq_i, mask_low_inds] = 0

    masked_data = np.fft.ifft(np.fft.fftshift(spectra, axes=0), axis=0)
    masked_data = masked_data / window_expand

    # Unphase from zenith and cut to the desired band
    masked_data *= 1.0 / rephase_vals
    masked_data = masked_data.transpose(2, 0, 1)
    freq_ind_min = nfreq // 4
    freq_ind_max = 3 * nfreq // 4
    vis_model_arr = vis_model_arr[:, freq_ind_min:freq_ind_max]
    vis_model_arr[:, :, bi_use] = masked_data[:, freq_ind_min:freq_ind_max]

    return vis_model_arr
