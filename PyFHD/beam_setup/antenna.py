import numpy as np
from numpy.typing import NDArray
from logging import Logger
from astropy.constants import c
from scipy.interpolate import interp1d

from PyFHD.pyfhd_tools.pyfhd_utils import histogram


def create_antenna(pyfhd_config: dict, obs: dict) -> dict:
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

    #     # Setup the constants and variables
    n_tiles = obs["n_tile"]
    n_freq = obs["n_freq"]
    n_pol = obs["n_pol"]
    # Almost all instruments have two instrumental polarizations (either linear or circular)
    n_ant_pol = 2
    obsra = obs["obsra"]
    obsdec = obs["obsdec"]
    zenra = obs["zenra"]
    zendec = obs["zendec"]
    obsx = obs["obsx"]
    obsy = obs["obsy"]
    dimension = obs["dimension"]
    elements = obs["elements"]
    kbinsize = obs["kpix"]
    degpix = obs["degpix"]
    astr = obs["astr"]
    psf_image_resolution = 10
    frequency_array = obs["baseline_info"]["freq"]
    freq_bin_i = obs["baseline_info"]["fbin_i"]
    nfreq_bin = int(np.max(freq_bin_i)) + 1
    tile_a = obs["baseline_info"]["tile_a"]
    tile_b = obs["baseline_info"]["tile_b"]
    ant_names = np.unique(tile_a[: obs["n_baselines"]])
    if pyfhd_config["beam_offset_time"] is not None:
        jdate_use = obs["jd0"] + pyfhd_config["beam_offset_time"] / 24 / 3600
    else:
        jdate_use = obs["jd0"]

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
        "n_pol": n_ant_pol,
        "antenna_type": pyfhd_config["instrument"],
        "names": ant_names,
        "beam_model_version": pyfhd_config["beam_model_version"],
        "freq": freq_center,
        "nfreq_bin": nfreq_bin,
        "n_ant_elements": 0,
        # Anything that was pointer arrays in IDL will be None until assigned in Python
        "jones": None,
        "coupling": None,
        "gain": None,
        "coords": None,
        "delays": None,
        "size_meters": 0,
        "height": 0,
        "response": None,
        "group_id": np.full(n_ant_pol, -1, dtype=np.int64),
        "pix_window": None,
        "pix_use": None,
        "psf_image_dim": 0,
        "psf_scale": 0,
    }

    # Get the jones matrix for the antenna

    # Get the antenna response
    antenna["response"] = general_antenna_response(
        obs,
        antenna,
        za_arr=np.zeros((n_tiles, nfreq_bin)),
        az_arr=np.zeros((n_tiles, nfreq_bin)),
    )

    return antenna


def general_antenna_response(
    obs: dict,
    antenna: dict,
    za_arr: NDArray[np.floating],
    az_arr: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """
    TODO: summary
    Parameters
    ----------
    obs : dict
        Observation metadata dictionary
    antenna : dict
        Antenna metadata dictionary
    za_arr : NDArray[np.floating]
        Zenith angle array in radians
    az_arr : NDArray[np.floating]
        Azimuth angle array in radians

    Returns
    -------
    response
        Antenna response
    """
    light_speed = c.value
    """
        Given that in FHD the antenna response is a pointer array of shape (antenna["n_pol", obs["n_tile"])
        where each pointer is an array of pointers of shape (antenna["n_freq_bin"]). Each pointer in the array
        of shape (antenna["n_freq_bin"]) points to a complex array of shape (antenna["pix_use"].size,).

        Furthermore, when the antenna response is calculated, it looks like this is done on a per frequency bin
        basis and each tile will point to the same antenna response for that frequency bin. This means we can ignore
        the tile dimension and just calculate the antenna response for each frequency bin and polarization to save
        memory in Python.
    """

    response = np.zeros(
        [antenna["n_pol"], antenna["nfreq_bin"], antenna["pix_use"].size],
        dtype=np.complex128,
    )

    # Calculate projections only at locations of non-zero pixels
    proj_east_use = np.sin(za_arr[antenna["pix_use"]]) * np.sin(
        az_arr[antenna["pix_use"]]
    )
    proj_north_use = np.sin(za_arr[antenna["pix_use"]]) * np.cos(
        az_arr[antenna["pix_use"]]
    )
    proj_z_use = np.cos(za_arr[antenna["pix_use"]])

    for pol_i in range(antenna["n_pol"]):
        g_hist, _, g_ri = histogram(antenna["group_id"][pol_i], min=0)
        for group_i in range(g_hist.size):
            ng = g_hist[group_i]
            if ng == 0:
                continue
            g_inds = g_ri[g_ri[group_i] : g_ri[group_i + 1]]
            ref_i = g_inds[0]

            # Phase of each dipole for the source (relative to the beamformer settings)
            D_d = (
                np.outer(antenna["coords"][0], proj_east_use)
                + np.outer(antenna["coords"][1], proj_north_use)
                + np.outer(antenna["coords"][2], proj_z_use)
            )

            for freq_i in range(antenna["nfreq_bin"]):
                Kconv = 2 * np.pi * antenna["freq"][freq_i] / light_speed
                voltage_delay = np.exp(
                    1j
                    * 2
                    * np.pi
                    * antenna["delays"]
                    * antenna["freq"][freq_i]
                    * antenna["gain"][pol_i, freq_i]
                )
                # TODO: Check if it's actually outer, although it does look like voltage_delay is likely 1D
                measured_current = np.outer(
                    voltage_delay, antenna["coupling"][pol_i, freq_i]
                )
                zenith_norm = np.outer(
                    np.ones(antenna["n_ant_elements"]),
                    antenna["coupling"][pol_i, freq_i],
                )
                measured_current /= zenith_norm

                # TODO: This loop can probably be vectorized
                for ii in range(antenna["n_ant_elements"]):
                    # TODO: check the way D_d needs to be indexed
                    antenna_gain_arr = np.exp(-1j * Kconv * D_d[ii, :])
                    response[pol_i, freq_i] += (
                        antenna_gain_arr
                        * measured_current[ii]
                        / antenna["n_ant_elements"]
                    )

    return response
