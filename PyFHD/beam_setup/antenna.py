import numpy as np
from numpy.typing import NDArray
from logging import Logger
from astropy.constants import c
from astropy.coordinates import EarthLocation
from scipy.interpolate import interp1d
from PyFHD.beam_setup.mwa import dipole_mutual_coupling
from PyFHD.pyfhd_tools.unit_conv import pixel_to_radec, radec_to_altaz


def init_beam(obs: dict, pyfhd_config: dict, logger: Logger) -> dict:
    """
    TODO: _summary_

    Parameters
    ----------
    obs : dict
        _description_
    pyfhd_config : dict
        _description_
    logger : Logger
        _description_

    Returns
    -------
    dict
        _description_
    """

    #     # Setup the constants and variables
    n_tiles = obs["n_tile"]
    # Almost all instruments have two instrumental polarizations (either linear or circular)
    n_ant_pol = 2
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

    antenna_size = {
        "mwa": 5,
        "hera": 14,
    }

    if pyfhd_config["instrument"] == "mwa":
        # Get the antenna coordinates
        n_dipoles = 16
        antenna_spacing = 1.1
        xc_arr, yc_arr = np.meshgrid(np.arange(4), np.arange(4))
        xc_arr = xc_arr.flatten() * antenna_spacing
        yc_arr = np.flipud(yc_arr).flatten() * antenna_spacing
        zc_arr = np.zeros(n_dipoles)
        coords = np.array([xc_arr, yc_arr, zc_arr])
        # Get the delays
        delays = obs["delays"] * 4.35e-10
        if pyfhd_config["dipole_mutual_coupling_factor"]:
            coupling = dipole_mutual_coupling(freq_center)
        else:
            coupling = np.tile(
                np.identity(n_dipoles), (n_ant_pol, freq_center.size, 1, 1)
            )

    else:
        n_dipoles = 1
        coords = np.zeros((3, n_dipoles))
        delays = np.zeros(n_dipoles)
        coupling = np.tile(np.identity(n_dipoles), (n_ant_pol, freq_center.size, 1, 1))

    # Create basic antenna dictionary
    antenna = {
        "n_pol": n_ant_pol,
        "antenna_type": pyfhd_config["instrument"],
        "size_meters": (
            antenna_size[pyfhd_config["instrument"]]
            if pyfhd_config["instrument"] in antenna_size
            else 10
        ),
        "names": ant_names,
        "beam_model_version": pyfhd_config["beam_model_version"],
        "freq": freq_center,
        "nfreq_bin": nfreq_bin,
        "n_ant_elements": 0,
        # Anything that was pointer arrays in IDL will be None until assigned in Python
        "jones": None,
        "coupling": coupling,
        "gain": np.ones([n_ant_pol, freq_center.size, n_dipoles], dtype=np.float64),
        "coords": coords,
        "delays": delays,
        "response": None,
        "group_id": np.full(n_ant_pol, -1, dtype=np.int64),
        "pix_window": None,
        "pix_use": None,
    }

    # Create the initial psf dict
    psf = {
        "dim": (
            pyfhd_config["psf_dim"]
            if pyfhd_config["psf_dim"]
            else np.ceil(
                (
                    antenna["size_meters"]
                    * 2
                    * np.max(obs["baseline_info"]["freq"])
                    / c.value
                )
                / obs["kpix"]
            )
        ),
        "resolution": pyfhd_config["psf_resolution"],
        # This is more of a placeholder, if we want PyFHD to support processing more than one instrument at a time we'll need to edit this to be calculated rather than hardcoded.
        "id": np.zeros(
            [obs["n_pol"], obs["n_freq"], obs["n_baselines"]], dtype=np.int64
        ),
        "beam_mask_threshold": pyfhd_config["beam_mask_threshold"],
        "freq_norm": np.ones(obs["n_freq"], dtype=np.int64),
        "image_resolution": 10,  # Add psf_image_resolution to the config file
        "fbin_i": obs["baseline_info"]["fbin_i"],
    }
    psf["intermediate_res"] = np.min(
        [np.ceil(np.sqrt(psf["resolution"]) / 2) * 2, psf["resolution"]]
    )
    psf["image_dim"] = int(
        psf["dim"] * psf["image_resolution"] * psf["intermediate_res"]
    )
    psf["scale"] = obs["dimension"] * psf["intermediate_res"] / psf["image_dim"]
    psf["pix_horizon"] = obs["dimension"] / psf["scale"]

    # We'll need chunk the data to get the to fit more into memory
    memory_threshold = pyfhd_config["memory_threshold"]  # default 1 GB
    required_bytes = 8 * (psf["image_dim"] ** 2)
    mem_iter = np.ceil(required_bytes / memory_threshold)
    chunk_size = int(np.ceil(psf["image_dim"] / mem_iter))
    location = EarthLocation.of_site(obs["instrument"])

    # Get the zenith angle and azimuth angle arrays
    xvals_celestial, yvals_celestial = np.meshgrid(
        np.arange(chunk_size),
        np.arange(chunk_size),
    )
    for chunk_i in range(mem_iter):
        xvals_celestial = (
            (xvals_celestial + chunk_i * chunk_size) * psf["scale"]
            - psf["image_dim"] * psf["scale"] / 2
            + obs["obsx"]
        )
        yvals_celestial = (
            (yvals_celestial + chunk_i * chunk_size) * psf["scale"]
            - psf["image_dim"] * psf["scale"] / 2
            + obs["obsy"]
        )
        ra_arr, dec_arr = pixel_to_radec(xvals_celestial, yvals_celestial, obs["astr"])
        del xvals_celestial, yvals_celestial
        valid_i = np.where(np.isfinite(ra_arr))
        ra_arr = ra_arr[valid_i]
        dec_arr = dec_arr[valid_i]
        alt_arr, az_arr = radec_to_altaz(
            ra_arr.value,
            dec_arr.value,
            location.lat.value,
            location.lon.value,
            location.height.value,
            jdate_use,
        )
        # Initialize the zenith angle array in degrees
        za_arr = np.full([psf["image_dim"], psf["image_dim"]], 90)
        za_arr[valid_i] = 90 - alt_arr.value
        # Initialize the azimuth angle array in degrees
        az_arr = np.zeros([psf["image_dim"], psf["image_dim"]])
        az_arr[valid_i] = az_arr.value

    # Get the jones matrix for the antenna

    # Get the antenna response
    antenna["response"] = general_antenna_response(
        obs,
        antenna,
        za_arr=np.zeros((n_tiles, nfreq_bin)),
        az_arr=np.zeros((n_tiles, nfreq_bin)),
    )

    return antenna, psf


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

    # FHD assumes you might be dealing with more than one antenna, hence the groupings it used.
    # PyFHD currently only supports one antenna, so we can ignore the groupings.
    for pol_i in range(antenna["n_pol"]):
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
                    antenna_gain_arr * measured_current[ii] / antenna["n_ant_elements"]
                )

    return response
