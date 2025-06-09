import numpy as np
from astropy.constants import c
from logging import Logger
from scipy.interpolate import interp1d
from scipy.io import readsav
from PyFHD.beam_setup.antenna import init_beam
from PyFHD.beam_setup.beam_utils import beam_power
from PyFHD.io.pyfhd_io import recarray_to_dict
from pathlib import Path
from PyFHD.io.pyfhd_io import save, load
from h5py import File
import sys


def create_psf(obs: dict, pyfhd_config: dict, logger: Logger) -> dict | File:
    """
    Creates the psf dictionary by loading in a `sav` or `HDF5` file from a FHD run.
    In the future it would be nice to have the ability to read in a beam fits file.
    Do note, that PyFHD was made with the assumption that the beam is the MWA beam,
    and assumes the beam does not differ on a per baseline basis. If you wish to use
    separate baselines you'll need to add that functionality yourself.

    Parameters
    ----------
    obs : dict
        The observation metadata dictionary
    pyfhd_config : dict
        The PyFHD configuration dictionary
    logger : Logger
        PyFHD's logger

    Raises
    ------
    ValueError
        If the beam file type is not recognized or if the beam file path is not set correctly.

    Returns
    -------
    dict | h5py.File
        _description_
    """
    if pyfhd_config["beam_file_path"] is None:
        # Form the beam from scratch using pyuvdata for the Jones Matrix
        # and translations from FHD for the antenna response.
        logger.info(
            "PyFHD will do the beam forming from scratch using pyuvdata and the antenna response from FHD."
            "Please note, gaussian decomp for MWA is not implemented yet."
        )
        antenna, psf = init_beam(obs, pyfhd_config, logger)
        # TODO: we'll see if the +1 is necessary, IDL indexing thing
        n_freq_bin = np.max(obs["baseline_info"]["fbin_i"]) + 1
        # TODO: Double check the shape
        beam_arr = np.zeros(
            [
                obs["n_pol"],
                n_freq_bin,
                psf["resolution"] + 1,
                psf["resolution"] + 1,
                psf["superres_dim"] * 2,
            ],
            dtype=np.complex128,
        )
        xvals_i, yvals_i = np.meshgrid(
            np.arange(psf["resolution"]), np.arange(psf["resolution"]), indexing="ij"
        )
        xvals_i *= psf["resolution"]
        yvals_i *= psf["resolution"]
        xvals_i = xvals_i.flatten()
        yvals_i = yvals_i.flatten()
        xvals_psf_dim, yvals_psf_dim = np.meshgrid(
            np.arange(psf["dim"]), np.arange(psf["dim"]), indexing="ij"
        )
        psf["xvals"] = np.zeros(
            [psf["resolution"], psf["resolution"], psf["dim"], psf["dim"]]
        )
        psf["yvals"] = np.zeros(
            [psf["resolution"], psf["resolution"], psf["dim"], psf["dim"]]
        )
        for i in range(psf["resolution"]):
            for j in range(psf["resolution"]):
                psf["xvals"][i, j, :, :] = (
                    xvals_psf_dim - psf["dim"] / 2 + i / psf["resolution"]
                )
                psf["yvals"][i, j, :, :] = (
                    yvals_psf_dim - psf["dim"] / 2 + j / psf["resolution"]
                )

        zen_int_x = (obs["zenx"] - obs["obsx"]) / psf["scale"] + psf["image_dim"] / 2
        zen_int_y = (obs["zeny"] - obs["obsy"]) / psf["scale"] + psf["image_dim"] / 2
        # Calculate the hyperresolved uv-vals of the beam kernel at highest precision prior to cast to
        # be accurate yet small
        res_super = 1 / (psf["resolution"] / psf["intermediate_res"])

        xvals_uv_superres, yvals_uv_superres = np.meshgrid(
            np.arange(psf["superres_dim"]),
            np.arange(psf["superres_dim"]),
        )
        xvals_uv_superres = (
            xvals_uv_superres * res_super
            - np.floor(psf["dim"] / 2) * psf["intermediate_res"]
            + np.floor(psf["dim"] / 2)
        )
        yvals_uv_superres = (
            yvals_uv_superres * res_super
            - np.floor(psf["dim"] / 2) * psf["intermediate_res"]
            + np.floor(psf["dim"] / 2)
        )

        freq_center = antenna["freq"][0]
        for pol_i in range(obs["n_pol"]):
            for freq_i in range(n_freq_bin):
                beam_int = 0
                beam_int_2 = 0
                # Calculate power beam from antenna beams
                psf_base_superres = beam_power(
                    antenna,
                    obs,
                    pol_i,
                    freq_i,
                    psf,
                    zen_int_x,
                    zen_int_y,
                    xvals_uv_superres,
                    yvals_uv_superres,
                    pyfhd_config,
                )

                # divide by psf_resolution^2 since the FFT is done at
                # a different resolution and requires a different normalization
                # TODO: add bi_use calcs for baseline_group_n
                # beam_int += baseline_group_n + np.sum(psf_base_superres) / psf["resolution"] ** 2
                # beam_int_2 += baseline_group_n + np.sum(np.abs(psf_base_superres)) / psf["resolution"] ** 2
                psf_single = np.zeros(
                    [psf["resolution"] + 1, psf["resolution"] + 1],
                    dtype=np.complex128,
                )

                for i in range(psf["resolution"]):
                    for j in range(psf["resolution"]):
                        psf_single[psf["resolution"] - i, psf["resolution"] - j] = (
                            psf_base_superres[xvals_i + i, yvals_i + j]
                        )
                # TODO: check the rolling (shifting) and reshaping done here
                for i in range(psf["resolution"]):
                    psf_single[psf["resolution"] - i, psf["resolution"]] = np.roll(
                        psf_base_superres[xvals_i + i, yvals_i + psf["resolution"]],
                        1,
                        0,
                    ).flatten()
                for j in range(psf["resolution"]):
                    psf_single[psf["resolution"], psf["resolution"] - j] = np.roll(
                        psf_base_superres[xvals_i + psf["resolution"], yvals_i + j],
                        1,
                        1,
                    ).flatten()
                psf_single[psf["resolution"], psf["resolution"]] = np.roll(
                    np.roll(
                        psf_base_superres[
                            xvals_i + psf["resolution"], yvals_i + psf["resolution"]
                        ],
                        1,
                        1,
                    ),
                    1,
                    0,
                ).flatten()

        return psf
    elif pyfhd_config["beam_file_path"].suffix == ".sav":
        # Read in a sav file containing the psf structure as we expect from FHD
        logger.info(
            "Reading in a beam sav file probably will take a long time. You will require double the storage size of the sav file in RAM at least. Do some other work or maybe watch your favourite long movie, for example the extended edition of LOTR: Return of the King is 4 hours 10 minutes. Check back when the Battle of the Pelennor Fields has finished or roughly 3 hours in."
        )
        beam = readsav(pyfhd_config["beam_file_path"], python_dict=True)
        psf = beam["psf"]
        # Delete the read in sav file, now that we got the psf, at this point we will have the psf size twice!
        del beam
        psf["beam_ptr"][0] = psf["beam_ptr"][0].T
        # Take only the first baseline (as it assumes every baseline points to the first i.e. the FFT is done per frequency)
        # Has a bonus of reducing memory use, unless NumPy is really good at using representations, maybe use double memory
        psf["beam_ptr"][0] = psf["beam_ptr"][0][:, :, 0]
        # Transpose the group array
        psf["id"][0] = psf["id"][0].T
        # Recarray to dict completely unpack object arrays into the dict, although will require the beam_ptr in memory twice potentially temporarily
        psf = recarray_to_dict(psf)
        # The to_chunk is a dictionary of dictionaries which contain the information necessary to chunk the beam_ptr
        to_chunk = {
            "beam_ptr": {
                "shape": psf["beam_ptr"].shape,
                "chunk": tuple([1] * 2 + list(psf["beam_ptr"].shape)[2:]),
            }
        }
        # By default save the file in the same place as the original beam
        output_path = Path(
            pyfhd_config["beam_file_path"].parent,
            pyfhd_config["beam_file_path"].stem + ".h5",
        )
        save(output_path, psf, "psf", logger=logger, to_chunk=to_chunk)
        # Since the psf is already in memory, return it
        return psf
    elif (
        pyfhd_config["beam_file_path"].suffix == ".h5"
        or pyfhd_config["beam_file_path"].suffix == ".hdf5"
    ):
        logger.info(f"Reading in the HDF5 file {pyfhd_config['beam_file_path']}")
        # If you selected to lazy load the beam, then psf will be a h5py File Object
        psf = load(
            pyfhd_config["beam_file_path"],
            logger=logger,
            lazy_load=pyfhd_config["lazy_load_beam"],
        )
        return psf
    elif pyfhd_config["beam_file_path"].suffix == ".fits":
        # Read in a fits file, when you do I assume you probably will be translating
        # FHD's beam setup while reading in a beam fits file.
        logger.error("The ability to read in a beam fits hasn't been implemented yet")
        sys.exit(1)
    raise ValueError(
        f"Unknown beam file type {pyfhd_config['beam_file_path'].suffix}. "
        "Please use a .sav, .h5, .hdf5"
        "If you meant for PyFHD to do the beam forming, please set the beam_file_path to None (~ in YAML)."
    )
