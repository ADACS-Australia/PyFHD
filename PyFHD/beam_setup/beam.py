import numpy as np
from astropy.constants import c
from logging import Logger
from scipy.interpolate import interp1d
from scipy.io import readsav
from PyFHD.io.pyfhd_io import recarray_to_dict
from pathlib import Path
from PyFHD.io.pyfhd_io import save, load
from h5py import File
import sys


def create_psf(pyfhd_config: dict, logger: Logger) -> dict | File:
    """
    Creates the psf dictionary by loading in a `sav` or `HDF5` file from a FHD run.
    In the future it would be nice to have the ability to read in a beam fits file.
    Do note, that PyFHD was made with the assumption that the beam is the MWA beam,
    and assumes the beam does not differ on a per baseline basis. If you wish to use
    separate baselines you'll need to add that functionality yourself.

    Parameters
    ----------
    pyfhd_config : dict
        _description_
    logger : Logger
        _description_

    Returns
    -------
    dict | h5py.File
        _description_
    """
    if pyfhd_config["beam_file_path"].suffix == ".sav":
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
    return None
