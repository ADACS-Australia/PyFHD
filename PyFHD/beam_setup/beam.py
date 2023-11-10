import numpy as np
from astropy.constants import c
from logging import RootLogger
from scipy.interpolate import interp1d
from PyFHD.beam_setup.beam_utils import mwa_beam_setup_init
from scipy.io import readsav
from PyFHD.io.pyfhd_io import recarray_to_dict
from pathlib import Path
from PyFHD.io.pyfhd_io import save, load
from h5py import File
import sys

def create_antenna(pyfhd_config : dict, obs : dict) -> dict:
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

    # Setup the constants and variables
    n_tiles = obs['n_tile']
    n_freq = obs['n_freq']
    n_pol = obs['n_pol']
    # Almost all instruments have two instrumental polarizations (either linear or circular)
    n_ant_pol = 2
    obsra = obs['obsra']
    obsdec = obs['obsdec']
    zenra = obs['zenra']
    zendec = obs['zendec']
    obsx = obs['obsx']
    obsy = obs['obsy']
    dimension = obs['dimension']
    elements = obs['elements']
    kbinsize = obs['kpix']
    degpix = obs['degpix']
    astr = obs['astr']
    psf_image_resolution = 10
    frequency_array = obs['baseline_info']['freq']
    freq_bin_i = obs['baseline_info']['fbin_i']
    nfreq_bin = int(np.max(freq_bin_i)) + 1
    tile_a = obs['baseline_info']['tile_a']
    tile_b = obs['baseline_info']['tile_b']
    ant_names = np.unique(tile_a[: obs['n_baselines']])
    if pyfhd_config['beam_offset_time'] is not None:
        jdate_use = obs['jd0'] + pyfhd_config['beam_offset_time'] / 24 / 3600
    else:
        jdate_use = obs['jd0']
    if psf['resolution'] is None:
        psf_resolution = 16
    
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
        'n_pol' : n_ant_pol,
        'antenna_type' : pyfhd_config['instrument'],
        'names' : ant_names,
        'beam_model_version' : pyfhd_config['beam_model_version'],
        'freq' : freq_center,
        'nfreq_bin' : nfreq_bin,
        'n_ant_elements' : 0,
        # Anything that was pointer arrays in IDL will be None until assigned in Python 
        'jones' : None,
        'coupling' : None,
        'gain' : None,
        'coords' : None,
        'delays' : None,
        'size_meters' : 0,
        'height' : 0,
        'response' : None,
        'group_id' : np.full(n_ant_pol, -1, dtype = np.int64),
        'pix_window' : None,
        'pix_use' : None,
        'psf_image_dim' : 0,
        'psf_scale' : 0
    }

    # We are building PyFHD with only MWA in mind at the moment
    antenna = mwa_beam_setup_init(pyfhd_config, obs, antenna)

    return antenna

def create_psf(pyfhd_config: dict, logger: RootLogger) -> dict | File:
    """
    Creates the psf dictionary by loading in a `sav` or `HDF5` file

    Parameters
    ----------
    pyfhd_config : dict
        _description_
    logger : RootLogger
        _description_

    Returns
    -------
    dict | h5py.File
        _description_
    """
    if pyfhd_config["beam_file_path"].suffix == '.sav':
        # Read in a sav file containing the psf structure as we expect from FHD
        logger.info("Reading in a beam sav file probably will take a long time. You will require double the storage size of the sav file in RAM at least. Do some other work or maybe watch your favourite long movie, for example the extended edition of LOTR: Return of the King is 4 hours 10 minutes. Check back when the Battle of the Pelennor Fields has finished or roughly 3 hours in.")
        beam = readsav(pyfhd_config["beam_file_path"], python_dict=True)
        psf = beam['psf']
        # Delete the read in sav file, now that we got the psf, at this point we will have the psf size twice!
        del beam
        psf['beam_ptr'][0] = psf['beam_ptr'][0].T
        # Take only the first baseline (as it assumes every baseline points to the first i.e. the FFT is done per frequency)
        # Has a bonus of reducing memory use, unless NumPy is really good at using representations, maybe use double memory
        psf['beam_ptr'][0] = psf['beam_ptr'][0][:,:,0]
        # Recarray to dict completely unpack object arrays into the dict, although will require the beam_ptr in memory twice potentially temporarily
        psf = recarray_to_dict(psf)
        # The to_chunk is a dictionary of dictionaries which contain the information necessary to chunk the beam_ptr
        to_chunk = {
            "beam_ptr": {
                "shape": psf['beam_ptr'].shape,
                "chunk": tuple([1] * 2 + list(psf['beam_ptr'].shape)[2:])
            }
        }
        # By default save the file in the same place as the original beam
        output_path = Path(pyfhd_config["beam_file_path"].parent ,pyfhd_config["beam_file_path"].stem + ".h5")
        save(output_path, psf, "psf", logger = logger, to_chunk = to_chunk)
        # Since the psf is already in memory, return it
        return psf
    elif pyfhd_config["beam_file_path"].suffix == ".h5" or pyfhd_config["beam_file_path"].suffix == ".hdf5":
        logger.info(f"Reading in the HDF5 file {pyfhd_config['beam_file_path']}")
        # If you selected to lazy load the beam, then psf will be a h5py File Object
        psf = load(pyfhd_config['beam_file_path'], logger = logger, lazy_load = pyfhd_config["lazy_load_beam"])
        return psf
    elif pyfhd_config["beam_file_path"].suffix == '.fits':
        # Read in a fits file, when you do I assume you probably will be translating
        # FHD's beam setup while reading in a beam fits file.
        logger.error("The ability to read in a beam fits hasn't been implemented yet")
        sys.exit(1)

