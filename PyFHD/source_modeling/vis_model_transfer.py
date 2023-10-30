import numpy as np
from PyFHD.data_setup.uvfits import extract_visibilities, create_params, extract_header
import logging
from PyFHD.pyfhd_tools.pyfhd_utils import run_command
from PyFHD.io.pyfhd_io import recarray_to_dict
import importlib_resources
import os
import shutil
import h5py
from scipy.io import readsav
from pathlib import Path

def vis_model_transfer(pyfhd_config: dict, obs : dict, logger: logging.RootLogger) -> tuple[np.array, dict]:
    """_summary_

    Parameters
    ----------
    pyfhd_config : dict
        _description_
    obs : dict
        _description_

    Returns
    -------
    tuple[np.array, dict]
        _description_
    """
    if (pyfhd_config['model_file_type'] == 'sav'):
        return import_vis_model_from_sav(pyfhd_config, obs, logger)
    elif (pyfhd_config['model_file_type'] == 'uvfits'):
        return import_vis_model_from_uvfits(pyfhd_config, obs, logger)
    

def import_vis_model_from_sav(pyfhd_config : dict, obs : dict, logger : logging.RootLogger) -> tuple[np.ndarray, dict]:
    """Read a model visibility array in from multiple IDL sav files which are in a directory
    given by pyfhd_config['model-file-path']. The data is assumed to be in the format of
    <obs_id>_params.sav, <obs_id>_vis_model_<pol_name>.sav. The pol_name follows the pol_names
    in the obs dictionary which are ['XX','YY','XY','YX','I','Q','U','V'].

    Parameters
    ----------
    pyfhd_config : dict
        The PyFHD config dictionary
    obs : dict
        The dictionary containing observation data and metadata
    logger : logging.RootLogger
        PyFHD's logger

    Returns
    -------
    tuple[vis_model_arr: np.ndarray, params_model: dict]
       A tuple containing the vis_model_arr and params_model (which is used for flagging)
    """
    try: 
        path = Path(pyfhd_config['model_file_path'], f"{pyfhd_config['obs_id']}_params.sav")
        params_model = readsav(path)
        params_model = recarray_to_dict(params_model.params)
        # Read in the first polarization from pol_names
        pol_i = 0
        path = Path(pyfhd_config['model_file_path'], f"{pyfhd_config['obs_id']}_vis_model_{obs['pol_names'][pol_i]}.sav")
        curr_vis_model = readsav(path)
        curr_vis_model = curr_vis_model.vis_model_ptr.transpose().astype(np.complex128)
        # The shape should be n_pol, n_freq, n_time * n_baselines
        vis_model_shape = [obs['n_pol']] + list(curr_vis_model.shape)
        vis_model_arr = np.empty(vis_model_shape, dtype=np.complex128)
        vis_model_arr[pol_i] = curr_vis_model
        for pol_i in range(1, obs['n_pol']):
            path = Path(pyfhd_config['model_file_path'], f"{pyfhd_config['obs_id']}_vis_model_{obs['pol_names'][pol_i]}.sav")
            curr_vis_model = readsav(path)
            # Should be a rec array containing one item vis_model_ptr
            curr_vis_model = curr_vis_model.vis_model_ptr
            vis_model_arr[pol_i] = curr_vis_model.transpose().astype(np.complex128)
    except FileNotFoundError as e:
            logger.error(f"PyFHD failed to load in the model visibilities while trying to transfer in file: {path} as the file wasn't found. PyFHD is exiting execution")
            exit()
    return vis_model_arr, params_model



def import_vis_model_from_uvfits(pyfhd_config : dict, obs : dict,
                                 logger : logging.RootLogger) -> tuple[np.ndarray, dict]:
    """Read a model visibility array in from a `uvfits` with filepath given
    by pyfhd_config['model_file_path']. Reads data in via 
    `PyFHD.data_setup.uvfits import extract_visibilities`.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    obs : dict
        The observation dictionary as populated by `PyFHD.data_setup.obs.create_obs`

    Returns
    -------
    vis_model_arr : np.array
        A `np.complex128` type array of shape (n_pol, n_vis_raw, n_freq)
    """
    
    #TODO WORRY about which order XX and YY are
    #TODO WORRY about order of baselines comparing WODEN sims and real data
    #TODO WORRY about weights
    
    header_model, params_data_model, _, _ = extract_header(pyfhd_config, logger, model_uvfits=True)

    if header_model['n_freq'] != obs['n_freq']:
        model_path = pyfhd_config['model_file_path']
        logger.error(f"The obs was expecting {obs['n_freq']} frequencies, "
                     f"but the model visibilities read in from {model_path} "
                     f"contain {header_model['n_freq']} freqs. Please supply "
                     "model visibilities that match the data. Exiting now.")
        exit()

    params_model = create_params(header_model, params_data_model, logger)
        
    vis_model_arr, _ = extract_visibilities(header_model, params_data_model,
                                            pyfhd_config, logger)
    
    return vis_model_arr, params_model

class _FlaggingInfoCounter(object):
    """Something to count and hold numbers to do with baselines"""

    def __init__(self, params : dict):
        """
        Given a populated `params` dict (as populated by
        `PyFHD.data_setup.uvfits.create_params`), calculate many useful quantities
        to do with antenna (tile) names and numbers, expected number of cross and
        auto correlations etc. 
        """

        self.unique_times = np.unique(params['time'])
        self.num_times = len(self.unique_times)

        ant_names1 = np.unique(params['antenna1'])
        ant_names2 = np.unique(params['antenna2'])

        self.num_visis = len(params['antenna1'])

        #If there are no auto-correlations, you don't get every unique tile
        #in either antenna1 or antenna2, so do a unique on both of them to be sure
        self.ant_names = np.unique(np.append(ant_names1, ant_names2))
        self.num_ants = len(self.ant_names)

        #indexes of the auto-correlations
        self.auto_locs = params['antenna1'] == params['antenna2']
        self.num_autos = np.count_nonzero(self.auto_locs)

        if self.num_autos == 0:
            self.num_autos_per_time = 0
        else:
            self.num_autos_per_time = self.num_ants
            
        #indexes of the cross-correlations
        self.cross_locs = params['antenna1'] != params['antenna2']
        
        #how many cross-correlations there should be per time step
        self.num_cross_per_time = int((self.num_ants*(self.num_ants - 1)) / 2)

        #number of visibilities per time step
        self.num_visi_per_time_step = self.num_cross_per_time + self.num_autos_per_time

        #within a single time step, where the cross-correlations are indexed
        #can use this while iterating over time to select the crosses only
        self.cross_locs_per_time = np.where(self.cross_locs[:self.num_visi_per_time_step])[0]

        #within a single time step, where the cross-correlations are indexed
        #can use this while iterating over time to select the crosses only
        self.auto_locs_per_time = np.where(self.auto_locs[:self.num_visi_per_time_step])[0]

        self.ant1_single_time = params['antenna1'][:self.num_visi_per_time_step]
        self.ant2_single_time = params['antenna2'][:self.num_visi_per_time_step]
    


def flag_model_visibilities(vis_model_arr : np.ndarray,
                            params_data : dict, params_model : dict,
                            obs : dict, pyfhd_config : dict,
                            logger : logging.RootLogger) -> np.ndarray:
    """Account for time offset and tile flags, and check that the uvfits
    and compatible. Needs to check if auto-correlations are present

    Parameters
    ----------
    vis_model_arr : np.ndarray
        The model visibility array from the model uvfits file
    params_data : dict
        The params data from the observation uvfits
    params_model : dict
        The params data from the model uvfits
    obs : dict
        The observaton dictionary containing all the data from the observation uvfits file
    pyfhd_config : dict
        The PyFHD configuration dictionary
    logger : logging.RootLogger
        The PyFHD logger

    Returns
    -------
    vis_model_arr_flagged: np.ndarray
        The flagged model visibility array
    """    

    # TODO - check if auto-correlations are present. If not, turn off
    # auto-correlation calibration options

    # Calculate a number of things we'll need to compare the data to the model
    flaginfo_data = _FlaggingInfoCounter(params_data)
    flaginfo_model = _FlaggingInfoCounter(params_model)

    # For all time steps in both data and model, check the minimum time offset
    # between the two. If there is a constant minimum > 0, the times might
    # be consistently offset because of QUACK time
    time_offsets = []
    for time in flaginfo_data.unique_times:
        #convert from julian to seconds via 24*60*60
        time_offsets.append(np.min(np.abs(flaginfo_model.unique_times - time))*(24.0*60*60))

    # The highest time resolution data we'll be dealing with. If the offset is
    # less than half of this, could be error in calculations rather than an
    # actual time offset
    min_integration = 0.5

    # Try to ascertain is the model is offset in time from the data
    # Only believe an offset to an accuracy of 0.25, so round to two decimal
    # places. Here we are essentially trying to divine the QUACK time
    rounded_offset = np.round(np.mean(time_offsets), 2)

    # If offset is big enough to be real, but smaller than half an integration
    # that add the `rounded_offset` (our calculated QUACK time) to the model
    # to work out which time steps are closest between model and data
    if rounded_offset >= min_integration / 2.0 and rounded_offset <= obs['time_res'] / 2.0:
        flaginfo_model.unique_times += rounded_offset / (24.0*60*60)

    logger.warning(f"Model time stamps are offset from data by an average of {rounded_offset}. Accounting for this to match model time steps to data")

    model_times_to_use = []
    for time in flaginfo_data.unique_times:
        t_ind = np.argmin(np.abs(flaginfo_model.unique_times - time))
        model_times_to_use.append(t_ind)

    model_times_to_use = np.array(model_times_to_use)

    # This means one or more time steps in the model match best to the same
    # time step in the data. This shouldn't happen if the data and model
    # have the same time resolution so something bad has happened
    if flaginfo_data.num_times != len(np.unique(model_times_to_use)):

        data_path = pyfhd_config['input_path'], pyfhd_config['obs_id'] + '.uvfits'
        model_path = pyfhd_config['import_model_uvfits']
        logger.error(f"Could not match the time steps in the data uvfits: {data_path}"
                     f" and model uvfits {model_path}. Please check the model "
                     "and try again. Exiting now.")
        exit()

    # Now to flag the model - some models have no flagged tiles (antennas),
    # whereas the data might have flagged tiles (and so missing baselines). 
    # This means we need to flag the missing tiles out of the data and 
    # reshape the model

    # If less antennas in the model than the data, we can't calibrate the whole
    # dataset so just error for now
    if flaginfo_model.num_ants < flaginfo_data.num_ants:
        model_path = pyfhd_config['import_model_uvfits']
        logger.error(f"There are less antennas (tiles) in the model uvfits "
                     f"{model_path} than in the data, so cannot calibrate the "
                     "whole dataset. Please check the model "
                     "and try again. Exiting now.")
        exit()

    # Test to see if there are auto-correlations in data
    # If they are in the data, but not the model, we need to reshape the
    # model to include empty autocorrelations in the correct spots

    include_autos = True
    if flaginfo_model.num_autos == 0:
        logger.warning("There are no auto-correlations present in model uvfits; "
                       "filling with zeros to match data shape, and switching "
                       "off all auto-correlation calibration")
        include_autos = False

    elif flaginfo_model.num_autos < flaginfo_model.num_times*flaginfo_model.num_ants:
        logger.warning("There are some auto-correlations present in model uvfits, "
                     "but less than number of antennas times number of time steps. "
                     "Cannot deal with missing autocorrelations so setting all "
                     "autos to zero and switching off all auto-correlation calibration")
        include_autos = False

    # This is where the fun begins - pyuvdata uses the tile number as written
    # in the 'TILE' column in the metafits file to encode baselines (and so
    # populate params[antenna1] and params[antenna2]). The numbers are MWA
    # assigned, and have nothing to do with antenna index. Birli and WODEN use
    # the tile index (one-indexed as the BASELINE encoding is one-indexed).
    # So to match a flagged tile from the data to the model, need to work out
    # the index of any flagged tiles in the data. All tile names are read in
    # from the metafits

    flag_indexes = []


    # if the data just have tile indexes, the maximum should be the number
    # of tiles - use that to work out if we have Birli of pyuvdata input

    # we should have a pyuvdata input in this case as a tile name is greater
    # than the number of tiles
    if np.max(flaginfo_data.ant_names) > len(obs['baseline_info']['tile_names']):

        # Loop over all possible antenna (tile) names, and if they're not in the
        # list of antennas in this data set, append to flag_indexes
        for ant_ind, ant_name in enumerate(obs['baseline_info']['tile_names']):
            if ant_name not in flaginfo_data.ant_names: flag_indexes.append(ant_ind + 1)

    # tiles are named by their index (1 indexed as per uvfits standard)
    else:
        for ant_name in range(1, len(obs['baseline_info']['tile_names'])+1):
            if ant_name not in flaginfo_data.ant_names: flag_indexes.append(ant_name)

    if len(flag_indexes) > 0:
        logger.info(f"Found flagged tiles {flag_indexes} in the data, flagging from the model")

    # This gives us true/false if the visibilities should be included
    # for a single time step based on antenna1 and antenna2
    include_per_time_ant1 = np.in1d(flaginfo_model.ant1_single_time,
                                          flag_indexes, invert=True)
    include_per_time_ant2 = np.in1d(flaginfo_model.ant2_single_time,
                                          flag_indexes, invert=True)

    # Doing a logic union combines info from ant1 and ant2
    include_per_time = include_per_time_ant1 & include_per_time_ant2

    # To get indexes of cross-correlations to use, ensure ant1 != ant2
    # These are now indexes of unflagged cross-correlations to grab from model
    include_cross_per_time = np.where(flaginfo_model.ant1_single_time[include_per_time] != flaginfo_model.ant2_single_time[include_per_time])[0]

    # To get indexes of auto-correlations to use, ensure ant1 == ant2
    # These are now indexes of unflagged auto-correlations to grab from model
    include_auto_per_time = np.where(flaginfo_model.ant1_single_time[include_per_time] == flaginfo_model.ant2_single_time[include_per_time])[0]

    # empty holder for the flagged model - this should be the same shape
    # as the data TODO check is same shape
    vis_model_arr_flagged = np.zeros((obs['n_pol'], obs['n_freq'],
                                      flaginfo_data.num_visis),
                                      dtype=complex)

    # For each time step that matches the data, copy across any visibilities
    # that aren't to be flagged
    for t_data_ind, t_model_ind in enumerate(model_times_to_use):

        # Subset of cross-corrs from flagged model to select for this time step
        t_flag_inds = t_data_ind*flaginfo_data.num_visi_per_time_step + flaginfo_data.cross_locs_per_time
        # Subset of cross-corrs from full model to select for this time step
        t_model_inds = t_model_ind*flaginfo_model.num_visi_per_time_step + include_cross_per_time
        # Stick it in the flagged model
        vis_model_arr_flagged[:, :, t_flag_inds] = vis_model_arr[:, :, t_model_inds]

        # If we're doing autocorrelations, jam them in as well
        if include_autos:
            t_flag_inds = t_data_ind*flaginfo_data.num_visi_per_time_step + flaginfo_data.auto_locs_per_time
            t_model_inds = t_model_ind*flaginfo_model.num_visi_per_time_step + include_auto_per_time

            vis_model_arr_flagged[:, :, t_flag_inds] = vis_model_arr[:, :, t_model_inds]

    return vis_model_arr_flagged

def convert_vis_model_arr_to_sav(vis_model_arr : np.ndarray,
                                 pyfhd_config : dict,
                                 logger : logging.RootLogger,
                                 model_vis_dir : str, n_pol : int):
    """Converts the contents of `vis_model_arr` into an FHD .sav file format
    so we can import into existing IDL code with ease. First writes data to
    `hdf5` format, then uses IDL code template to convert to IDL `.sav` format
    compatible with FHD. Sticks the outputs into `model_vis_dir`.

    Parameters
    ----------
    vis_model_arr : np.ndarray
        Complex array hold the model visibilities
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.RootLogger
        PyFHD logger to feed information to
    model_vis_dir : str
        Directory location to write the output files to
    n_pol : int
        Number of polarisations to write out (each is written to an individual)
        `.sav` file
    """

    pol_names = ['XX', 'YY', 'XY', 'YX']

    logger.info(f"vis_model_transfer: saving {model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5")

    with h5py.File(f"{model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5", 'w') as hf:

        for pol, pol_name in enumerate(pol_names[:n_pol]):
            
            hf.create_dataset(f"{pyfhd_config['obs_id']}_vis_model_{pol_name}",
                                    data=vis_model_arr[pol].transpose())
                
        hf.close()

    #Grab the template IDL code and transfer so people can see what code was used
    #and modify if they want
    model_arr_convert_pro = importlib_resources.files('PyFHD.templates').joinpath('convert_model_arr_to_sav.pro')
    shutil.copy(model_arr_convert_pro, model_vis_dir)

    #Move into the output directory so IDL can see all the .pro files
    os.chdir(model_vis_dir)
    
    #Run the IDL code
    logger.info(f"vis_model_transfer: converting {model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5 to .sav format")

    idl_command = f"idl -IDL_DEVICE ps -e convert_model_arr_to_sav -args {model_vis_dir} {pyfhd_config['obs_id']} {n_pol}"

    run_command(idl_command, False)






