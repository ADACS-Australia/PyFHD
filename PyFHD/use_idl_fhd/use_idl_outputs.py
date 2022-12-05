import subprocess
import pathlib
import os
from PyFHD.use_idl_fhd.run_idl_fhd import run_command
from PyFHD.gridding import visibility_grid
import importlib_resources
import shutil
import logging
from scipy.io import readsav
import numpy as np
import time
import h5py

def convert_sav_to_dict(sav_path : str, logger : logging.RootLogger):
    """
    Given a path to an IDL style .sav file, load into a python dictionary
    using scipy.io.readsav.

    If the file was saved with the IDL /compress option, the readsav function
    has to save a decompressed version of the file. By default this uses
    the tempfile module to find a location, but this usually finds a bad
    location with little storage when called on a super cluster. So explicitly
    make our own temp dir `tmp_pyfhd` where the code is being called. It is
    assumed many files are to be convert, so `tmp_pyfhd` should be deleted
    after all calls.

    Parameters
    ----------
    sav_path : str
        Filepath for an IDL .sav file
    logger : logging.RootLogger
        The logger to output any error messages to

    Returns
    --------
    sav_dict : dict
        Dictionary containing whatever was in the .sav file

    """

    if os.path.isfile(sav_path):
        # logger.info(f"{sav_path} found, converting now.")

        ##Ensure the tmp dir exists, create if not
        run_command('mkdir -p tmp_pyfhd')

        ##Strip off any leading path to leave just the file name
        temp_name = f"tmp_pyfhd/{sav_path.split('/')[-1]}"

        ##Load into a dictionary, decompressed and saving a temporary file if need
        ##be
        sav_dict = readsav(sav_path, python_dict=True, uncompressed_file_name=temp_name)

        return sav_dict
    else:
        # sys.exit(f"{sav_path} does not exist. Cannot grid so exiting")
        logger.error(f"{sav_path} doesn't exist, please check your input path")

        for handler in logger.handlers:
            handler.close()
        exit()

def convert_IDL_calibration_outputs(tag : str, idl_output_dir : str,
                                    logger : logging.RootLogger,
                                    save_npz = False, read_npz = False):
    """Convert IDL FHD .sav files output after calibration into dictionaries,
    in preparation to run gridding on them with the Python code.

    Returns a dictionary containing the various converted .sav files. Contains
    the following keys:

     - obs_dict
     - params_dict
     - variables_dict
     - vis_XX_dict
     - vis_YY_dict
     - vis_model_XX_dict
     - vis_model_YY_dict
     - vis_flags_dict

    Parameters
    ----------
    tag : str
        This is usually the observation id, as it's a string inferrerd by FHD
        from the name of the .uvfits file, e.g. if the input data was
        1088716176.uvfits, tag = 1088716176.
    idl_output_dir : str
        Parent directory in which all IDL FHD outputs reside
    logger : logging.RootLogger
        The logger to output info and errors to
    save_npz : bool, optional
        Save the converted dictionary outputs into .npz files. Useful if you want to quickly load them up in future. By default False.
    read_npz : bool, optional
        If outputs have already been converted to .npz, set this to True to read from .npz and not covert from .sav files. By defauly False.

    Returns
    -------
    idl_cal_dict : dict
        A dictionary contain various converted IDL outputs (each in the form
        of it's own dictionary)
    """
    

    if read_npz:
        logger.info("Reading previously converted IDL FHD calibration .npz files now ")

        obs_dict = np.load(f"{idl_output_dir}/metadata/{tag}_obs_dict.npz",
                           allow_pickle=True)
        params_dict = np.load(f"{idl_output_dir}/metadata/{tag}_params_dict.npz",
                           allow_pickle=True)
        variables_dict = np.load(f"{idl_output_dir}/{tag}_variables_dict.npz",
                           allow_pickle=True)
        vis_XX_dict = np.load(f"{idl_output_dir}/vis_data/{tag}_vis_XX_dict.npz",
                           allow_pickle=True)
        vis_YY_dict = np.load(f"{idl_output_dir}/vis_data/{tag}_vis_YY_dict.npz",
                           allow_pickle=True)
        vis_model_XX_dict = np.load(f"{idl_output_dir}/vis_data/{tag}_vis_model_XX_dict.npz",
                           allow_pickle=True)
        vis_model_YY_dict = np.load(f"{idl_output_dir}/vis_data/{tag}_vis_model_YY_dict.npz",
                           allow_pickle=True)
        vis_flags_dict = np.load(f"{idl_output_dir}/vis_data/{tag}_vis_flags_dict.npz",
                           allow_pickle=True)
    else:
        logger.info("Converting IDL FHD calibration .sav files now ")

        run_command('mkdir -p tmp_pyfhd')

        ##Observational parameters and variables
        obs_dict = convert_sav_to_dict(f"{idl_output_dir}/metadata/{tag}_obs.sav", logger)
        params_dict = convert_sav_to_dict(f"{idl_output_dir}/metadata/{tag}_params.sav", logger)
        variables_dict = convert_sav_to_dict(f"{idl_output_dir}/{tag}_variables.sav", logger)

        ##Visibility data
        vis_XX_dict = convert_sav_to_dict(f"{idl_output_dir}/vis_data/{tag}_vis_XX.sav", logger)
        vis_YY_dict = convert_sav_to_dict(f"{idl_output_dir}/vis_data/{tag}_vis_YY.sav", logger)
        vis_model_XX_dict = convert_sav_to_dict(f"{idl_output_dir}/vis_data/{tag}_vis_model_XX.sav", logger)
        vis_model_YY_dict = convert_sav_to_dict(f"{idl_output_dir}/vis_data/{tag}_vis_model_YY.sav", logger)
        vis_flags_dict = convert_sav_to_dict(f"{idl_output_dir}/vis_data/{tag}_flags.sav", logger)

        ##Remove the temporary folder holding the decompressed .sav files
        run_command('rm -r tmp_pyfhd')

        if save_npz:

            np.savez(f"{idl_output_dir}/metadata/{tag}_obs_dict.npz", **obs_dict)
            np.savez(f"{idl_output_dir}/metadata/{tag}_params_dict.npz", **params_dict)
            np.savez(f"{idl_output_dir}/{tag}_variables_dict.npz", **variables_dict)
            np.savez(f"{idl_output_dir}/vis_data/{tag}_vis_XX_dict.npz", **vis_XX_dict)
            np.savez(f"{idl_output_dir}/vis_data/{tag}_vis_YY_dict.npz", **vis_YY_dict)
            np.savez(f"{idl_output_dir}/vis_data/{tag}_vis_model_XX_dict.npz", **vis_model_XX_dict)
            np.savez(f"{idl_output_dir}/vis_data/{tag}_vis_model_YY_dict.npz", **vis_model_YY_dict)
            np.savez(f"{idl_output_dir}/vis_data/{tag}_vis_flags_dict.npz", **vis_flags_dict)

    keys = ['obs_dict', 'params_dict', 'variables_dict', 'vis_XX_dict', 'vis_YY_dict', 'vis_model_XX_dict', 'vis_model_YY_dict', 'vis_flags_dict']

    values = [obs_dict, params_dict, variables_dict, vis_XX_dict, vis_YY_dict, vis_model_XX_dict, vis_model_YY_dict, vis_flags_dict]

    idl_cal_dict = dict(zip(keys, values))

    return idl_cal_dict


def run_gridding_on_IDL_outputs(pyfhd_config : dict, idl_output_dir : str,
                                logger : logging.RootLogger):
    """Assuming that `run_IDL_calibration_only` has been run to create IDL
    FHD outputs, read in those outputs, and grid them, according to the
    paramaters defined in `pyfhd_config`.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    idl_output_dir : str
        Parent directory in which all IDL FHD outputs reside
    logger : logging.RootLogger
        The logger to output info and errors to
    """

    before = time.time()

    ##Convert/load in the IDL FHD outputs
    idl_cal_dict = convert_IDL_calibration_outputs(pyfhd_config['obs_id'],
                                                   idl_output_dir, logger)

    ##Grab specific arrays/rec arrays from idl_cal_dict needed to run gridding
    visibility_XX = idl_cal_dict['vis_XX_dict']['vis_ptr']
    visibility_YY = idl_cal_dict['vis_YY_dict']['vis_ptr']
    
    vis_model_XX = idl_cal_dict['vis_model_XX_dict']['vis_model_ptr']
    vis_model_YY = idl_cal_dict['vis_model_YY_dict']['vis_model_ptr']
    
    obs = idl_cal_dict['obs_dict']['obs']
    status_str = "OH NO"
    
    if pyfhd_config['ps_kspan']:
        obs.dimension = int(2*pyfhd_config['ps_kspan'])
        obs.elements = int(2*pyfhd_config['ps_kspan'])
        logger.info(f"Resetting obs.dimension and obs.elements to {int(2*pyfhd_config['ps_kspan'])}.")
        
    params = idl_cal_dict['params_dict']['params']
    
    variables_dict = idl_cal_dict['variables_dict']
    weights_flag = variables_dict['weights_flag']
    variance_flag = variables_dict['variance_flag']
    # preserve_visibilities = variables_dict['preserve_visibilities']
    # model_return = variables_dict['model_return']

    bi_use_even = variables_dict['bi_use'][0]
    bi_use_odd = variables_dict['bi_use'][1]
    vis_weights = idl_cal_dict['vis_flags_dict']['vis_weights']
    
    n_freq = obs['n_freq'][0]

    ##Report how long it took to load in the IDL calibration sav files
    after = time.time()
    logger.info(f"Reading/importing IDL calibration sav files took {after - before:.1f} seconds\n\tNow loading in the gridding psf object. This can take minutes.")

    ##Load up the gridding psf object, and report how long it takes
    before = time.time()
    psf_dict = np.load(pyfhd_config['grid_psf_file_npz'], allow_pickle=True)
    psf = psf_dict['psf']
    after = time.time()
    logger.info(f"Loading the gridding psf object took {after - before:.1f} seconds.\n\tNow launching python gridding loop")

    ##MUST change dimension of psf ID or things will crash
    ##Has to do with flagging of the real data no being applied
    ##to the perfect beam model
    psf.id[0] = psf.id[0][:obs.nbaselines[0], :, :]

    freq_use = obs['baseline_info'][0]['freq_use'][0]
    ##TODO how do we remove this hardcording
    n_avg = 2. #Hardcode to grid two freqs together for 160kHz res uvf cubes, FHD/epp standard
    freq_bin_i2 = np.floor(np.arange(n_freq) / n_avg).astype(int)
    nf = np.max(freq_bin_i2) + 1

    pol_names = ['XX', 'YY']
    visibilities = [visibility_XX, visibility_YY]
    models = [vis_model_XX, vis_model_YY]
    bi_use_labels = ['even', 'odd']
    bi_uses = [bi_use_even, bi_use_odd]

    gridding_dir = f"{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}/gridding_outputs"

    ## Make somewhere to store the outputs
    run_command(f'mkdir -p {gridding_dir}')

    before = time.time()
    
    for pol_ind, pol_label in enumerate(pol_names):
        for bi_use, bi_use_label in zip(bi_uses, bi_use_labels):
            
            save_name = f"{gridding_dir}/{pyfhd_config['obs_id']}_gridded_uv_cube_{bi_use_label}_{pol_label}.h5"
                
            with h5py.File(save_name, 'w') as hf:
            
                for freq_ind in range(nf):

                    print(f"gridding {pol_label}_{bi_use_label}_freqind{freq_ind:03d}")

                    ##Which frequencies to use in this subset of gridding
                    fi_use = np.where((freq_bin_i2 == freq_ind) & (freq_use > 0))[0]
                    
                    these_visis = visibilities[pol_ind]
                    these_model = models[pol_ind]
                    
                    gridding_dict = visibility_grid.visibility_grid(visibilities[pol_ind],
                            vis_weights[pol_ind],
                            obs, status_str, psf, params,
                            weights_flag = weights_flag,
                            variance_flag = variance_flag,
                            polarization = pol_ind,
                            fi_use = fi_use, bi_use = bi_use,
                            model = models[pol_ind])

                    name = f"{pol_label}_{bi_use_label}_freqind{freq_ind:03d}"
                    
                    hf.create_dataset(f'dirty_uv_{name}',
                                      data=gridding_dict['image_uv'])
                    hf.create_dataset(f'weights_holo_{name}',
                                      data=gridding_dict['weights'])
                    hf.create_dataset(f'variance_holo_{name}',
                                      data=gridding_dict['variance'])
                    hf.create_dataset(f'model_return_{name}',
                                      data=gridding_dict['model_return'])
                    ##This one is n_vis = np.sum(bin_n)
                    hf.create_dataset(f'n_vis_{name}',
                                    data=gridding_dict['n_vis'])

                ##This one is the number of visibilities per frequency
                hf.create_dataset(f'nf_vis',
                                    data=gridding_dict['obs']['nf_vis'][0])
                
                hf.close()

    after = time.time()
            
    logger.info(f"Gridding/saving outputs took {(after - before) / 60.0:.1f} minutes")