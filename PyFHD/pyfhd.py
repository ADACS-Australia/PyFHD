import logging
import sys
import time
from datetime import timedelta
import h5py
import numpy as np
from pathlib import Path
from PyFHD.beam_setup.beam import create_psf
from PyFHD.calibration.calibrate import calibrate, calibrate_qu_mixing
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.uvfits import (create_layout, create_params,
                                     extract_header, extract_visibilities)
from PyFHD.flagging.flagging import vis_flag, vis_flag_basic
from PyFHD.gridding.gridding_utils import crosspol_reformat
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.pyfhd_tools.pyfhd_setup import (pyfhd_parser, pyfhd_setup,
                                           write_collated_yaml_config)
from PyFHD.pyfhd_tools.pyfhd_utils import (simple_deproject_w_term,
                                           vis_noise_calc, vis_weights_update)
from PyFHD.source_modeling.vis_model_transfer import (flag_model_visibilities,
                                                      vis_model_transfer)
from PyFHD.io.pyfhd_io import save, load
from PyFHD.use_idl_fhd.run_idl_fhd import (
    run_IDL_calibration_only, run_IDL_convert_gridding_to_healpix_images)
from PyFHD.use_idl_fhd.use_idl_outputs import run_gridding_on_IDL_outputs


def _print_time_diff(start : float, end : float, description : str, logger : logging.Logger):
    """
    Print the time difference in a nice format between start and end time

    Parameters
    ----------
    start : float
        Start time in seconds since epoch
    end : float
        End time in seconds since epoch
    """
    runtime = end - start
    if runtime > 60:
        runtime = timedelta(seconds=end - start)
        logger.info(f'{description} completed in: {runtime}')
    elif runtime < 1:
        logger.info(f'{description} completed in: {round(runtime * 1000,5)} milliseconds')
    else:
        logger.info(f'{description} completed in: {round(runtime,5)} seconds')

def main_python_only(pyfhd_config : dict, logger : logging.Logger):
    """One day, this python only loop will just be main. For now, only try and
    run it if none of the IDL options are asked for.

    Parameters
    ----------
    pyfhd_config : dict
        _The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.Logger
        _The logger to output info and errors to
    """

    if pyfhd_config['obs_checkpoint'] is None and pyfhd_config['calibrate_checkpoint'] is None:
        header_start = time.time()
        # Get the header
        pyfhd_header, params_data, antenna_header, antenna_data = extract_header(pyfhd_config, logger)
        header_end = time.time()
        _print_time_diff(header_start, header_end, 'PyFHD Header Created', logger)

        params_start = time.time()
        # Get params
        params = create_params(pyfhd_header, params_data, logger)
        params_end = time.time()
        _print_time_diff(params_start, params_end, 'Params Created', logger)

        visibility_start = time.time()
        vis_arr, vis_weights = extract_visibilities(pyfhd_header, params_data, pyfhd_config, logger)
        visibility_end = time.time()
        _print_time_diff(visibility_start, visibility_end, 'Visibilities Extracted', logger)

        # If you wish to reorder your visibilities, insert your function to do that here.
        # If you wish to average your fits data by time or frequency, insert your functions to do that here

        layout_start = time.time()
        layout = create_layout(antenna_header, antenna_data, pyfhd_config, logger)
        layout_end = time.time()
        _print_time_diff(layout_start, layout_end, 'Layout Dictionary Extracted', logger)
        
        # Get obs
        obs_start = time.time()
        obs = create_obs(pyfhd_header, params, layout, pyfhd_config, logger)
        obs_end = time.time()
        _print_time_diff(obs_start, obs_end, 'Obs Dictionary Created', logger)

        # If you decide to use the PyFHD checkpoint system, save the uncalibrated visibility & 
        # observation data and metadata now
        if pyfhd_config['save_checkpoints']:
            checkpoint = {
                'obs': obs,
                'params': params,
                'vis_arr': vis_arr,
                'vis_weights': vis_weights
            }
            save(Path(pyfhd_config['output_dir'], 'obs_checkpoint.h5'), checkpoint, 'obs_checkpoint', logger = logger)
            logger.info(f"Checkpoint Saved: Uncalibrated visibility parameters, array and weights and the observation metadata dictionary saved into {Path(pyfhd_config['output_dir'], 'obs_checkpoint.h5')}")
    else:
        # Load the checkpoint and initialize the required variables
        obs_checkpoint = load(pyfhd_config['obs_checkpoint'], logger = logger)
        obs = obs_checkpoint['obs']
        params = obs_checkpoint['params']
        vis_arr = obs_checkpoint['vis_arr']
        vis_weights = obs_checkpoint['vis_weights']
        logger.info(f"Checkpoint Loaded: Uncalibrated visibility parameters, array and weights and the observation metadata dictionary loaded from {Path(pyfhd_config['output_dir'], 'obs_checkpoint.h5')}")

    # Read in the beam from a file returning a psf dictionary
    psf_start = time.time()
    psf = create_psf(pyfhd_config, logger)
    psf_end = time.time()
    _print_time_diff(psf_start, psf_end, 'Beam and PSF dictionary imported.', logger)

    if psf['image_info']['image_power_beam_arr'] is not None and psf['image_info']['image_power_beam_arr'].shape == 1:
        # Turn off beam_per_baseline if image_power_beam_arr is 
        # only one value
        pyfhd_config['beam_per_baseline'] = False
    
    # Check if the calibrate checkpoint has been used, if not run the calibration steps
    if pyfhd_config['calibrate_checkpoint'] is None:
        if pyfhd_config['deproject_w_term'] is not None:
            w_term_start = time.time()
            vis_arr = simple_deproject_w_term(obs, params, vis_arr, pyfhd_config['deproject_w_term'], logger)
            w_term_end = time.time()
            _print_time_diff(w_term_start, w_term_end, 'Simple W-Term Deprojection Applied', logger)

        # Peform basic flagging
        if (pyfhd_config['flag_basic']):
            basic_flag_start = time.time()
            vis_weights, obs = vis_flag_basic(vis_weights, vis_arr, obs, pyfhd_config, logger)
            basic_flag_end = time.time()
            _print_time_diff(basic_flag_start, basic_flag_end, 'Basic Flagging Completed', logger)

        # Update the visibility weights
        weight_start = time.time()
        vis_weights, obs = vis_weights_update(vis_weights, obs, psf, params)
        weight_end = time.time()
        _print_time_diff(weight_start, weight_end, 'Visibilities Weights Updated After Basic Flagging', logger)

        # Get the vis_model_arr from a UVFITS file or SAV files and flag any issues
        vis_model_arr_start = time.time()
        vis_model_arr, params_model = vis_model_transfer(pyfhd_config, obs, logger)
        if pyfhd_config['flag_model']:
            vis_model_arr = flag_model_visibilities(vis_model_arr, params, params_model, obs, pyfhd_config, logger)
        vis_model_arr_end = time.time()
        _print_time_diff(vis_model_arr_start, vis_model_arr_end, 'Model Imported and Flagged From UVFITS', logger)

        # Skipped initializing the cal structure as it mostly just copies values from the obs, params, config and the skymodel from FHD
        # However, there is resulting cal structure for logging and output purposes to store the resulting gain and any other associated
        # arrays
        if pyfhd_config['calibrate_visibilities']:
            logger.info("Beginning Calibration")
            cal_start = time.time()
            vis_arr, cal, obs = calibrate(obs, params, vis_arr, vis_weights, vis_model_arr, pyfhd_config, logger)
            cal_end = time.time()
            _print_time_diff(cal_start, cal_end, 'Visibilities calibrated and cal dictionary with gains created', logger)

            if (obs['n_pol'] >= 4):
                qu_mixing_start = time.time()
                cal["stokes_mix_phase"] = calibrate_qu_mixing(vis_arr, vis_model_arr, vis_weights, obs)
                qu_mixing_end = time.time()
                _print_time_diff(qu_mixing_start, qu_mixing_end, 'Calibrate QU-Mixing has finished, result in cal["stokes_mix_phase"]', logger)

            weight_start = time.time()
            vis_weights, obs = vis_weights_update(vis_weights, obs, psf, params)
            weight_end = time.time()
            _print_time_diff(weight_start, weight_end, 'Visibilities Weights Updated After Calibration', logger)

            if (pyfhd_config['flag_visibilities']):
                flag_start = time.time()
                vis_weights, obs = vis_flag(vis_arr, vis_weights, obs, params)
                flag_end = time.time()
                _print_time_diff(flag_start, flag_end, 'Visibilities Flagged', logger)

            # TODO: save flagged weights and obs here

            noise_start = time.time()
            obs['vis_noise'] = vis_noise_calc(obs, vis_arr, vis_weights)
            noise_end = time.time()
            _print_time_diff(noise_start, noise_end, 'Noise Calculated and added to obs', logger)

            if pyfhd_config['save_checkpoints']:
                checkpoint = {
                    "obs": obs,
                    'params': params,
                    "vis_arr": vis_arr,
                    "vis_model_arr": vis_model_arr,
                    "vis_weights": vis_weights,
                    "cal": cal,
                }
                save(Path(pyfhd_config['output_dir'], 'calibrate_checkpoint.h5'), checkpoint, "calibrate_checkpoint", logger = logger)
                logger.info(f"Checkpoint Saved: Calibrated and Flagged visibility parameters, array and weights, the flagged observation metadata dictionary and the calibration dictionary saved into {Path(pyfhd_config['output_dir'], 'calibrate_checkpoint.h5')}")
    else:
        # Load the calibration checkpoint
        cal_checkpoint = load(pyfhd_config['calibrate_checkpoint'], logger = logger)
        obs = cal_checkpoint['obs']
        params = cal_checkpoint['params']
        vis_arr = cal_checkpoint['vis_arr']
        vis_model_arr = cal_checkpoint['vis_model_arr']
        vis_weights = cal_checkpoint['vis_weights']
        cal = cal_checkpoint['cal']
        logger.info(f"Checkpoint Loaded: Calibrated and Flagged visibility parameters, array and weights, the flagged observation metadata dictionary and the calibration dictionary loaded from {Path(pyfhd_config['output_dir'], 'calibrate_checkpoint.h5')}")

    # TODO: save the psf here as h5, sav files take a while to read, and then add in hdf5 reader into the import beam function
    if pyfhd_config['recalculate_grid'] and pyfhd_config['gridding_checkpoint'] is None:
        grid_start = time.time()
        # Since it's done per polarization, we can do multi-processing if it's not fast enough
        for pol_i in range(obs["n_pol"]):
            logger.info(f"Gridding has begun for polarization {pol_i}")
            image_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]), dtype = np.complex128)
            weights_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]), dtype = np.complex128)
            variance_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]))
            uniform_filter_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]))
            if vis_model_arr is not None:
                model_uv = np.empty((obs["n_pol"], obs["elements"], obs["dimension"]), dtype = np.complex128)
            if pol_i == 0:
                uniform_flag = True
                no_conjugate = False
            else:
                uniform_flag = False
                no_conjugate = True
            gridding_dict = visibility_grid(
                vis_arr[pol_i], 
                vis_weights[pol_i], 
                obs, 
                psf, 
                params, 
                pol_i, 
                pyfhd_config, 
                logger, 
                uniform_flag = uniform_flag, 
                no_conjugate = no_conjugate, 
                model = vis_model_arr[pol_i]
            )
            if len(gridding_dict.keys()) != 0:
                image_uv[pol_i] = gridding_dict['image_uv']
                weights_uv[pol_i] = gridding_dict['weights']
                variance_uv[pol_i] = gridding_dict['variance']
                uniform_filter_uv[pol_i] = gridding_dict['uniform_filter']
                obs['nf_vis'] = gridding_dict["obs"]["nf_vis"]
                if vis_model_arr is not None:
                    model_uv[pol_i] = gridding_dict['model_return']
                logger.info(f"Gridding has finished for polarization {pol_i}")
            else:
                logger.error("All data was flagged during gridding, exiting")
                sys.exit(1)
        if obs["n_pol"] == 4:
            logger.info("Performing Crosspol reformatting")
            image_uv = crosspol_reformat(image_uv)
            weights_uv = crosspol_reformat(weights_uv)
            if vis_model_arr is not None:
                model_uv = crosspol_reformat(model_uv)
        if pyfhd_config['save_checkpoints']:
            checkpoint = {
                "image_uv": image_uv,
                'weights_uv': weights_uv,
                "variance_uv": variance_uv,
                "uniform_filter_uv": uniform_filter_uv,
            }
            if vis_model_arr is not None:
                checkpoint["model_uv"] = model_uv
            save(Path(pyfhd_config['output_dir'], 'gridding_checkpoint.h5'), checkpoint, "gridding_checkpoint", logger = logger)
            logger.info(f"Checkpoint Saved: The Gridded UV Planes saved into {Path(pyfhd_config['output_dir'], 'gridding_checkpoint.h5')}")
        grid_end = time.time()
        _print_time_diff(grid_start, grid_end, 'Visibilities gridded', logger)
    else:
        grid_checkpoint = load(pyfhd_config['gridding_checkpoint'], logger = logger)
        image_uv = grid_checkpoint['image_uv']
        weights_uv = grid_checkpoint['weights_uv']
        variance_uv = grid_checkpoint['variance_uv']
        uniform_filter_uv = grid_checkpoint['uniform_filter_uv']
        if 'model_uv' in grid_checkpoint:
            model_uv = grid_checkpoint['model_uv']
        logger.info(f"Checkpoint Loaded: The Gridded UV Planes loaded from {Path(pyfhd_config['output_dir'], 'gridding_checkpoint.h5')}")

    # TODO: Translate fhd_quickview and add it here

    # TODO: Translate snapshot_healpix_export and add it here

    # Close all open h5 files
    if isinstance(psf, h5py.File):
        psf.close()

    # Write a final collated yaml for the final pyfhd_config
    write_collated_yaml_config(pyfhd_config, pyfhd_config['output_dir'], "-final")

def main():

    pyfhd_start = time.time()
    options = pyfhd_parser().parse_args()
    
    # Validate options and Create the Logger
    pyfhd_config, logger = pyfhd_setup(options)

    #If any of the hybrid options have been asked for, circumnavigate the
    #main_loop_python_only function, and run the required hybrid options
    if options.IDL_calibrate or options.grid_IDL_outputs or options.IDL_healpix_gridded_outputs:

        idl_output_dir = None

        if options.IDL_calibrate:
            idl_output_dir = run_IDL_calibration_only(pyfhd_config, logger)

        if options.grid_IDL_outputs:
            if idl_output_dir != None:
                pass
            else:
                idl_output_dir = f"{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}/fhd_{pyfhd_config['top_level_dir']}"

            run_gridding_on_IDL_outputs(pyfhd_config, idl_output_dir, logger)

        if options.IDL_healpix_gridded_outputs:
            run_IDL_convert_gridding_to_healpix_images(pyfhd_config, logger)

    else:
        main_python_only(pyfhd_config, logger)


    pyfhd_end = time.time()
    runtime = timedelta(seconds = pyfhd_end - pyfhd_start)
    logger.info(f'PyFHD Run Completed for {pyfhd_config["obs_id"]}\nTotal Runtime (Days:Hours:Minutes:Seconds.Millseconds): {runtime}')
    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()

    
