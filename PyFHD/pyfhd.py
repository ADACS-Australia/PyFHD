import time
from datetime import timedelta
from logging import RootLogger
import numpy as np
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.uvfits import extract_header, create_params, extract_visibilities, create_layout
from PyFHD.pyfhd_tools.pyfhd_utils import simple_deproject_w_term, vis_weights_update, vis_noise_calc
from PyFHD.calibration.calibrate import calibrate, calibrate_qu_mixing
from PyFHD.use_idl_fhd.run_idl_fhd import run_IDL_calibration_only, run_IDL_convert_gridding_to_healpix_images
from PyFHD.use_idl_fhd.use_idl_outputs import run_gridding_on_IDL_outputs
from PyFHD.flagging.flagging import vis_flag
import logging

def _print_time_diff(start : float, end : float, description : str, logger : RootLogger):
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

def main_python_only(pyfhd_config : dict, logger : logging.RootLogger):
    """One day, this python only loop will just be main. For now, only try and
    run it if none of the IDL options are asked for.

    Parameters
    ----------
    pyfhd_config : dict
        _The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.RootLogger
        _The logger to output info and errors to
    """

    header_start = time.time()
    # Get the header
    pyfhd_header, params_data, antenna_table = extract_header(pyfhd_config, logger)
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
    layout = create_layout(antenna_table, logger)
    layout_end = time.time()
    _print_time_diff(layout_start, layout_end, 'Layout Dictionary Extracted', logger)

    # TODO: Save the layout here later

    # if pyfhd_config['run_simulation']:
        # TODO: in_situ_sim_input()
    
    # Get obs
    obs_start = time.time()
    obs = create_obs(pyfhd_header, params, pyfhd_config, logger)
    obs_end = time.time()
    _print_time_diff(obs_start, obs_end, 'Obs Dictionary Created', logger)

    if pyfhd_config['deproject_w_term'] is not None:
        w_term_start = time.time()
        vis_arr = simple_deproject_w_term(obs, params, vis_arr, pyfhd_config['deproject_w_term'], logger)
        w_term_end = time.time()
        _print_time_diff(w_term_start, w_term_end, 'Simple W-Term Deprojection Applied', logger)

    # Skipped initializing the cal structure as it mostly just copies values from the obs, params, config and the skymodel from FHD
    # However, there may be a resulting cal structure for logging and output purposes depending on calibration translation.
    vis_arr, vis_model_arr, cal = calibrate(obs, params, vis_arr, vis_weights, pyfhd_config, logger)

    if (obs['n_pol'] >= 4):
        cal["stokes_mix_phase"] = calibrate_qu_mixing(vis_arr, vis_model_arr, vis_weights, obs)

    vis_weights, obs = vis_weights_update(vis_weights, obs, params, pyfhd_config)

    vis_weights, obs = vis_flag(vis_arr, vis_weights, obs, params)

    obs['vis_noise'] = vis_noise_calc(obs, vis_arr, vis_weights)

    # TODO: add the gridding function after calibration testing is finished

    # np.save('../notebooks/pyfhd_config.npy', pyfhd_config, allow_pickle=True)

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
    logger.info(f'PyFHD Run Completed\nTotal Runtime (Days:Hours:Minutes:Seconds.Millseconds): {runtime}')
    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()

    
