import time
from datetime import timedelta
from logging import RootLogger
import numpy as np
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.uvfits import extract_header, create_params, extract_visibilities, create_layout
from PyFHD.pyfhd_tools.pyfhd_utils import simple_deproject_w_term
from PyFHD.beam_setup.beam import create_psf

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

def main():

    pyfhd_start = time.time()
    options = pyfhd_parser().parse_args()
    
    # Validate options and Create the Logger
    pyfhd_config, logger = pyfhd_setup(options)

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
    
    # Beam Setup
    beam_start = time.time()
    psf, antenna = create_psf(pyfhd_config, obs)
    beam_end = time.time()
    _print_time_diff(beam_start, beam_end, 'Beam Setup', logger)

    np.save('../notebooks/pyfhd_config.npy', pyfhd_config, allow_pickle=True)

    pyfhd_end = time.time()
    runtime = timedelta(seconds = pyfhd_end - pyfhd_start)
    logger.info(f'PyFHD Run Completed\nTotal Runtime (Days:Hours:Minutes:Seconds.Millseconds): {runtime}')
    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()
