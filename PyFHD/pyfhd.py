import time
from datetime import timedelta
from logging import RootLogger
import numpy as np
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.uvfits import extract_header, create_params, extract_visibilities

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
    if end - start > 60:
        runtime = timedelta(seconds=end - start)
        logger.info(f'{description} run time: {runtime}')
    else:
        runtime = end - start
        logger.info(f'{description} run time: {runtime} seconds')
    

def main():

    pyfhd_start = time.time()
    options = pyfhd_parser().parse_args()
    
    # Validate options and Create the Logger
    pyfhd_config, logger = pyfhd_setup(options)

    header_start = time.time()
    # Get the header
    pyfhd_header, fits_data = extract_header(pyfhd_config, logger)
    header_end = time.time()
    _print_time_diff(header_start, header_end, 'PyFHD Header Created', logger)

    params_start = time.time()
    # Get params
    params = create_params(pyfhd_header, fits_data, logger)
    params_end = time.time()
    _print_time_diff(params_start, params_end, 'PyFHD Params Created', logger)

    vis_arr, vis_weights = extract_visibilities(fits_data)

    # If you wish to reorder your visibilities, insert your function to do that here.
    # If you wish to average your fits data by time or frequency, insert your functions to do that here

    # Get obs
    # create_obs(pyfhd_header, params, pyfhd_config, logger)

    pyfhd_end = time.time()
    runtime = timedelta(seconds = pyfhd_end - pyfhd_start)
    logger.info(f'PyFHD run time: {runtime}')
    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()
