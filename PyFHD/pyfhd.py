import time
from datetime import timedelta
from logging import RootLogger
import numpy as np
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.uvfits import extract_header, create_params, extract_visibilities, create_layout

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

    np.save('/home/skywatcher/Nextcloud/Projects and Experiments/Curtin/ADACS/Modernization_of_FHD_Epoch_of_Reionization/data/FHD/uvfits_read/config.npy', pyfhd_config)

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
    
    # Get obs
    obs_start = time.time()
    obs = create_obs(pyfhd_header, params, pyfhd_config, logger)
    obs_end = time.time()
    _print_time_diff(obs_start, obs_end, 'Obs Dictionary Created', logger)



    pyfhd_end = time.time()
    runtime = timedelta(seconds = pyfhd_end - pyfhd_start)
    logger.info(f'PyFHD Run Completed, Total Runtime (Days:Hours:Minutes:Seconds.Millseconds): {runtime}')
    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()
