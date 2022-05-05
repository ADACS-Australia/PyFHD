import time
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.params import extract_header, create_params

def main():
    pyfhd_start = time.time()
    options = pyfhd_parser().parse_args()
    
    # Validate options and Create the Logger
    pyfhd_config, logger = pyfhd_setup(options)

    # Get the header
    pyfhd_header = extract_header(pyfhd_config, logger)

    # Get params
    # params = create_params(pyfhd_header, pyfhd_config, logger)

    # Get obs
    # create_obs(pyfhd_header, params, pyfhd_config, logger)

    pyfhd_end = time.time()
    runtime = pyfhd_end - pyfhd_start
    logger.info('PyFHD run time (HH:MM:SS){}'.format(runtime.strptime('%H:%M:%S')))
    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()
