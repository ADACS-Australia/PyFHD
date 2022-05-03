import numpy as np
import astropy 
from pathlib import Path
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, pyfhd_setup

def main():
    options = pyfhd_parser().parse_args()

    # Validate options and Create the Logger
    pyfhd_config, logger = pyfhd_setup(options)

    # Close the handlers in the log
    for handler in logger.handlers:
        handler.close()