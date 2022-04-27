import numpy as np
import astropy 
from pathlib import Path
from PyFHD.pyfhd_tools.parser import pyfhd_parser, setup_parser

def main():
    options = pyfhd_parser().parse_args()
    # if not options.silent:
    print("Hi, I'm the main PyFHD function, if you're seeing me, well it's good, just trust me, it's good.")
    setup_parser(options)
    # Start Logging
    # if options.logging:
        # pass