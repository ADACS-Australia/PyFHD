import numpy as np
import astropy 
from pathlib import Path
from PyFHD.pyfhd_tools.pyfhd_setup import pyfhd_parser, setup_parser
import time
import subprocess
import logging
import shutil

def main():
    options = pyfhd_parser().parse_args()
    # Get the time, Git commit and setup the name of the output directory
    run_time = time.localtime()
    stdout_time = time.strftime("%c", run_time)
    log_time = time.strftime("%H_%M_%S_%d_%m_%Y", run_time)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text = True).stdout
    if options.description is None:
        log_name = "pyfhd_" + log_time
    else:
        log_name = "pyfhd_" + options.description.replace(' ', '_').lower() + log_time
    # Format the starting string for logging
    start_string = """\
    ________________________________________________________________________
    |    ooooooooo.               oooooooooooo ooooo   ooooo oooooooooo.    |
    |    8888   `Y88.             8888       8 8888    888   888     Y8b    |
    |    888   .d88' oooo    ooo  888          888     888   888      888   |
    |    888ooo88P'   `88.  .8'   888oooo8     888ooooo888   888      888   |
    |    888           `88..8'    888          888     888   888      888   |
    |    888            `888'     888          888     888   888     d88'   |
    |    o888o            .8'     o888o        o888o   o888o o888bood8P'    |
    |                 .o..P'                                                |
    |                `Y8P'                                                  |
    |_______________________________________________________________________|
        Python Fast Holographic Deconvolution 

        Translated from IDL to Python as a collaboration between Astronomy Data and Computing Services (ADACS) and the Epoch of Reionisation (EoR) Team.

        Repository: https://github.com/ADACS-Australia/PyFHD

        Documentation: https://pyfhd.readthedocs.io/en/latest/

        Git Commit Hash: {}

        PyFHD Run Started At: {}

        Observation ID: {}
        
        Validating your input...""".format(commit.replace('\n',''), stdout_time, options.obs_id)
    
    # Validate the Options given
    setup_parser(options)

    # Setup logging
    log_string = ""
    for line in start_string.split('\n'):
        log_string += line.lstrip().replace('_', ' ').replace('|    ', '').replace('|', '') +'\n'
    # Start the PyFHD run
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Create the logging for the temrinal
    if not options.silent:
        log_terminal = logging.StreamHandler()
        log_terminal.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(log_terminal)

    # Create the output directory
    output_dir = Path(options.output_path, log_name)
    if Path.is_dir(output_dir):
        raise OSError("You have a directory inside your specified output directory already named by this date and time...Probably not a time travel paradox right...?")
    else:
        Path.mkdir(output_dir)

    # Create the logger for the file
    if not options.disable_log:
        log_file = logging.FileHandler(Path(output_dir, log_name + '.log'))
        log_file.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(log_file)
    
    # Show that start message in the terminal and/or log file, unless both are turned off.
    logger.info(log_string)
    if not options.silent:
        log_terminal.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n\t%(message)s', datefmt = '%Y-%m-%d %H:%M:%S'))
    if not options.disable_log:
        log_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n\t%(message)s', datefmt = '%Y-%m-%d %H:%M:%S'))
    logger.info("Log Created and Input Validated Starting Run Now")

    # Copy the Configuration File if it exists to the output directory
    if options.config_file is None:
        shutil.copy('pyfhd.yaml', Path(output_dir, log_name + '.yaml'))
    else:
        shutil.copy(options.config_file, Path(output_dir, log_name + '.yaml'))
    
    logger.info('Configuration copied to the output file, filename: {}'.format(Path(output_dir, log_name + '.yaml')))