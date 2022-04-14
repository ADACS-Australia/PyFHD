from ssl import Options
import numpy as np
import astropy 
from pathlib import Path
import configargparse

def pyfhd_parser():
    parser = configargparse.ArgumentParser(
        default_config_files = ["./pyfhd.yaml"], 
        prog = "PyFHD", 
        description = "This is the Python Fast Holographic Deconvolution package, only the path is required to start your run, but you should need to modify these arguments below to get something useful.", 
        config_file_parser_class = configargparse.YAMLConfigFileParser,
        args_for_setting_config_path = ['-c', '--config'],

    )
    parser.add_argument('obs_id', help="The Observation ID as per the MWA file naming standards")
    parser.add_argument( '-u', '--uvfits-path', help = "Directory for the uvfits files, by default it looks for a directory called uvfits in the working directory", default = "./uvfits/")
    parser.add_argument('-o','--output-path', help = "Set the output path for the current run, note a directory will still be created inside the given path")
    parser.add_argument('-d', '--description', help = "A more detailed description of the current task, will get applied to the output directory and logging where all output will be stored.")
    parser.add_argument('--stringy', type = str)
    parser.add_argument('--test-arg', help = "Test argument 2")
    #parser.add_argument('-p', '--path', type = Path)

    return parser

def main():
    options = pyfhd_parser().parse_args()
    print("Hi, I'm the main PyFHD function, if you're seeing me, well it's good, just trust me, it's good.")
    print(options)