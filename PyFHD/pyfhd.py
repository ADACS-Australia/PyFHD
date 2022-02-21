from ssl import Options
import numpy as np
import astropy 
from pathlib import Path
import configargparse

def pyfhd_parser():
    parser = configargparse.ArgumentParser(
        default_config_files = ["./pyfhd.yaml"], 
        prog = "PyFHD", 
        description = "This is the Python Fast Holographic Deconvolution package, only file_path_vis is required to start your run, but you should need to modify these arguments below to get something useful.", 
        config_file_parser_class = configargparse.YAMLConfigFileParser,
        args_for_setting_config_path = ['-c', '--config'],

    )
    parser.add_argument('--test', help = "Test argument", action="store_true")
    parser.add_argument('--stringy', type = str)
    parser.add_argument('--test2', help = "Test argument 2", action="store_true")
    parser.add_argument('-p', '--path', type = Path)

    return parser

def main():
    options = pyfhd_parser().parse_args()
    print(options)
    print(Path.cwd())
    if options.test:
        print("Hello world!")
        print(options.stringy)
        print(options.path)
    else:
        print("Fail On Purpose World!")
    print("Hi, I'm the main PyFHD function, if you're seeing me, well it's good, just trust me, it's good.")