import numpy as np
from pathlib import Path
import configargparse
import argparse

def pyfhd_parser():
    parser = configargparse.ArgumentParser(
        default_config_files = ["./pyfhd.yaml"], 
        prog = "PyFHD", 
        description = "This is the Python Fast Holographic Deconvolution package, only the observation ID (obs_id) is required to start your run, but you should need to modify these arguments below to get something useful.", 
        config_file_parser_class = configargparse.YAMLConfigFileParser,
        args_for_setting_config_path = ['-c', '--config'],
        formatter_class= configargparse.RawTextHelpFormatter,
    )
    # Add All the Groups
    calibration = parser.add_argument_group('Calibration', 'Adjust Parameters for Calibration')
    flag = parser.add_argument_group('Flagging', 'Adjust Parameters for Flagging')
    beam = parser.add_argument_group('Beam Setup', 'Adjust Parameters for the Beam Setup')
    gridding = parser.add_argument_group('Gridding', 'Tune the Gridding in PyFHD')
    deconv = parser.add_argument_group('Deconvolution', 'Tune the Degridding in PyFHD')
    export = parser.add_argument_group('Export', 'Adjust the outputs of the PyFHD pipeline')
    model = parser.add_argument_group('Model', 'Tune the modelling in PyFHD')
    sim = parser.add_argument_group('Simulation', 'Turn On Simulation and Tune the simulation')
    healpix = parser.add_argument_group('HEALPIX', 'Adjust the HEALPIX output')
    # General Defaults
    parser.add_argument('obs_id', help="The Observation ID as per the MWA file naming standards. Assumes the fits files for this observation is in the uvfits-path. obs_id and uvfits replace file_path_vis from FHD")
    parser.add_argument( '-u', '--uvfits-path', type = Path, help = "Directory for the uvfits files, by default it looks for a directory called uvfits in the working directory", default = "./uvfits/")
    parser.add_argument('-r', '--recalculate-all', default = False, action='store_true', help = 'Forces PyFHD to recalculate all values. This will ignore values set for recalculate-grid, recalculate-beam, recalculate-mapfn as it will set all of them to True')
    parser.add_argument('-s', '--silent', default = False, action = 'store_true', help = 'This PyFHD stops all output to the terminal unless there is an error or warning.')
    parser.add_argument('-l', '--logging', default = True, action = 'store_true', help = 'Logging is enabled by default, set to False to disable it')
    parser.add_argument('--dimension', type = int, default = 2048, help = 'The number of pixels in the UV plane along one axis.')
    parser.add_argument('--FoV', '--fov', type = float, default = 0, help = 'A proxy for the field of view in degrees. FoV is actually used to determine kbinsize, which will be set to !RaDeg/FoV.\nThis means that the pixel size at phase center times dimension is approximately equal to FoV, which is not equal to the actual field of view owing to larger pixel sizes away from phase center.\nIf set to 0, then kbinsize determines the UV resolution.')
    parser.add_argument('--deproject-w-term', type = float, default = None, help = 'Enables the function for simple_deproject_w_term and uses the parameter value for the direction value in the function')
    parser.add_argument('--conserve-memory', default = False, action = 'store_true', help = 'Optionally split many loops into chunks in the case of high memory usage.')
    parser.add_argument('--memory-threshold', type = int, default = 1e8, help = 'Set a memory threshold for each chunk in set in bytes. By default it is set at ~100MB')
    parser.add_argument('--n-avg', type = int, default = 2, help = 'Number of frequencies to average over to smooth the frequency band.')
    parser.add_argument('--min-baseline', type = float, default = 1.0, help = 'The minimum baseline length in wavelengths to include in the analysis')

    # Calibration Group
    calibration.add_argument('-cv', '--calibrate-visibilities', default = True, action = 'store_true', help = 'Turn on the calibration of the visibilities. If turned on, calibration of the dirty, modelling, and subtraction to make a residual occurs. Otherwise, none of these occur and an uncalibrated dirty cube is output.')
    calibration.add_argument('--diffuse-calibrate', type = Path, help = 'Path to a file containing a map/model of the diffuse in which to calibrate on.\nThe map/model undergoes a DFT for every pixel, and the contribution from every pixel is added to the model visibilities from which to calibrate on.\nIf no diffuse_model is specified, then this map/model is used for the subtraction model as well. See diffuse_model for information about the formatting of the file.')
    calibration.add_argument('--transfer-calibration', type = Path, help = 'The file path of a calibration to be read-in, if you give a directory PyFHD expects there to be a file called <obs_id>_cal.hdf5 using the same observation as you plan to process.')
    calibration.add_argument('--calibration-catalog-file-path', type = Path, default = 'GLEAM_v2_plus_rlb2019.sav', help = 'The file path to the desired source catalog to be used for calibration')
    calibration.add_argument('--return-cal-visibilities', default = True, action = 'store_true', help = "Saves the visibilities created for calibration for use in the model.\nIf model_visibilities is set to False, then the calibration model visibilities and the model visibilities will be the same if return_cal_visibilities is set.\nIf model_visibilities is set to True, then any new modelling (of more sources, diffuse, etc.) will take place and the visibilities created for the calibration model will be added.\nIf n_pol = 4 (full pol mode), return_cal_visibilites must be set because the visibilites are required for calculating the mixing angle between Q and U.")
    calibration.add_argument('--cal-stop', default = False, action = 'store_true', help = 'Stops the code right after calibration, and saves unflagged model visibilities along with the obs structure in a folder called cal_prerun in the PyFHD file structure.\nThis allows for post-processing calibration steps like multi-day averaging, but still has all of the needed information for minimal reprocessing to get to the calibration step.\nTo run a post-processing run, see keywords model_transfer and transfer_psf')
    calibration.add_argument('--transfer-model-uv', type = Path, default = None, help = "A path to save a model uv array.\nIf it's a file that doesnt exist then vis_calibrate will create one for this run, otherwise if the file exists PyFHD will read it in for this run.\nReplaces model_uv_transfer")
    calibration.add_argument('--min-cal-baseline', type = float, default = 50.0, help = 'The minimum baseline length in wavelengths to be used in calibration.')
    calibration.add_argument('--allow-sidelobe-cal-sources', default = True, action = 'store_true', help = 'Allows PyFHD to calibrate on sources in the sidelobes.\nForces the beam_threshold to 0.01 in order to go down to 1\% of the beam to capture sidelobe sources during the generation of a calibration source catalog for the particular observation.')


    # Flagging Group
    flag.add_argument('-fv', '--flag-visibilities', default = False, action = 'store_true', help = 'Flag visibilities based on calculations in vis_flag')
    flag.add_argument('-fc', '--flag-calibration', default = False, action = 'store_true', help = 'Flags antennas based on calculations in vis_calibration_flag')
    flag.add_argument('--flag-freq-start', default = None, type = float, help = 'Frequency in MHz to begin the observation. Flags frequencies less than it. Replaces freq_start from FHD')
    flag.add_argument('--flag-freq-end', default = None, type = float, help = 'Frequency in MHz to end the observation. Flags frequencies greater than it. Replaces freq_end from FHD')
    flag.add_argument('--transfer-weights', type = Path, default = None, help = 'Transfer weights information from another PyFHD run.')

    # Beam Setup Group
    beam.add_argument('-b', '--recalculate-beam', default = True, action = 'store_true', help = "Forces PyFHD to redo the beam setup using PyFHD's beam setup.")
    beam.add_argument('--beam-nfreq-avg', type = int, default = 16, help = "The number of fine frequency channels to calculate a beam for, using the average of the frequencies.\nThe beam is a function of frequency, and a calculation on the finest level is most correct (beam_nfreq_avg=1).\nHowever, this is computationally difficult for most machines.")
    beam.add_argument('--psf-resolution', default = 100, type = int, help = 'Super-resolution factor of the psf in UV space. Values greater than 1 increase the resolution of the gridding kernel.')
    beam.add_argument('--beam-model-version', type = int, default = 2, help = 'A number that indicates the tile beam model calculation.\nThis is dependent on the instrument, and specific calculations are carried out in <instrument>_beam_setup_gain.\nMWA range: 0, 1 (or anything else captured in the else statement), 2\nPAPER range: 1 (or anything else captured in the else statement), 2\nHERA range: 2 (or anything else captured in the else statement)')
    beam.add_argument('--beam-clip-floor', default = True, action = 'store_true', help = 'Set to subtract the minimum non-zero value of the beam model from all pixels.')

    # Gridding Group
    gridding.add_argument('-g', '--recalculate-grid', default = False, action ='store_true', help = 'Forces PyFHD to recalculate the gridding function. Replaces grid_recalculate from FHD')
    gridding.add_argument('-map', '--recalculate-mapfn', default = False, action = 'store_true', help = 'Forces PyFHD to recalculate the mapping function. Replaces mapfn_recalculate from FHD')

    # Deconvolution Group
    deconv.add_argument('-d', '--deconvolve', default = False, action = 'store_true', help = 'Run Fast Holographic Deconvolution')
    deconv.add_argument('--max-deconvolution-components', type = int, default = 20000, help = 'The number of source components allowed to be found in fast holographic deconvolution.')
    deconv.add_argument('--dft-threshold', default = 1.0, type = float, help = 'Set equal to 1 to use the DFT approximation. When set equal to 0 the true DFT is calculated for each source.\nIt can also be explicitly set to a value that determines the accuracy of the approximation.')
    deconv.add_argument('--return-decon-visibilities', default = False, action = 'store_true', help = 'When activated degrid and export the visibilities formed from the deconvolution model')

    # Export Group
    export.add_argument('-o','--output-path', type=Path, help = "Set the output path for the current run, note a directory will still be created inside the given path", default = "./output/")
    export.add_argument('--description', type=str,help = "A more detailed description of the current task, will get applied to the output directory and logging where all output will be stored.\nBy default the date and time is used")
    export.add_argument('--export-images', help = 'Export fits files and images of the sky.', action = 'store_true',  default = True)
    export.add_argument('--cleanup', help = 'Deletes some intermediate data products that are easy to recalculate in order to save disk space', default = False, action='store_true')
    export.add_argument('--save-visibilities', default = True, action = 'store_true', help = 'Save the calibrated data visibilities, the model visibilities, and the visibility flags.')
    export.add_argument('--snapshot-healpix-export', default = True, action = 'store_true', help = 'Save model/dirty/residual/weights/variance cubes as healpix arrays, split into even and odd time samples, in preparation for epsilon.')
    export.add_argument('--pad_uv_image', type = float, default = 1.0, help = "Pad the UV image by this factor with 0's along the outside so that output images are at a higher resolution.")
    export.add_argument('--ring-radius-multi', type = float, default = 10, help = 'Sets the multiplier for the size of the rings around sources in the restored images.\nRing Radius will equal pad-uv-image * ring-radius-multi.\nTo generate restored images without rings, set ring_radius = 0.')

    # Model Group
    model.add_argument('-m', '--model-visibilities', default = False, action = 'store_true', help = 'Make visibilities for the subtraction model separately from the model used in calibration.\nThis is useful if the user sets parameters to make the subtraction model different from the model used in calibration.\nIf not set, the model used for calibration is the same as the subtraction model.')
    model.add_argument('--diffuse-model', type = Path, help = """File path to the diffuse model file.The file should contain the following:\nMODEL_ARR = A healpix map with the diffuse model. Diffuse model has units Jy/pixel unless keyword diffuse_units_kelvin is set.\n            The model can be an array of pixels, a pointer to an array of pixels, or an array of four pointers corresponding to I, Q, U, and V Stokes polarized maps.\n    NSIDE = The corresponding NSIDE parameter of the healpix map.\n HPX_INDS = The corresponding healpix indices of the model_arr.\nCOORD_SYS = (Optional) 'galactic' or 'celestial'. Specifies the coordinate system of the healpix map. GSM is in galactic coordinates, for instance. If missing, defaults to equatorial.""")
    model.add_argument('--model-catalog-file-path', type = Path, default = 'GLEAM_v2_plus_rlb2019.sav', help = 'A file containing a catalog of sources to be used to make model visibilities for subtraction.')
    model.add_argument('--allow-sidelobe-model-sources', default = True, action = 'store_true', help = 'Allows PyFHD to model sources in the sidelobes for subtraction.\nForces the beam_threshold to 0.01 in order to go down to 1\% of the beam to capture sidelobe sources during the generation of a model calibration source catalog for the particular observation.')
    
    # Simultation Group
    sim.add_argument('-sim', '--run-simulation', default = False, action = 'store_true', help = 'Run an in situ simulation, where model visibilities are made and input as the dirty visibilities (see Barry et. al. 2016 for more information on use-cases).\nIn the case where in-situ-sim-input is not provided visibilities will be made within the current PyFHD run.')
    sim.add_argument('--in-situ-sim-input', type = Path, default = None, help = 'Inputs model visibilities from a previous run, which is the preferred method since that run is independently documented.')
    sim.add_argument('--eor-vis-filepath', type = Path, default = None, help = 'A path to a file of EoR visibilities to include the EoR in the dirty input visibilities. in-situ-sim-input must be used in order to use this parameter.')
    sim.add_argument('--enhance-eor', type = float, default = 1., help = 'Input a multiplicative factor to boost the signal of the EoR in the dirty input visibilities. in-situ-sim-input must be used in order to use this parameter.')
    sim.add_argument('--sim-noise', type = Path, default = None, help = 'Add a uncorrelated thermal noise to the input dirty visibilities from a file, or create them for the run. in-situ-sim-input must be used in order to use this parameter.')
    sim.add_argument('--tile-flag-list', type = list, help = 'A string array of tile names to manually flag tiles. Note that this is an array of tile names, not tile indices!')
    sim.add_argument('--remove-sim-flags', default = False, action = 'store_true', help = 'Bypass main flagging for in situ simulations and remove all weighting to remove pfb effects and flagged channels.')
    sim.add_argument('--extra_vis_filepath', type = Path, default = None, help = 'Optionally add general visibilities to the simulation, must be a uvfits file.')

    # HEALPIX Group
    healpix.add_argument('--ps-kbinsize', type = float, default = 0.5, help = 'UV pixel size in wavelengths to grid for Healpix cube generation. Overrides ps_fov and the kpix in the obs structure if set.')
    healpix.add_argument('--ps-kspan', type = float, default = 600, help = 'UV plane dimension in wavelengths for Healpix cube generation.\nOverrides ps_dimension and ps_degpix if set.\nIf ps_kspan, ps_dimension, or ps_degpix are not set, the UV plane dimension is calculated from the FoV and the degpix from the obs structure.')
    healpix.add_argument('--restrict-hpx-inds', type = Path, default = None, help = "Only allow gridding of the output healpix cubes to include the healpix pixels specified in a file.\nThis is useful for restricting many observations to have consistent healpix pixels during integration, and saves on memory and walltime.")

    return parser

def setup_parser(options: argparse.Namespace) :
    """_summary_

    Parameters
    ----------
    options : argparse.Namespace
        _description_
    """
    print(options)