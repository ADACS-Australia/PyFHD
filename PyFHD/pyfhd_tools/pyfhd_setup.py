import numpy as np
from pathlib import Path
import configargparse
import argparse
import time
import subprocess
import logging
import shutil
from typing import Tuple

def pyfhd_parser():
    """
    The pyfhd_parser configures the argparse for PyFHD

    Returns
    -------
    configargparse.ArgumentParser
        The parser for PyFHD which contains the help strings for the terminal and Usage section of the docs.
    """

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
    parser.add_argument( '-i', '--input-path', type = Path, help = "Directory for the uvfits files and other inputs, by default it looks for a directory called input in the working directory", default = "./input/")
    parser.add_argument('-r', '--recalculate-all', action='store_true', help = 'Forces PyFHD to recalculate all values. This will ignore values set for recalculate-grid, recalculate-beam, recalculate-mapfn as it will set all of them to True')
    parser.add_argument('-s', '--silent', default = False, action = 'store_true', help = 'This PyFHD stops all output to the terminal except in the case of an error and/or exception')
    parser.add_argument('-l', '--disable-log', action = 'store_true', help = 'Logging in a log file is enabled by default, set to False in the config or use this argument to disable the log file.')
    parser.add_argument('--instrument', type = str, default = 'mwa', choices = ['mwa'], help = 'Set the instrument used for the FHD run, currently only MWA is supported')
    parser.add_argument('--dimension', type = int, default = 2048, help = 'The number of pixels in the UV plane along one axis.')
    parser.add_argument('--elements', type = int, default = 2048, help = 'The number of pixels in the UV plane along the other axis.')
    parser.add_argument('--kbinsize', type = float, default = 0.5, help = "Size of UV pixels in wavelengths. Given a defined number of pixels in dimension, this sets the UV space extent. This will supersede degpix if dimension is also set.")
    parser.add_argument('--FoV', '--fov', type = float, default = None, help = 'A proxy for the field of view in degrees. FoV is actually used to determine kbinsize, which will be set to !RaDeg/FoV.\nThis means that the pixel size at phase center times dimension is approximately equal to FoV, which is not equal to the actual field of view owing to larger pixel sizes away from phase center.\nIf set to 0, then kbinsize determines the UV resolution.')
    parser.add_argument('--deproject-w-term', type = float, default = None, help = 'Enables the function for simple_deproject_w_term and uses the parameter value for the direction value in the function')
    parser.add_argument('--conserve-memory', default = False, action = 'store_true', help = 'Optionally split many loops into chunks in the case of high memory usage.')
    parser.add_argument('--memory-threshold', type = int, default = 1e8, help = 'Set a memory threshold for each chunk in set in bytes. By default it is set at ~100MB')
    parser.add_argument('--n-avg', type = int, default = 2, help = 'Number of frequencies to average over to smooth the frequency band.')
    parser.add_argument('--min-baseline', type = float, default = 1.0, help = 'The minimum baseline length in wavelengths to include in the analysis')
    parser.add_argument('--n-pol', type = int, default = 2, choices = [0, 2, 4], help = 'Set number of polarizations to use (XX, YY versus XX, YY, XY, YX).')

    # Calibration Group
    calibration.add_argument('-cv', '--calibrate-visibilities', default = True, action = 'store_true', help = 'Turn on the calibration of the visibilities. If turned on, calibration of the dirty, modelling, and subtraction to make a residual occurs. Otherwise, none of these occur and an uncalibrated dirty cube is output.')
    calibration.add_argument('--diffuse-calibrate', type = Path, help = 'Path to a file containing a map/model of the diffuse in which to calibrate on.\nThe map/model undergoes a DFT for every pixel, and the contribution from every pixel is added to the model visibilities from which to calibrate on.\nIf no diffuse_model is specified, then this map/model is used for the subtraction model as well. See diffuse_model for information about the formatting of the file.')
    calibration.add_argument('--transfer-calibration', type = Path, help = 'The file path of a calibration to be read-in, if you give a directory PyFHD expects there to be a file called <obs_id>_cal.hdf5 using the same observation as you plan to process.')
    calibration.add_argument('--calibration-catalog-file-path', type = Path, default = None, help = 'The file path to the desired source catalog to be used for calibration')
    calibration.add_argument('--return-cal-visibilities', default = True, action = 'store_true', help = "Saves the visibilities created for calibration for use in the model.\nIf model_visibilities is set to False, then the calibration model visibilities and the model visibilities will be the same if return_cal_visibilities is set.\nIf model_visibilities is set to True, then any new modelling (of more sources, diffuse, etc.) will take place and the visibilities created for the calibration model will be added.\nIf n_pol = 4 (full pol mode), return_cal_visibilites must be set because the visibilites are required for calculating the mixing angle between Q and U.")
    calibration.add_argument('--cal-stop', default = False, action = 'store_true', help = 'Stops the code right after calibration, and saves unflagged model visibilities along with the obs structure in a folder called cal_prerun in the PyFHD file structure.\nThis allows for post-processing calibration steps like multi-day averaging, but still has all of the needed information for minimal reprocessing to get to the calibration step.\nTo run a post-processing run, see keywords model_transfer and transfer_psf')
    calibration.add_argument('--transfer-model-uv', type = Path, default = None, help = "A path to save a model uv array.\nIf it's a file that doesnt exist then vis_calibrate will create one for this run, otherwise if the file exists PyFHD will read it in for this run.\nReplaces model_uv_transfer")
    calibration.add_argument('--min-cal-baseline', type = float, default = 50.0, help = 'The minimum baseline length in wavelengths to be used in calibration.')
    calibration.add_argument('--allow-sidelobe-cal-sources', default = True, action = 'store_true', help = 'Allows PyFHD to calibrate on sources in the sidelobes.\nForces the beam_threshold to 0.01 in order to go down to 1%% of the beam to capture sidelobe sources during the generation of a calibration source catalog for the particular observation.')
    calibration.add_argument('--cable-bandpass-fit', default = True, action = 'store_true', help = 'Average the calibration solutions across tiles within a cable grouping for the particular instrument.\nDependency: instrument_config/<instrument>_cable_length.txt')
    calibration.add_argument('--cal-bp-transfer', type = Path, default = None, help = 'Use a saved bandpass for bandpass calibration. Read in the specified file with calfits format greatly preferred.')
    calibration.add_argument('--calibration-polyfit', default = True, action = 'store_true', help = 'Calculates a polynomial fit across the frequency band for the gain, and allows a cable reflection to be fit.\nThe orders of the polynomial fit are determined by cal_phase_degree_fit and cal_amp_degree_fit.\nIf unset, no polynomial fit or cable reflection fit are used.')
    calibration.add_argument('--cal-amp-degree-fit', default = 2, type = int, help = "The nth order of the polynomial fit over the whole band to create calibration solutions for the amplitude of the gain.\nSetting it to 0 gives a 0th order polynomial fit (one number for the whole band),\n1 gives a 1st order polynomial fit (linear fit),\n2 gives a 2nd order polynomial fit (quadratic),\nn gives nth order polynomial fit.\nRequires calibration_polyfit to be enabled.")
    calibration.add_argument('--cal-phase-degree-fit', default = 1, type = int, help = "The nth order of the polynomial fit over the whole band to create calibration solutions for the phase of the gain.\nSetting it to 0 gives a 0th order polynomial fit (one number for the whole band),\n1 gives a 1st order polynomial fit (linear fit),\n2 gives a 2nd order polynomial fit (quadratic),\nn gives nth order polynomial fit.\nRequires calibration_polyfit to be enabled.")
    calibration.add_argument('--cal-reflection-hyperresolve', default = True, action = 'store_true', help = 'Hyperresolve and fit residual gains using nominal reflection modes (calculated from cal_reflection_mode_delay or cal_reflection_mode_theory),\nproducing a finetuned mode fit, amplitude, and phase.\nWill be ignored if cal_reflection_mode_file is set because it is assumed that a file read-in contains mode/amp/phase to use.')
    calibration.add_argument('--cal-reflection-mode-theory', default = 150, type = float, help = 'Calculate theoretical cable reflection modes given the velocity and length data stored in a config file named <instrument>_cable_length.txt.\nFile must have a header line and at least five columns (tile index, tile name, cable length, cable velocity factor, logic on whether to fit (1) or not (0)).\nCan set it to positive/negative cable lengths (see cal_mode_fit) to include/exclude certain cable types.')
    calibration.add_argument('--cal-reflection-mode-delay', default = False, action = 'store_true', help = 'Calculate cable reflection modes by Fourier transforming the residual gains, removing modes contaminated by frequency flagging, and choosing the maximum mode.')
    calibration.add_argument('--cal-reflection-mode-file', default = None, type = Path, help = 'Use predetermined cable reflection parameters (mode, amplitude, and phase) in the calibration solutions from a file.\nThe specified format of the text file must have one header line and eleven columns:\ntile index\ntile name\ncable length\ncable velocity factor\nlogic on whether to fit (1) or not (0)\nmode for X\namplitude for X\nphase for X\nmode for Y\namplitude for Y\nphase for Y')
    calibration.add_argument('--vis-baseline-hist', default = True, action = 'store_true', help = 'Calculates the vis_baseline_hist dictionary containing the visibility resolution ratio average and standard deviation')
    calibration.add_argument('--bandpass-calibrate', default = True, action = 'store_true', help = 'Calculates a bandpass.\nThis is an average of tiles by frequency by polarization (default), beamformer-to-LNA cable types by frequency by polarization (see cable_bandpass_fit),\nor over the whole season by pointing by by cable type by frequency by polarization via a read-in file (see saved_run_bp).\nIf unset, no by-frequency bandpass is used')
    calibration.add_argument('--calibration-flag-iterate', default = 0, type = int, help = 'Number of times to repeat calibration in order to better identify and flag bad antennas so as to exclude them from the final result.')

    # Flagging Group
    flag.add_argument('-fv', '--flag-visibilities', default = False, action = 'store_true', help = 'Flag visibilities based on calculations in vis_flag')
    flag.add_argument('-fc', '--flag-calibration', default = False, action = 'store_true', help = 'Flags antennas based on calculations in vis_calibration_flag')
    flag.add_argument('--flag-freq-start', default = None, type = float, help = 'Frequency in MHz to begin the observation. Flags frequencies less than it. Replaces freq_start from FHD')
    flag.add_argument('--flag-freq-end', default = None, type = float, help = 'Frequency in MHz to end the observation. Flags frequencies greater than it. Replaces freq_end from FHD')
    flag.add_argument('--transfer-weights', type = Path, default = None, help = 'Transfer weights information from another PyFHD run.')
    flag.add_argument('--time-cut', type = list, default = None, help = 'Seconds to cut (rounded up to next time integration step) from the beginning of the observation. Can also specify a negative time to cut off the end of the observation. Specify a vector to cut at both the start and end.')

    # Beam Setup Group
    beam.add_argument('-b', '--recalculate-beam', default = True, action = 'store_true', help = "Forces PyFHD to redo the beam setup using PyFHD's beam setup.")
    beam.add_argument('--beam-nfreq-avg', type = int, default = 16, help = "The number of fine frequency channels to calculate a beam for, using the average of the frequencies.\nThe beam is a function of frequency, and a calculation on the finest level is most correct (beam_nfreq_avg=1).\nHowever, this is computationally difficult for most machines.")
    beam.add_argument('--psf-resolution', default = 100, type = int, help = 'Super-resolution factor of the psf in UV space. Values greater than 1 increase the resolution of the gridding kernel.')
    beam.add_argument('--beam-model-version', type = int, default = 2, help = 'A number that indicates the tile beam model calculation.\nThis is dependent on the instrument, and specific calculations are carried out in <instrument>_beam_setup_gain.\nMWA range: 0, 1 (or anything else captured in the else statement), 2\nPAPER range: 1 (or anything else captured in the else statement), 2\nHERA range: 2 (or anything else captured in the else statement)')
    beam.add_argument('--beam-clip-floor', default = True, action = 'store_true', help = 'Set to subtract the minimum non-zero value of the beam model from all pixels.')
    beam.add_argument('--interpolate-kernel', default = True, action = 'store_true', help = "Use interpolation of the gridding kernel while gridding and degridding, rather than selecting the closest super-resolution kernel.")
    beam.add_argument('--dipole-mutual-coupling-factor', default = True, action = 'store_true', help = 'Allows a modification to the beam as a result of mutual coupling between dipoles calculated in mwa_dipole_mutual_coupling (See Sutinjo 2015 for more details).')
    beam.add_argument('--beam-offset-time', type = float, default = 56, help = "Calculate the beam at a specific time within the observation. 0 seconds indicates the start of the observation, and the # of seconds in an observation indicates the end of the observation.")

    # Gridding Group
    gridding.add_argument('-g', '--recalculate-grid', default = False, action ='store_true', help = 'Forces PyFHD to recalculate the gridding function. Replaces grid_recalculate from FHD')
    gridding.add_argument('-map', '--recalculate-mapfn', default = False, action = 'store_true', help = 'Forces PyFHD to recalculate the mapping function. Replaces mapfn_recalculate from FHD')
    gridding.add_argument('--image-filter', default = 'filter_uv_uniform', type = str, choices = ['filter_uv_uniform', 'filter_uv_hanning', 'filter_uv_natural', 'filter_uv_radial', 'filter_uv_tapered_uniform', 'filter_uv_optimal'], help = 'Weighting filter to be applyed to resulting snapshot images and fits files. Replaces image_filter_fn from FHD')

    # Deconvolution Group
    deconv.add_argument('-d', '--deconvolve', default = False, action = 'store_true', help = 'Run Fast Holographic Deconvolution')
    deconv.add_argument('--max-deconvolution-components', type = int, default = 20000, help = 'The number of source components allowed to be found in fast holographic deconvolution.')
    deconv.add_argument('--dft-threshold', default = False, action = 'store_true', help = 'Set to True to use the DFT approximation. When set equal to 0 the true DFT is calculated for each source.\nIt can also be explicitly set to a value that determines the accuracy of the approximation.')
    deconv.add_argument('--return-decon-visibilities', default = False, action = 'store_true', help = 'When activated degrid and export the visibilities formed from the deconvolution model')
    deconv.add_argument('--deconvolution-filter', default = 'filter_uv_uniform', type = str, choices = ['filter_uv_uniform', 'filter_uv_hanning', 'filter_uv_natural', 'filter_uv_radial', 'filter_uv_tapered_uniform', 'filter_uv_optimal'], help = 'Filter applied to images from deconvolution.')
    deconv.add_argument('--smooth-width', default = 32, type = int, help = 'Integer equal to the size of the region to smooth when filtering out large-scale background fluctuations.')
    deconv.add_argument('--filter-background', default = True, action = 'store_true', help = 'Filters out large-scale background fluctuations before deconvolving point sources.')

    # Export Group
    export.add_argument('-o','--output-path', type=Path, help = "Set the output path for the current run, note a directory will still be created inside the given path", default = "./output/")
    export.add_argument('--description', type=str, default = None, help = "A more detailed description of the current task, will get applied to the output directory and logging where all output will be stored.\nBy default the date and time is used")
    export.add_argument('--export-images', help = 'Export fits files and images of the sky.', action = 'store_true',  default = True)
    export.add_argument('--cleanup', help = 'Deletes some intermediate data products that are easy to recalculate in order to save disk space', default = False, action='store_true')
    export.add_argument('--save-visibilities', default = True, action = 'store_true', help = 'Save the calibrated data visibilities, the model visibilities, and the visibility flags.')
    export.add_argument('--snapshot-healpix-export', default = True, action = 'store_true', help = 'Save model/dirty/residual/weights/variance cubes as healpix arrays, split into even and odd time samples, in preparation for epsilon.')
    export.add_argument('--pad-uv-image', type = float, default = 1.0, help = "Pad the UV image by this factor with 0's along the outside so that output images are at a higher resolution.")
    export.add_argument('--ring-radius-multi', type = float, default = 10, help = 'Sets the multiplier for the size of the rings around sources in the restored images.\nRing Radius will equal pad-uv-image * ring-radius-multi.\nTo generate restored images without rings, set ring_radius = 0.')

    # Model Group
    model.add_argument('-m', '--model-visibilities', default = False, action = 'store_true', help = 'Make visibilities for the subtraction model separately from the model used in calibration.\nThis is useful if the user sets parameters to make the subtraction model different from the model used in calibration.\nIf not set, the model used for calibration is the same as the subtraction model.')
    model.add_argument('--diffuse-model', type = Path, help = """File path to the diffuse model file.The file should contain the following: \nMODEL_ARR = A healpix map with the diffuse model. Diffuse model has units Jy/pixel unless keyword diffuse_units_kelvin is set.\n            The model can be an array of pixels, a pointer to an array of pixels, or an array of four pointers corresponding to I, Q, U, and V Stokes polarized maps.\n    NSIDE = The corresponding NSIDE parameter of the healpix map.\n HPX_INDS = The corresponding healpix indices of the model_arr.\nCOORD_SYS = (Optional) 'galactic' or 'celestial'. Specifies the coordinate system of the healpix map. GSM is in galactic coordinates, for instance. If missing, defaults to equatorial.""")
    model.add_argument('--model-catalog-file-path', type = Path, default = None, help = 'A file containing a catalog of sources to be used to make model visibilities for subtraction.')
    model.add_argument('--allow-sidelobe-model-sources', default = False, action = 'store_true', help = 'Allows PyFHD to model sources in the sidelobes for subtraction.\nForces the beam_threshold to 0.01 in order to go down to 1%% of the beam to capture sidelobe sources during the generation of a model calibration source catalog for the particular observation.')
    
    # Simultation Group
    sim.add_argument('-sim', '--run-simulation', default = False, action = 'store_true', help = 'Run an in situ simulation, where model visibilities are made and input as the dirty visibilities (see Barry et. al. 2016 for more information on use-cases).\nIn the case where in-situ-sim-input is not provided visibilities will be made within the current PyFHD run.')
    sim.add_argument('--in-situ-sim-input', type = Path, default = None, help = 'Inputs model visibilities from a previous run, which is the preferred method since that run is independently documented.')
    sim.add_argument('--eor-vis-filepath', type = Path, default = None, help = 'A path to a file of EoR visibilities to include the EoR in the dirty input visibilities. in-situ-sim-input must be used in order to use this parameter. Replaces eor_savefile from FHD')
    sim.add_argument('--enhance-eor', type = float, default = 1., help = 'Input a multiplicative factor to boost the signal of the EoR in the dirty input visibilities. in-situ-sim-input must be used in order to use this parameter.')
    sim.add_argument('--sim-noise', type = Path, default = None, help = 'Add a uncorrelated thermal noise to the input dirty visibilities from a file, or create them for the run. in-situ-sim-input must be used in order to use this parameter.')
    sim.add_argument('--tile-flag-list', type = list, help = 'A string array of tile names to manually flag tiles. Note that this is an array of tile names, not tile indices!')
    sim.add_argument('--remove-sim-flags', default = False, action = 'store_true', help = 'Bypass main flagging for in situ simulations and remove all weighting to remove pfb effects and flagged channels.')
    sim.add_argument('--extra_vis_filepath', type = Path, default = None, help = 'Optionally add general visibilities to the simulation, must be a uvfits file.')

    # HEALPIX Group
    healpix.add_argument('--ps-kbinsize', type = float, default = 0.5, help = 'UV pixel size in wavelengths to grid for Healpix cube generation. Overrides ps_fov and the kpix in the obs structure if set.')
    healpix.add_argument('--ps-kspan', type = float, default = 600, help = 'UV plane dimension in wavelengths for Healpix cube generation.\nOverrides ps_dimension and ps_degpix if set.\nIf ps_kspan, ps_dimension, or ps_degpix are not set, the UV plane dimension is calculated from the FoV and the degpix from the obs structure.')
    healpix.add_argument('--restrict-hpx-inds', type = Path, default = None, help = "Only allow gridding of the output healpix cubes to include the healpix pixels specified in a file.\nThis is useful for restricting many observations to have consistent healpix pixels during integration, and saves on memory and walltime.")
    healpix.add_argument('--split-ps-export', default = True, action = 'store_true', help = 'Split up the Healpix outputs into even and odd time samples.\nThis is essential to propogating errors in εppsilon.\nRequires more than one time sample.')

    return parser

def _check_file_exists(config : dict, key : str) -> int:
    """
    Helper function to check if the key is not None and if it isn't None, then if file exists given from the config.

    Parameters
    ----------
    config : dict
        Should be the pyfhd_config
    key : str
        The keyword in the config we are checking.

    Returns
    -------
    int
        Will return 0 if there is no error, 1 if there is.
    """
    if config[key]:
        if not Path(config[key]).exists():
            logging.error("{} has been enabled with a path that doesn't exist, check the path.".format(key))
            return 1
    return 0

def pyfhd_setup(options : argparse.Namespace) -> Tuple[dict, logging.RootLogger]:
    """
    Check for any incompatibilities among the options given for starting the PyFHD pipeline as some options
    do conflict with each other or have dependencies on other options. This function should catch all of those
    potential errors and exit the program with errors once these have been found before any output. This function
    should also replace fhd_setup

    Parameters
    ----------
    options : argparse.Namespace
        The parsed argparse object.
    
    Returns
    -------
    pyfhd_config : dict
        The configuration dictionary for PyFHD containing all the options.
    logger : logging.RootLogger
        The logger with the appropriate handlers added.
    """
    # Keep track of the errors and warnings.
    errors = 0
    warnings = 0
    pyfhd_config = vars(options)
    # Start the logger
    logger = pyfhd_logger(pyfhd_config)
    # Check input_path exists and obs_id uvfits and metafits files exist (Error)
    if not pyfhd_config['input_path'].exists():
        logger.error("{} doesn't exist, please check your input path".format(options.input_path))
        errors += 1
    obs_uvfits_path = Path(pyfhd_config['input_path'], pyfhd_config['obs_id'] + '.uvfits')
    obs_metafits_path = Path(pyfhd_config['input_path'], pyfhd_config['obs_id'] + '.metafits')
    if not obs_uvfits_path.exists():
        logger.error("{} doesn't exist, please check your input path".format(obs_uvfits_path))
        errors += 1
    if not obs_metafits_path.exists():
        logger.error("{} doesn't exist, please check your input path".format(obs_metafits_path))
        errors += 1
    
    # Force PyFHD to recalculate the beam, gridding and mapping functions
    if pyfhd_config['recalculate_all']:
        pyfhd_config['recalculate_beam'] = True
        pyfhd_config['recalculate_grid'] = True
        pyfhd_config['recalculate_mapfn'] = True
    logger.info('Recalculate All options has been enabled, the beam, gridding and map function will be recalculated')

    # Check if recalculate_mapfn has been enabled and recalculate_grids disabled, if so enable recaluclate_grid (Warning)
    if pyfhd_config['recalculate_mapfn'] and not pyfhd_config['recalculate_grid']:
        pyfhd_config['recalculate_grid'] = True
        logger.warning('The grid has to be recalculated in order for the mapping function to be recalculated. Grid recalculation has now been enabled.') 
        warnings += 1

    # If both mapping function and healpi export are on save the visibilities (Warning)
    if pyfhd_config['recalculate_mapfn'] and pyfhd_config['snapshot_healpix_export'] and not pyfhd_config['save_visibilities']:
        pyfhd_config['save_visibilities'] = True
        logger.warning("If the mapping function is being recalculated and we're exporting healpix we should also save the visibilities that created them.")
        warnings += 1

    if pyfhd_config['beam_offset_time'] < 0:
        pyfhd_config['beam_offset_time'] = 0
        logger.warning("You set the offset time to less than 0, it was reset to 0.")
        warnings += 1

    # If cable_bandpass_fit has been enabled an instrument text file should also exist. (Error)
    if pyfhd_config['cable_bandpass_fit']:
        if not Path(pyfhd_config['input_path'], 'instrument_config', 'mwa_cable_length' + '.txt').exists():
            logging.error('Cable bandpass fit has been enabled but the required text file is missing')
            errors += 1
    
    # cal_bp_transfer when enabled should point to a file with a saved bandpass (Error)
    errors += _check_file_exists(pyfhd_config, 'cal_bp_transfer')
    
    # If cal_amp_degree_fit or cal_phase_degree_fit have ben set but calibration_polyfit isn't warn the user (Warning)
    if (pyfhd_config['cal_amp_degree_fit'] or pyfhd_config['cal_phase_degree_fit'] or pyfhd_config['cal_reflection_mode_theory'] or pyfhd_config['cal_reflection_mode_delay']) \
        and not pyfhd_config['calibration_polyfit']:
        logger.warning('cal_amp_degree_fit and/or cal_amp_phase_fit have been set but calibration_polyfit has been disabled.')
        warnings += 1
    
    # cal_reflection_hyperresolve gets ignored when cal_reflection_mode_file is set (Warning)
    if pyfhd_config['cal_reflection_hyperresolve'] and pyfhd_config['cal_reflection_mode_file']:
        logging.warning("cal_reflection_hyperresolve and cal_reflection_mode_file have both been turned on, cal_reflection_mode_file will be prioritised.")
        pyfhd_config['cal_reflection_hyperresolve'] = False
        warnings += 1

    # cal_reflection_mode_file depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'cal_reflection_mode_file')

    # cal_reflection_mode_theory and cal_reflection_mode_delay cannot be on at the same time, prioritise mode_theory (Warning)
    if pyfhd_config['cal_reflection_mode_theory'] and pyfhd_config['cal_reflection_mode_delay']: 
        logging.warning('Both cal_reflection_mode_theory and cal_reflection_mode_delay have been enabled, prioritising cal_reflection_mode_theory')
        pyfhd_config['cal_reflection_mode_delay'] = False
        warnings += 1
    
    # diffuse_calibrate depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'diffuse_calibrate')

    # calibration_catalog_file_path depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'calibration_catalog_file_path')

    # transfer_calibration depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'transfer_calibration')

    # transfer_model_uv depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'transfer_model_uv')

    # transfer-weights depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'transfer_weights')

    # smooth-width depends on filter_background (Warning)
    if not pyfhd_config['filter_background'] and pyfhd_config['smooth_width']:
        logger.warning('filter_background must be True for smooth_width to have any effect')
        warnings += 1
    
    # diffuse-model depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'diffuse_model')

    # model_catalog_file_path depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'model_catalog_file_path')

    # allow_sidelobe_model_sources depends on model_visibilities (Error)
    if pyfhd_config['allow_sidelobe_model_sources'] and not pyfhd_config['model_visibilities']:
        logger.error("allow_sidelobe_model_sources shouldn't be True when model_visibilities is not, check if you meant to turn on model_visibilities")
        errors += 1

    # Entirety of Simulation Group depends on run-simulation (Error)
    if not pyfhd_config['run_simulation'] and \
        (pyfhd_config['in_situ_sim_input'] or \
         pyfhd_config['eor_vis_filepath'] or \
         pyfhd_config['sim_noise']):
        logger.error("run_simulation should be True if you're planning on running any type of simulation and therefore using in_situ_sim_input, eor_vis_filepath or sim_noise shouldn't be used when run_simulation is False")
        errors += 1

    # in-situ-sim-input depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'in_situ_sim_input')

    # eor_vis_filepath depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'eor_vis_filepath')

    # enhance_eor depends on eor_vis_filepath when its not 1
    if pyfhd_config['enhance_eor'] > 1 and pyfhd_config['eor_vis_filepath']:
        logger.error('enhance_eor is only used when importing general visibilities for a simulation, it should stay as 1 when eor_vis_filepath is not being used')
        errors += 1
    
    # sim_noise depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'sim_noise')

    # Restrict_hpx_inds depends on a file (Error)
    errors += _check_file_exists(pyfhd_config, 'restrict_hpx_inds')
    
    pyfhd_config['ring_radius'] =  pyfhd_config['pad_uv_image'] * pyfhd_config['ring_radius_multi']

    # If there are any errors exit the program.
    if errors:
        logger.error('{} errors detected, check the log above to see the errors, stopping PyFHD now'.format(errors))
        # Close the handlers in the log
        for handler in logger.handlers:
            handler.close()
        exit()

    if warnings:
        logger.warning('{} warnings detected, check the log above, these cause some weird behavior'.format(warnings))

    logger.info('Input validated, starting PyFHD run now')

    return pyfhd_config, logger

def pyfhd_logger(pyfhd_config: dict) -> logging.RootLogger:
    '''
    Creates the the logger for PyFHD. If silent is True in the pyfhd_config then
    the StreamHandler won't be added to logger meaning there will be no terminal output
    even if logger is called later. If diable_log is True then the FileHandler won't be added
    to the logger preventing the creation fo the log file meaning subsequent calls to the logger
    will not add to or create a log file.

    Parameters
    ----------
    pyfhd_config : dict
        The options from the argparse in a dictionary

    Returns
    -------
    logger : logging.RootLogger
        The logger with the appropriate handlers added.
    '''
    # Get the time, Git commit and setup the name of the output directory
    run_time = time.localtime()
    stdout_time = time.strftime("%c", run_time)
    log_time = time.strftime("%H_%M_%S_%d_%m_%Y", run_time)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text = True).stdout
    if pyfhd_config['description'] is None:
        log_name = "pyfhd_" + log_time
    else:
        log_name = "pyfhd_" + pyfhd_config['description'].replace(' ', '_').lower() + log_time
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
        
        Validating your input...""".format(commit.replace('\n',''), stdout_time, pyfhd_config['obs_id'])

    # Setup logging
    log_string = ""
    for line in start_string.split('\n'):
        log_string += line.lstrip().replace('_', ' ').replace('|    ', '').replace('|', '') +'\n'
    # Start the PyFHD run
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create the logging for the temrinal
    if not pyfhd_config['silent']:
        log_terminal = logging.StreamHandler()
        log_terminal.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(log_terminal)

    # Create the output directory
    output_dir = Path(pyfhd_config['output_path'], log_name)
    if Path.is_dir(output_dir):
        raise OSError("You have a directory inside your specified output directory already named by this date and time...Probably not a time travel paradox right...?")
    else:
        Path.mkdir(output_dir)

    # Create the logger for the file
    if not pyfhd_config['disable_log']:
        log_file = logging.FileHandler(Path(output_dir, log_name + '.log'))
        log_file.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(log_file)
    
    # Show that start message in the terminal and/or log file, unless both are turned off.
    logger.info(log_string)
    if not pyfhd_config['silent']:
        log_terminal.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n\t%(message)s', datefmt = '%Y-%m-%d %H:%M:%S'))
    if not pyfhd_config['disable_log']:
        log_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:\n\t%(message)s', datefmt = '%Y-%m-%d %H:%M:%S'))

    # Copy the Configuration File if it exists to the output directory
    if pyfhd_config['config_file'] is None:
        shutil.copy('pyfhd.yaml', Path(output_dir, log_name + '.yaml'))
    else:
        shutil.copy(pyfhd_config['config_file'], Path(output_dir, log_name + '.yaml'))
    
    logger.info('Logging and configuration file created and copied to here: {}'.format(Path(output_dir).resolve()))

    return logger
