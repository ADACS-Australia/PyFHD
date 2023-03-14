import subprocess
import pathlib
import os
import importlib_resources
import shutil
import logging
import time
import io
from PyFHD.data_setup.obs import create_obs
from PyFHD.data_setup.uvfits import extract_header, create_params
from PyFHD.source_modeling.vis_model_transfer import import_vis_model_from_uvfits, convert_vis_model_arr_to_sav, flag_model_visibilities


def run_command(cmd : str, dry_run=False):
    """
    Runs the command string `cmd` using `subprocess.run`. Returns any text output to stdout

    Parameters
    ----------
    cmd : str
         The command to run on the command line
    dry_run : bool
         If True, don't actually run the command. Defaults to False (so defaults to running the command)
    """

    if dry_run:
        stdout = "This was a dry run, not launching IDL code\n"
    else:
        stdout = subprocess.run(cmd.split(), stdout=subprocess.PIPE,
                            text = True).stdout

    return stdout

def convert_argdict_to_pro(pyfhd_config: str, output_dir: str):
    """
    Given a dictionary of PyFHD parsed arguments, convert them into an IDL
    file `pyfhd_config.pro`. Edits any names as necessary to run correctly,
    including:
    - IDL will not allow '-' in variable names, so replace with '_'
    - Booleans are converted into integers
    - None is replaced with !NULL

    Parameters
    ----------
    pyfhd_config : dict
        The options from the argparse in a dictionary
    output_dir: str
        Where to save the .pro file

    """

    ##IDL will not allow '-' in variable names, so replace them all with underscore
    ##(le sigh)
    ##.keys() freaks out if the keys change during looping, so make a copy into
    ##a list
    old_keys = list(pyfhd_config.keys())

    ##New dict to hold the IDL style args
    idl_dict = {}

    for old_key in old_keys:
        if '-' in old_key:
            new_key = old_key.replace('-', '_')
        else:
            new_key = old_key

        idl_dict[new_key] = pyfhd_config[old_key]

    ##For all the arguments, loop through and write to text file
    with open(f"{output_dir}/pyfhd_config.pro", 'w') as pro_file:
        pro_file.write("PRO pyfhd_config,extra\n\n")

        ##Convert the yaml-python types into IDL compatible variables
        for key, value in idl_dict.items():

            ##some keywords shouldn't be passed to FHD, so skip them
            if key in ['top_level_dir', 'log_name', 'commit', 'config_file']:
                pass

            elif key == "output_path":
                pro_file.write(f"  output_directory='{value}/{idl_dict['top_level_dir']}'\n")
            elif key == 'import_model_uvfits':
                pro_file.write(f"  model_transfer='{idl_dict['output_path']}/{idl_dict['top_level_dir']}/model_vis'\n")
            elif type(value) == str:
                pro_file.write(f"  {key}='{value}'\n")
            elif type(value) == bool:
                pro_file.write(f"  {key}={int(value)}\n")
            elif value is None:
                pro_file.write(f"  {key}=!NULL\n")
            elif type(value) == pathlib.PosixPath:
                pro_file.write(f"  {key}='{str(value)}'\n")
            else:
                pro_file.write(f"  {key}={value}\n")

        ##This is some native FHD talk to ensure there are no duplicate
        ##keywords
        pro_file.write("\n  extra=var_bundle(level=0) ; first gather all variables set in the top-level wrapper\n")
        pro_file.write("  extra=var_bundle(level=1) ; next gather all variables set in this file, removing any duplicates.\n")
        pro_file.write("END\n")

def _write_user_IDL_variable_lines(pyfhd_config : dict, outfile : io.TextIOWrapper):
    """If the user has set the --IDL_variables_file argument, read the contents
    of that file and write to `outfile`

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    outfile : io.TextIOWrapper
        An opened text file wrapper for lines to be written into

    """
    if pyfhd_config['IDL_variables_file']:

        outfile.write(f"    ; user defined IDL variables input via file {pyfhd_config['IDL_variables_file']}\n")

        with open(pyfhd_config['IDL_variables_file'], 'r') as input:
            lines = input.read().split('\n')
            ##Put an indent on the front of everything, fits into the .pro
            ##file clearly that way
            for line in lines:
                outfile.write('    ' + line + '\n')

        outfile.write('\n')

def write_run_FHD_calibration_pro(pyfhd_config : dict,
                                  output_dir : str):
    """
    Write the top level run_fhd_calibration_only.pro file into the appropriate
    `output_dir`, using arguments found in the dict `pyfhd_config`.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    output_dir: str
        Where to save run_fhd_calibration_only.pro

    """

    with open(f"{output_dir}/run_fhd_calibration_only.pro", 'w') as outfile:
        outfile.write("pro run_fhd_calibration_only\n")
        outfile.write("\n")
        ##First of all, explicitly set some keywords that are used by the
        ##`fhd_path_setup` function below. Have to do this first, as the
        ##keywords set by `fhd_path_setup` need to be bundled into the `extra`
        ##structure  
        outfile.write("    ; Keywords\n")
        outfile.write(f"    obs_id='{pyfhd_config['obs_id']}'\n")
        
        vis_file_list = f"{pyfhd_config['input_path']/pyfhd_config['obs_id']}.uvfits"
        outfile.write(f'    vis_file_list="{vis_file_list}"\n')
        outfile.write(f"    output_directory='{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}'\n")
        outfile.write(f"    version='{pyfhd_config['top_level_dir']}'\n")
        outfile.write("\n")

        ##This sets up a bunch of paths so IDL-FHD knows where to output thigns            
        outfile.write("    ; Directory setup\n")
        outfile.write("    fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory)\n")
        outfile.write("    healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version)\n")
        outfile.write("\n")

        ##If we are conserving memory, pass that on to FHD
        if pyfhd_config['conserve_memory']:
            outfile.write("    ; user has asked to conserve memory\n")
            outfile.write(f"    conserve_memory={pyfhd_config['memory_threshold']}\n\n")

        ##If the user has passed the `--IDL_variables_file` arg, this will write
        ##out the contents to the master file with the same indent level
        _write_user_IDL_variable_lines(pyfhd_config, outfile)


        ##This reads in all the other keywords that have been written to
        ##pyfhd_config.pro, and bundles them into a structure `extra` along
        ##with any other keywords already set
        outfile.write("    ; Set global defaults and bundle all the variables into a structure.\n")
        outfile.write("    ; Any keywords set on the command line or in the top-level wrapper will supercede these defaults\n")
        outfile.write("    pyfhd_config,extra\n")
        outfile.write("\n")

        ##print out all the variables so we can check them later
        outfile.write("\n")
        outfile.write("    ; print all the keywords that are now set\n")
        outfile.write('    print,""\n')
        outfile.write('    print,"Keywords set in wrapper:"\n')
        outfile.write("    print,structure_to_text(extra)\n")
        outfile.write("\n")

        ##This finally actually calls FHD
        outfile.write('    print,""\n')
        outfile.write("    ; this runs FHD proper\n")
        outfile.write("    general_calibration_only,_Extra=extra\n")
        outfile.write("    end\n")


def run_IDL_calibration_only(pyfhd_config : dict,
                             logger : logging.RootLogger):
    """Run the IDL FHD code up to and including calibration, based on all the
    keywords in `pyfhd_config`. The function assumes IDL FHD is installed with
    all the necessary paths and environment variables defined. To run IDL,
    this function converts all contents of `pyfhd_config` into a file
    called `pyfhd_config.pro`. It then writes a wrapper file
    `run_fhd_calibration_only.pro`, which not only reads from `pyfhd_config.pro`,
    but two other template .pro files (which are bundled with the repo/installation.
    Finally, it uses `subprocess` to call `idl`. The function returns a path
    to where you will find your outputs.
    
    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.RootLogger
        The logger to output info and errors to

    Returns
    -------
    IDL_output_dir : str
        Where the IDL outputs will be stored; this is made to be a subdir of
        `output_dir` so everything gets shoved in one location. IDL will prepend
        `fhd` onto the front the subdir.

    """

    output_dir = f"{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}"
    
    ##If reading model visibilities from a uvfits file, grab the visis
    if pyfhd_config['import_model_uvfits']:
        
        ##Need the `obs` struct to be initialised to use `extract_visibilities`,
        ##which is used by import_vis_model_from_uvfits to grab the 
        pyfhd_header, params_data, _ = extract_header(pyfhd_config, logger)
        params = create_params(pyfhd_header, params_data, logger)
        obs = create_obs(pyfhd_header, params, pyfhd_config, logger)

        if pyfhd_config['n_pol'] == 0:
            n_pol = pyfhd_header['n_pol']
        else:
            n_pol = pyfhd_config['n_pol']

        obs['n_pol'] = n_pol
        
        vis_model_arr, params_model = import_vis_model_from_uvfits(pyfhd_config,
                                                                   obs, logger)

        ##Need to flag the model depending on what tiles have been flagged
        ##in the data
        vis_model_arr, pyfhd_config = flag_model_visibilities(vis_model_arr, params,
                                      params_model, obs, pyfhd_config, logger)
        
        ##To feed into FHD, have to convert it to an IDL .sav file, so setup
        ##an output directory and save to that
        model_vis_dir = f"{output_dir}/model_vis"
        os.makedirs(model_vis_dir, exist_ok=True)

        if pyfhd_config['n_pol'] == 0:
            n_pol = pyfhd_header['n_pol']
        else:
            n_pol = pyfhd_config['n_pol']

        convert_vis_model_arr_to_sav(vis_model_arr, pyfhd_config,
                                     logger, model_vis_dir, n_pol)

    logger.info("Writing IDL .pro files to run IDL FHD calibration only")

    ##Convert the lovely arguments into pyfhd_config.pro to feed into IDL 
    convert_argdict_to_pro(pyfhd_config, output_dir)
    
    ##Write run_fhd_calibration_only.pro which we can call via command line
    ##This reads in all the arguments set in pyfhd_config.pro
    write_run_FHD_calibration_pro(pyfhd_config, output_dir)

    ##Copy some template .pro code into the directory where we will run
    ##IDL; these are called by run_fhd_calibration_only.pro
    fhd_cal_only = importlib_resources.files('PyFHD.templates').joinpath('fhd_calibration_only.pro')
    shutil.copy(fhd_cal_only, output_dir)

    general_cal_only = importlib_resources.files('PyFHD.templates').joinpath('general_calibration_only.pro')
    shutil.copy(general_cal_only, output_dir)

    ##Move into the output directory so IDL can see all the .pro files
    os.chdir(output_dir)

    ##Let the user know how IDL was launched, might help with bug
    ##hunting later on

    idl_command = "idl -IDL_DEVICE ps -e run_fhd_calibration_only"

    logger.info(f"Launching IDL on the command line via the command:\n\t$ {idl_command}")

    before = time.time()

    ##Launch the IDL code and cross your fingers
    idl_lines = run_command(idl_command, pyfhd_config['IDL_dry_run'])

    ##Stick some tabs on front out idl lines so they sit nice in the log
    idl_lines = "\t" + idl_lines.replace("\n", "\n\t")

    logger.info('Here is everything IDL FHD printed:\n' + idl_lines)

    IDL_output_dir = f"{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}/fhd_{pyfhd_config['top_level_dir']}"

    after = time.time()

    logger.info(f"Running IDL FHD calibration took {(after - before) / 60.0:.1f} minutes")

    return IDL_output_dir


def write_run_FHD_healpix_imaging_pro(pyfhd_config : dict,
                                      output_dir : str):
    """
    Write the top level run_fhd_healpix_imaging.pro file into the appropriate
    `output_dir`, using arguments found in the dict `pyfhd_config`.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    output_dir: str
        Where to save run_fhd_calibration_only.pro

    """

    with open(f"{output_dir}/run_fhd_healpix_imaging_{pyfhd_config['obs_id']}.pro", 'w') as outfile:
        outfile.write(f"pro run_fhd_healpix_imaging_{pyfhd_config['obs_id']}\n")
        outfile.write("\n")
        ##First of all, explicitly set some keywords that are used by the
        ##`fhd_path_setup` function below. Have to do this first, as the
        ##keywords set by `fhd_path_setup` need to be bundled into the `extra`
        ##structure  
        outfile.write("    ; Keywords\n")
        vis_file_list = f"{pyfhd_config['input_path']/pyfhd_config['obs_id']}.uvfits"
        outfile.write(f"    obs_id='{pyfhd_config['obs_id']}'\n")
        outfile.write(f'    vis_file_list="{vis_file_list}"\n')
        outfile.write(f"    output_directory='{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}'\n")
        outfile.write(f"    version='{pyfhd_config['top_level_dir']}'\n")
        outfile.write("\n")

        ##This sets up a bunch of paths so IDL-FHD knows where to output thigns            
        outfile.write("    ; Directory setup\n")
        outfile.write("    fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory)\n")
        outfile.write("    healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version)\n")
        outfile.write("\n")
        
        ##Add in a path to where the gridded hdf5 files should live so IDL
        ##can find them
        outfile.write("    ; Path to where the python-gridded hdf5 files live\n")
        
        outfile.write(f"    python_grid_path='{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}/gridding_outputs/'\n\n")
        
        ##Add in some extra healpix-ps related keywords
        
        outfile.write("    ; Add in some extra healpix-ps related keywords\n")
        outfile.write("    model_flag=1\n")
        outfile.write("    restrict_hpx_inds='EoR0_high_healpix_inds_3x.idlsave'\n")
        outfile.write(f"    grid_psf_file='{pyfhd_config['grid_psf_file_sav']}'\n")
        
        ##If the user has passed the `--IDL_variables_file` arg, this will write
        ##out the contents to the master file with the same indent level
        _write_user_IDL_variable_lines(pyfhd_config, outfile)
        
        ##This reads in all the other keywords that have been written to
        ##pyfhd_config.pro, and bundles them into a structure `extra` along
        ##with any other keywords already set
        outfile.write("    ; Set global defaults and bundle all the variables into a structure.\n")
        outfile.write("    ; Any keywords set on the command line or in the top-level wrapper will supercede these defaults\n")
        outfile.write("    pyfhd_config,extra\n")
        outfile.write("\n")

        ##print out all the variables so we can check them later
        outfile.write("\n")
        outfile.write("    ; print all the keywords that are now set\n")
        outfile.write('    print,""\n')
        outfile.write('    print,"Keywords set in wrapper:"\n')
        outfile.write("    print,structure_to_text(extra)\n")
        outfile.write("\n")

        ##This finally actually calls FHD
        outfile.write('    print,""\n')
        outfile.write("    ; this runs FHD proper\n")
        outfile.write("    general_healpix_imaging,_Extra=extra\n")
        outfile.write("end\n")


def run_IDL_convert_gridding_to_healpix_images(pyfhd_config : dict,
                                              logger : logging.RootLogger):
    """Assuming that `run_gridding_on_IDL_outputs` has been run to create gridding visibility hdf5 files, run IDL code to make image slices,
    and project them into healpix.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.RootLogger
        The logger to output info and errors to
    """
    
    
    output_dir = f"{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}"

    logger.info("Writing IDL .pro files to image PyFHD gridded outputs")

    ##Convert the lovely arguments into pyfhd_config.pro to feed into IDL 
    convert_argdict_to_pro(pyfhd_config, output_dir)
    
    ##Write the top level controlling script to run IDL code
    write_run_FHD_healpix_imaging_pro(pyfhd_config, output_dir)
    
    ##Copy some template .pro code into the directory where we will run
        ##IDL; these are called by run_fhd_calibration_only.pro
    files_needed = ["fhd_healpix_imaging.pro",
                    "general_healpix_imaging.pro",
                    "healpix_snapshot_cube_generate_read_python.pro",
                    "vis_model_freq_split_read_python.pro"]
    
    for file in files_needed:
        file_path = importlib_resources.files('PyFHD.templates').joinpath(file)
        shutil.copy(file_path, output_dir)

    ##Move into the output directory so IDL can see all the .pro files
    os.chdir(output_dir)

    ##Let the user know how IDL was launched, might help with bug
    ##hunting later on

    idl_command = f"idl -IDL_DEVICE ps -e run_fhd_healpix_imaging_{pyfhd_config['obs_id']}"

    logger.info(f"Launching IDL on the command line via the command:\n\t$ {idl_command}")

    before = time.time()

    ##Launch the IDL code and cross your fingers
    idl_lines = run_command(idl_command, pyfhd_config['IDL_dry_run'])
    
    ##Stick some tabs on front out idl lines so they sit nice in the log
    idl_lines = "\t" + idl_lines.replace("\n", "\n\t")

    logger.info('Here is everything IDL FHD printed:\n' + idl_lines)

    IDL_output_dir = f"{pyfhd_config['output_path']}/{pyfhd_config['top_level_dir']}/fhd_{pyfhd_config['top_level_dir']}"

    after = time.time()

    logger.info(f"Running IDL FHD imaging/healpix projection took {(after - before) / 60.0:.1f} minutes")