from astropy.io import fits
import numpy as np
from PyFHD.data_setup.uvfits import extract_visibilities
import logging
from PyFHD.use_idl_fhd import run_idl_fhd
import importlib_resources
import os
import shutil
import h5py

def vis_model_transfer(obs : dict) -> np.array:
    """Placeholder incase we decide to add functionality to read in IDL .sav
    model visibilities"""
    
    # function vis_model_transfer,obs,model_transfer

    #   ;; Option to transfer pre-made and unflagged model visbilities
    #   vis_model_arr=PTRARR(obs.n_pol,/allocate)

    #   for pol_i=0, obs.n_pol-1 do begin
    #     transfer_name = model_transfer + '/' + obs.obsname + '_vis_model_'+obs.pol_names[pol_i]+'.sav'
    #     if ~file_test(transfer_name) then $
    #       message, transfer_name + ' not found during model transfer.'
    #     vis_model_arr[pol_i] = getvar_savefile(transfer_name,'vis_model_ptr')
    #     print, "Model visibilities transferred from " + transfer_name
    #   endfor

    #   return, vis_model_arr

    # end
    
    return

def import_vis_model_from_uvfits(pyfhd_config : dict, obs : dict,
                                 logger : logging.RootLogger) -> np.ndarray:
    """Read a model visibility array in from a `uvfits` with filepath given
    by pyfhd_config['import_model_uvfits']. Reads data in via 
    `PyFHD.data_setup.uvfits import extract_visibilities`.

    Parameters
    ----------
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    obs : dict
        The observation dictionary as populated by `PyFHD.data_setup.obs.create_obs`

    Returns
    -------
    vis_model_arr : np.array
        A `np.complex128` type array of shape (n_pol, n_vis_raw, n_freq)
    """
    
    ##TODO WORRY about which order XX and YY are
    ##TODO WORRY about order of baselines comparing WODEN sims and real data
    ##TODO WORRY about weights
    
    with fits.open(pyfhd_config['import_model_uvfits']) as hdu:
        
        params_data = hdu[0].data
    
    ##These are the only parts of the model_header that are needed by
    ##`extract_visibilities`
    model_header = {}
    # Retrieve data from the params_header
    model_header['real_index'] = 0
    model_header['imaginary_index'] = 1
    model_header['weights_index'] = 2
        
    vis_model_arr, _ = extract_visibilities(model_header, params_data,
                                            pyfhd_config, logger)
    
    return vis_model_arr

def convert_vis_model_arr_to_sav(vis_model_arr : np.ndarray,
                                 pyfhd_config : dict,
                                 logger : logging.RootLogger,
                                 model_vis_dir : str, n_pol : int):
    """Converts the contents of `vis_model_arr` into an FHD .sav file format
    so we can import into existing IDL code with ease. First writes data to
    `hdf5` format, then uses IDL code template to convert to IDL `.sav` format
    compatible with FHD. Sticks the outputs into `model_vis_dir`.

    Parameters
    ----------
    vis_model_arr : np.ndarray
        Complex array hold the model visibilities
    pyfhd_config : dict
        The options from argparse in a dictionary, that have been verified using
        `PyFHD.pyfhd_tools.pyfhd_setup.pyfhd_setup`.
    logger : logging.RootLogger
        PyFHD logger to feed information to
    model_vis_dir : str
        Directory location to write the output files to
    n_pol : int
        Number of polarisations to write out (each is written to an individual)
        `.sav` file
    """

    pol_names = ['XX', 'YY', 'XY', 'YX']

    with h5py.File(f"{model_vis_dir}/{pyfhd_config['obs_id']}_vis_model.h5", 'w') as hf:

        for pol, pol_name in enumerate(pol_names[:n_pol]):
            logger.info(f"vis_model_transfer: saving {pyfhd_config['obs_id']}_vis_model_{pol_name}")
            hf.create_dataset(f"{pyfhd_config['obs_id']}_vis_model_{pol_name}",
                                    data=vis_model_arr[pol].transpose())
                
        hf.close()

    ##Grab the template IDL code and transfer so people can see what code was used
    ##and modify if they want
    model_arr_convert_pro = importlib_resources.files('PyFHD.templates').joinpath('convert_model_arr_to_sav.pro')
    shutil.copy(model_arr_convert_pro, model_vis_dir)

    ##Move into the output directory so IDL can see all the .pro files
    os.chdir(model_vis_dir)
    
    ##Run the IDL code
    idl_command = f"idl -IDL_DEVICE ps -e convert_model_arr_to_sav -args {model_vis_dir} {pyfhd_config['obs_id']} {n_pol}"
    run_idl_fhd.run_command(idl_command, pyfhd_config['IDL_dry_run'])