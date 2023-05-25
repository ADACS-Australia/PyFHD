import numpy as np
from typing import Tuple
from logging import RootLogger
from PyFHD.calibration.calibration_utils import (
    vis_extract_autocorr, 
    vis_cal_auto_init, 
    vis_calibration_flag, 
    vis_cal_bandpass, 
    vis_cal_polyfit,  
    vis_cal_auto_fit, 
    vis_calibration_apply,
    vis_baseline_hist,
    cal_auto_ratio_divide, 
    cal_auto_ratio_remultiply
)
from PyFHD.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from PyFHD.pyfhd_tools.pyfhd_utils import resistant_mean

def calibrate(obs: dict, params: dict, vis_arr: np.array, vis_weights: np.array, pyfhd_config: dict, logger: RootLogger) -> Tuple[np.array, np.array, dict] :
    """TODO: Docstring

    Parameters
    ----------
    obs : dict
        _description_
    params : dict
        _description_
    vis_arr : np.array
        _description_
    vis_weights : np.array
        _description_
    pyfhd_config : dict
        _description_
    logger : RootLogger
        _description_

    Returns
    -------
    Tuple[np.array, np.array, dict]
        _description_
    """
    # Initialize cal dict
    cal = {}
    # Calculate this here as it's used throughout the calibration process
    cal["n_pol"] = min(obs["n_pol"], 2)
    cal["conv_thresh"] = 1e-7
    
    # TODO: Get the vis_model_arr from sources, This will be the code from the branch model_transfer, add a placeholder for now
    vis_model_arr = np.array([])

    # Calculate auto-correlation visibilities, optionally use them for initial calibration estimates
    vis_auto, auto_tile_i = vis_extract_autocorr(obs, vis_arr);
    # Calculate auto-correlation visibilities 
    vis_auto_model, auto_tile_i = vis_extract_autocorr(obs, vis_model_arr, auto_tile_i = auto_tile_i)
    # Initalize the gain
    if (pyfhd_config['calibration_auto_initialize']):
        cal["gain"] = vis_cal_auto_init(obs, cal, vis_arr, vis_model_arr, vis_auto, vis_auto_model, auto_tile_i)
    else:
        cal["gain"] = np.full((cal['n_pol'], obs['n_tile'], obs['n_freq']), pyfhd_config['cal_gain_init'])

    # Do the calibration with vis_calibrate_subroutine 
    # TODO: vis_calibrate_subroutine outputs cal structure in FHD, likely will need to change here, or for the cal dictionary to be passed in and edited
    cal = vis_calibrate_subroutine(vis_arr, vis_model_arr, vis_weights, obs, cal)
    obs = vis_calibration_flag(obs, cal, pyfhd_config, logger)
    cal_base = cal.copy()

    # Perform bandpass (amp + phase per fine freq) and polynomial fitting (low order amp + phase fit plus cable reflection fit)
    if (pyfhd_config["bandpass-calibrate"]):
        if (pyfhd_config['auto_ratio_calibration']):
            cal, auto_ratio = cal_auto_ratio_divide(obs, cal, vis_auto, auto_tile_i)
        else:
            auto_ratio = None
        cal_bandpass, cal_remainder = vis_cal_bandpass(obs, cal, params, pyfhd_config, logger)

        if (pyfhd_config["calibration-polyfit"]):
            cal_polyfit = vis_cal_polyfit(cal_remainder, obs, auto_ratio, pyfhd_config, logger)
            # Replace vis_cal_combine with this line as the gain is the same size for polyfit and bandpass
            cal['gain'] = cal_polyfit['gain'] * cal_bandpass['gain']
        else:
            cal = cal_bandpass
        
        if(pyfhd_config['auto_ratio_calibration']):
            cal = cal_auto_ratio_remultiply(obs, cal, auto_tile_i)
    elif (pyfhd_config["calibration-polyfit"]):
        cal = vis_cal_polyfit(cal, obs)

    # Get the gain residuals
    cal_res = cal_base
    if (pyfhd_config['calibration_auto_fit']):
        # Get amp from auto-correlation visibilities for plotting (or optionally for the calibration solution itself)
        cal_auto = vis_cal_auto_fit(obs, cal, vis_auto, vis_auto_model, auto_tile_i)
        cal_res = cal_base['gain'] - cal_auto['gain']
    else:
        cal_res = cal_base['gain'] - cal['gain']

    # Add plotting later here, plot_cals was the function in IDL if you wish to translate

    # If calibration_auto_fit was set then replace cal with cal_auto, usually for diagnostic purposes
    if (pyfhd_config['calibration_auto_fit']):
        cal = cal_auto
    # Apply Calibration
    vis_cal, cal = vis_calibration_apply(vis_arr, obs, cal, vis_model_arr, vis_weights)
    cal["gain_resolution"] = cal_res["gain"]

    # Save the ratio and sigma average variance related to vis_cal
    cal['vis_baseline_hist'] = vis_baseline_hist(obs, params, vis_cal, vis_model_arr)

    # Calculate statistics to put into the calibration dictionary for output purposes
    nc_pol = min(obs["n_pol"], 2)
    cal_gain_avg = np.zeros(nc_pol)
    cal_res_avg = np.zeros(nc_pol)
    cal_res_restrict = np.zeros(nc_pol)
    cal_res_stddev = np.zeros(nc_pol)
    for pol_i in range(nc_pol):
        tile_use_i = np.where(obs["baseline_info"]["tile_use"])[0]
        freq_use_i = np.where(obs["baseline_info"]["freq_use"])[0]
        if (tile_use_i.size == 0 or freq_use_i.size == 0):
            continue
        # Replaced extract_subarray with just the proper indexing
        gain_ref = cal["gain"][pol_i, tile_use_i, freq_use_i]
        gain_res = cal_res["gain"][pol_i, tile_use_i, freq_use_i]
        cal_gain_avg[pol_i] = np.mean(np.abs(gain_ref))
        cal_res_avg[pol_i] = np.mean(np.abs(gain_res))
        res_mean = resistant_mean(np.abs(gain_res), 2)
        cal_res_restrict[pol_i] = res_mean
        cal_res_stddev[pol_i] = np.std(np.abs(gain_res))
    
    if ("mean_gain" in cal):
        cal["mean_gain"] = cal_gain_avg
    if ("mean_gain_residual" in cal):
        cal["mean_gain_residual"] = cal_res_avg
    if ("mean_gain_restrict" in cal):
        cal["mean_gain_restrict"] = cal_res_restrict
    if ("stddev_gain_residual" in cal):
        cal["stddev_gain_residual"] = cal_res_stddev


    # Return the calibrated visibility array
    return vis_cal, vis_model_arr, cal