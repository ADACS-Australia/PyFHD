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
from PyFHD.pyfhd_tools.pyfhd_utils import resistant_mean, reshape_and_average_in_time

def calibrate(obs: dict, params: dict, vis_arr: np.array, vis_weights: np.array, pyfhd_config: dict, logger: RootLogger) -> Tuple[np.array, np.array, dict] :
    """
    TODO: Docstring

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
    vis_auto, auto_tile_i = vis_extract_autocorr(obs, vis_arr, pyfhd_config);
    # Calculate auto-correlation visibilities 
    vis_auto_model, auto_tile_i = vis_extract_autocorr(obs, vis_model_arr, pyfhd_config, auto_tile_i = auto_tile_i)
    # Initalize the gain
    if (pyfhd_config['calibration_auto_initialize']):
        cal["gain"] = vis_cal_auto_init(obs, cal, vis_arr, vis_model_arr, vis_auto, vis_auto_model, auto_tile_i)
    else:
        cal["gain"] = np.full((cal['n_pol'], obs['n_freq'], obs['n_tile']), pyfhd_config['cal_gain_init'])

    # Do the calibration with vis_calibrate_subroutine 
    # TODO: vis_calibrate_subroutine outputs cal structure in FHD, likely will need to change here, or for the cal dictionary to be passed in and edited
    cal = vis_calibrate_subroutine(vis_arr, vis_model_arr, vis_weights, obs, cal, pyfhd_config)
    if (pyfhd_config['flag_calibration']):
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

def calibrate_qu_mixing(vis_arr: np.ndarray, vis_model_arr : np.ndarray, vis_weights: np.ndarray, obs : dict) -> float:
    """
    TODO: _summary_

    Parameters
    ----------
    vis_arr : np.ndarray
        The visibility array
    vis_model_arr : np.ndarray
        The array containing the model for the visibilities
    vis_weights : np.ndarray
        The visibility weights array
    obs : dict
        The observation dictionary

    Returns
    -------
    stokes_mix_phase : float
        TODO: _description_
    """

    n_freq = obs['n_freq']
    # n_tile = obs['n_tile']
    n_time = obs['n_time']
    #This should be number of baselines for one time step
    n_baselines = obs['nbaselines']

    # reshape from (n_freq, n_baselines*n_times) to (n_freq, n_times, n_baselines). Turns out due to the row major vs col major difference
    # between IDL and python, this shape also changes
    new_shape = (n_freq, n_time, n_baselines)

    # Use the xx weightss (yy should be identical at this point)
    weights_use = np.reshape(vis_weights[0, :, :], new_shape)
    #carried over from FHD code - not sure this is necessary (maybe avoids NaNs?)
    weights_use = np.maximum(weights_use, np.zeros_like(weights_use))
    weights_use = np.minimum(weights_use, np.ones_like(weights_use))

    #Q = YY - XX for data
    pseudo_q = reshape_and_average_in_time(vis_arr[1, :, :] - vis_arr[0, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)
    
    #U = YX + XY for data
    pseudo_u = reshape_and_average_in_time(vis_arr[3, :, :] + vis_arr[2, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)
    
    #Q = YY - XX for model
    pseudo_q_model = reshape_and_average_in_time(vis_model_arr[1, :, :] - vis_model_arr[0, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)
    
    #U = YX + XY for model
    pseudo_u_model = reshape_and_average_in_time(vis_model_arr[3, :, :] + vis_model_arr[2, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)

    weights_t_avg = np.sum(weights_use, axis=1)

    i_use = np.nonzero(weights_t_avg)
    pseudo_u = np.squeeze(pseudo_u[i_use]).flatten()
    pseudo_q = np.squeeze(pseudo_q[i_use]).flatten()
    pseudo_q_model = np.squeeze(pseudo_q_model[i_use]).flatten()
    pseudo_u_model = np.squeeze(pseudo_u_model[i_use]).flatten()

    # LA_LEAST_SQUARES does not use double precision by default as such you will see differences.
    # Also LA_LEAST_SQUARES uses different method to numpy, generally LA_LEAST_SQUARES assumes the first matrix
    # has full rank, while numpy does not assume this.

    x = np.vstack([pseudo_u, np.ones(pseudo_u.size)]).T
    u_q_mix = np.linalg.lstsq(x, pseudo_q, rcond = None)[0]
    u_q_phase = np.arctan2(u_q_mix[0].imag, u_q_mix[0].real)

    x = np.vstack([pseudo_u_model, np.ones(pseudo_u_model.size)]).T
    u_q_mix_model = np.linalg.lstsq(x, pseudo_q_model, rcond = None)[0]
    u_q_phase_model = np.arctan2(u_q_mix_model[0].imag, u_q_mix_model[0].real)

    return u_q_phase_model - u_q_phase