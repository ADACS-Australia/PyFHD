import numpy as np
from typing import Tuple
from PyFHD.calibration.calibration_utils import vis_extract_autocorr, vis_cal_auto_init, vis_calibration_flag, vis_cal_bandpass, vis_cal_polyfit, vis_cal_combine, vis_cal_auto_fit, vis_cal_subtract, vis_calibration_apply
from PyFHD.calibration.vis_calibrate_subroutine import vis_calibrate_subroutine
from PyFHD.pyfhd_tools.pyfhd_utils import extract_subarray, resistant_mean

def calibrate(obs: dict, params: dict, vis_arr: np.array, vis_weights: np.array, pyfhd_config: dict) -> Tuple[np.array, np.array, dict] :
    # Initialize cal dict
    cal = {}
    
    # TODO: Get the vis_model_arr from sources, This will be the code from the branch model_transfer, add a placeholder for now
    vis_model_arr = np.array([])

    # Calculate auto-correlation visibilities, optionally use them for initial calibration estimates
    vis_auto, auto_tile_i = vis_extract_autocorr(obs, vis_arr);
    # Auto Initialize in FHD is set to 1, and is always true
    cal["gain"] = vis_cal_auto_init(obs, vis_arr, vis_model_arr)
    # Calculate auto-correlation visibilities 
    vis_auto_model = vis_extract_autocorr(obs, vis_model_arr, auto_tile_i = auto_tile_i)

    # Setup tile_A_i and tile_B_i
    tile_A_i = obs['baseline_info']['tile_A'] - 1
    tile_B_i = obs["baseline_info"]["tile_B"] - 1
    tile_A_i = tile_A_i[0:obs["nbaselines"]]

    # Do the calibration with vis_calibrate_subroutine 
    # TODO: vis_calibrate_subroutine outputs cal structure in FHD, likely will need to change here, or for the cal dictionary to be passed in and edited
    cal = vis_calibrate_subroutine(vis_arr, vis_model_arr, vis_weights, obs, cal)
    obs = vis_calibration_flag(obs, cal)
    cal_base = cal.copy()

    # Perform bandpass (amp + phase per fine freq) and polynomial fitting (low order amp + phase fit plus cable reflection fit)
    if (pyfhd_config["bandpass-calibrate"]):
        cal_bandpass, cal_remainder = vis_cal_bandpass(cal, obs, params)

        if (pyfhd_config["calibration-polyfit"]):
            cal_polyfit = vis_cal_polyfit(cal_remainder, obs)
            cal = vis_cal_combine(cal_polyfit, cal_bandpass)
    elif (pyfhd_config["calibration-polyfit"]):
        cal = vis_cal_polyfit(cal, obs)

    # Get amp from auto-correlation visibilities for plotting (or optionally for the calibration solution itself)
    cal_auto = vis_cal_auto_fit(obs, cal, vis_auto, vis_auto_model, auto_tile_i)
    cal_res = vis_cal_subtract(cal_base, cal)

    # TODO: Add plotting later

    # Apply Calibration
    vis_cal = vis_calibration_apply(vis_arr, cal, vis_model_arr, vis_weights)
    cal["gain_resolution"] = cal_res["gain"]

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
        gain_ref = extract_subarray(cal["gain"][pol_i], freq_use_i, tile_use_i)
        gain_res = extract_subarray(cal_res["gain"][pol_i], freq_use_i, tile_use_i)
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
    return vis_arr, vis_model_arr, cal