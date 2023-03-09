import numpy as np
from typing import Tuple
from calibration_utils import vis_extract_autocorr, vis_cal_auto_init

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

    # Do the calibration loop with vis_calibrate_subroutine 
    # TODO: vis_calibrate_subroutine outputs cal structure in FHD, likely will need to change here, or for the cal dictionary to be passed in and edited

    # Perform bandpass (amp + phase per fine freq) and polynomial fitting (low order amp + phase fit plus cable reflection fit)

    # Get amp from auto-correlation visibilities for plotting (or optionally for the calibration solution itself)

    # TODO: Add plotting later

    # Apply Calibration

    # Calculate statistics to put into the calibration dictionary for output purposes

    # Return the calibrated visibility array

    return vis_arr, vis_model_arr, cal