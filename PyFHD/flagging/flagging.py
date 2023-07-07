import numpy as np
from scipy.ndimage import median_filter
from logging import RootLogger
from PyFHD.pyfhd_tools.pyfhd_utils import idl_median

def vis_flag(vis_arr : np.ndarray, vis_weights: np.ndarray, obs: dict, params: dict, logger: RootLogger) -> tuple[np.ndarray, dict] :
    """
    TODO: __summary__

    Parameters
    ----------
    vis_arr : np.ndarray
        _description_
    vis_weights : np.ndarray
        _description_
    obs : dict
        _description_
    params : dict
        _description_
    logger : RootLogger
        _description_

    Returns
    -------
    tuple[np.ndarray, dict]
        _description_
    """
    flag_nsigma = 3
    data_abs = np.abs(vis_arr[0])
    if (obs['n_pol'] > 1):
        data_abs = np.sqrt(data_abs ** 2 + np.abs(vis_arr[1]) ** 2)
    n_tiles_use = max(np.max(obs["baseline_info"]["tile_a"]), np.max(obs["baseline_info"]["tile_b"]))
    uv_dist = np.sqrt(params["uu"] ** 2 + params["vv"] ** 2) * idl_median(obs["baseline_info"]["freq"])
    freq = obs["baseline_info"]["freq"]
    cut_baselines_i = np.where((uv_dist < obs["min_baseline"]) | (uv_dist > obs["max_baseline"]))[0]
    if (cut_baselines_i.size > 0):
        vis_weights[:obs["n_pol"], :, cut_baselines_i] = 0

    tile_fom = np.empty(n_tiles_use)
    for tile_i in range(n_tiles_use):
        tile_ai = np.where((obs["baseline_info"]["tile_a"] - 1) == tile_i)
        tile_bi = np.where((obs["baseline_info"]["tile_b"] - 1) == tile_i)
        if (tile_ai[0].size == 0 and tile_bi[0].size == 0):
            continue
        elif (tile_bi[0].size > 0 and tile_ai[0].size > 0):
            tile_abi = np.hstack([tile_ai[0], tile_bi[0]])
        elif (tile_bi[0].size > 0 and tile_ai[0].size == 0):
            tile_abi = tile_bi
        else:
            tile_abi = tile_ai
        data_subset = data_abs[:, tile_abi]
        for pol_i in range(min(obs["n_pol"], 2)):
            #Dunno why this needs two separate indexes but it works so leave it
            i_use = np.where((vis_weights[pol_i][:, tile_abi] > 0) & (data_subset > 0))
            if (i_use[0].size > 10):
                tile_fom[tile_i] += np.std(data_subset[i_use])
    
    freq_fom = np.zeros(obs["n_freq"])
    for freq_i in range(obs["n_freq"]):
        data_subset = data_abs[freq_i,:]
        for pol_i in range(min(obs["n_pol"], 2)):
            i_use = np.where((vis_weights[pol_i, freq_i, :] > 0) & (data_subset > 0))
            if (i_use[0].size > 10):
                freq_fom[freq_i] += np.std(data_subset[i_use])

    freq_nonzero = np.nonzero(freq_fom)[0]
    tile_nonzero = np.nonzero(tile_fom)
    tile_mean = idl_median(tile_fom[tile_nonzero])
    tile_dev = np.std(tile_fom[tile_nonzero])
    # IDL Median with WIDTH set is doing a median_filter with the size indicating the kernel size
    freq_mean1 = idl_median(freq_fom[freq_nonzero], width=obs["n_freq"] // 20)
    freq_mean = np.zeros(obs["n_freq"])
    freq_mean[freq_nonzero] = freq_mean1
    freq_dev = np.std(freq_fom[freq_nonzero] - freq_mean[freq_nonzero])

    # We actually want the complements of the where from IDL translation, adjusted conditions accordingly
    tile_cut0 = np.where((np.abs(tile_mean - tile_fom) <= 2 * flag_nsigma * tile_dev) | (tile_fom != 0))[0]
    freq_cut0 = np.where((np.abs(freq_mean - freq_fom) <= 2 * flag_nsigma * freq_dev) | (freq_fom != 0))[0]
    tile_mean2 = idl_median(tile_fom[tile_cut0])
    tile_dev2 = np.std(tile_fom[tile_cut0])
    freq_mean2 = idl_median(freq_fom, width = obs["n_freq"] // 20)
    freq_dev2 = np.std((freq_fom - freq_mean2)[freq_cut0])
    # Currently assuming tile_cut and freq_cut are 1D
    tile_cut = np.where((np.abs(tile_mean2 - tile_fom) > flag_nsigma * tile_dev2) | (tile_fom == 0))[0]
    freq_cut = np.where((np.abs(freq_mean2 - freq_fom) > flag_nsigma * freq_dev2) | (freq_fom == 0))[0]

    if (tile_cut.size > 0):
        logger.info(f"Tiles Cut: {obs['baseline_info']['tile_names'][tile_cut]}")
        for bad_i in range(tile_cut.size):
            cut_a_i = np.where(obs["baseline_info"]["tile_a"] == tile_cut[bad_i]+1)
            cut_b_i = np.where(obs["baseline_info"]["tile_b"] == tile_cut[bad_i]+1)
            if (cut_a_i[0].size > 0):
                vis_weights[0 : obs["n_pol"], :, cut_a_i] = 0 
            if (cut_b_i[0].size > 0):
                vis_weights[0 : obs["n_pol"], :, cut_b_i] = 0
        obs["baseline_info"]["tile_use"][tile_cut] = 0
    if (freq_cut.size > 0):
        vis_weights[0 : obs["n_pol"], freq_cut] = 0
        obs["baseline_info"]["freq_use"][freq_cut] = 0
    
    bin_offset = np.append(obs["baseline_info"]["bin_offset"], obs["baseline_info"]["tile_a"].size)
    time_bin = np.zeros(obs["baseline_info"]["tile_a"].size)
    for ti in range(len(obs["baseline_info"]["time_use"])):
        time_bin[bin_offset[ti] : bin_offset[ti + 1]] = ti
    time_fom = np.zeros(obs["baseline_info"]["time_use"].size)
    for ti in range(len(obs["baseline_info"]["time_use"])):
        data_subset = data_abs[:, bin_offset[ti]:bin_offset[ti + 1]]
        for pol_i in range(min(obs["n_pol"], 2)):
            i_use = np.where(vis_weights[pol_i, :, bin_offset[ti]:bin_offset[ti + 1]] > 0)
            if (i_use[0].size > 10):
                time_fom[ti] += np.std(data_subset[i_use])
    time_nonzero = np.nonzero(time_fom)
    time_mean = idl_median(time_fom[time_nonzero])
    time_dev = np.std(time_fom[time_nonzero])
    time_cut0 = np.where((np.abs(time_mean - time_fom) <= 2 * flag_nsigma * time_dev) | (time_fom != 0))
    time_mean2 = idl_median(time_fom[time_cut0])
    time_dev2 = np.std(time_fom[time_cut0])
    time_cut = np.where((np.abs(time_mean2 - time_fom) > 2 * flag_nsigma * time_dev2) | (time_fom == 0))[0]
    for ti in range(time_cut.size):
        ti_cut = np.where(time_bin == time_cut[ti])
        if (ti_cut.size > 0):
            vis_weights[:obs["n_pol"], :, ti_cut] = 0

    obs['n_vis'] = (np.where(vis_weights[0] > 0)[0]).size

    return vis_weights, obs

