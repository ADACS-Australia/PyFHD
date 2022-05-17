import numpy as np
import logging
from PyFHD.pyfhd_tools.pyfhd_utils import idl_argunique, histogram

def create_obs(pyfhd_header : dict, params : dict, pyfhd_config : dict, logger : logging.RootLogger) -> dict:
    """
    TODO: Docstring
    _summary_

    Parameters
    ----------
    pyfhd_header : dict
        _description_
    params : dict
        _description_
    pyfhd_config : dict
        _description_

    Returns
    -------
    obs : dict
        The obs dictionary for this observation which contains and PyFHD run.
    """

    obs = {}
    baseline_info = {}
    speed_of_light = 299792458

    obs['n_pol'] = pyfhd_header['n_pol']
    obs['n_tile'] = pyfhd_header['n_tile']
    obs['n_freq'] = pyfhd_header['n_freq']
    obs['instrument'] = pyfhd_config['instrument']
    time = params['time']
    b0i = idl_argunique(time)
    obs['n_time'] = b0i.size
    bin_width = np.empty(obs['n_time'])
    if obs['n_time'] > 1:
        bin_width[0 : obs['n_time']] = b0i + 1
    else:
        bin_width = time.size
    b0i_range = np.arange(1, obs['n_time'])
    bin_width[b0i_range] = b0i[b0i_range] - b0i[b0i_range - 1]
    baseline_info['bin_offset'] = np.zeros(obs['n_time'], dtype = np.int64)
    if obs['n_time'] > 1:
        baseline_info['bin_offset'][1:] = np.cumsum(bin_width[: obs['n_time'] - 1])
    obs['n_baselines'] = bin_width[0]
    obs['n_vis'] = time.size * obs['n_freq']
    obs['n_vis_raw'] = obs['n_vis_in'] = obs['n_vis']
    obs['n_vis_arr'] = np.zeros(obs['n_freq'], dtype = np.int64)

    obs['freq_res'] = pyfhd_header['freq_res']
    baseline_info['freq'] = pyfhd_header['frequency_array']
    if pyfhd_config['beam_nfreq_avg'] is not None:
        obs['beam_nfreq_avg'] = pyfhd_config['beam_nfreq_avg']
    else:
        obs['beam_nfreq_avg'] = 1
    freq_bin = obs['beam_nfreq_avg'] * obs['freq_res']
    freq_hist, _, freq_ri = histogram(baseline_info['freq'], bin_size = freq_bin)
    freq_bin_i = np.zeros(obs['n_freq'])
    for bin in range(freq_hist.size):
        if freq_ri[bin] < freq_ri[bin + 1]:
            freq_bin_i[freq_ri[freq_ri[bin] : freq_ri[bin + 1]]] = bin
    baseline_info['fbin_i'] = freq_bin_i
    obs['freq_center'] = np.median(baseline_info['freq'])
    
    antenna_flag = True
    if np.max(params['antenna1']) > 0 and (np.max(params['antenna2']) > 0):
        baseline_info['tile_A'] = params['antenna1']
        baseline_info['tile_B'] = params['antenna2']
        obs['n_tile'] = max(np.max(baseline_info['tile_A']), np.max(baseline_info['tile_B']))
        antenna_flag = False
    # if antenna_flag:
    #     # 256 tile upper limit is hard-coded in CASA format
    #     # these tile numbers have been verified to be correct


    meta = read_metafits(pyfhd_header, params, pyfhd_config)
    return obs

def read_metafits(pyfhd_header : dict, params : dict, pyfhd_config : dict) -> dict:
    
    meta = {}
    return meta