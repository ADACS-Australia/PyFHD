import numpy as np
import logging
from math import pi, log10
from PyFHD.pyfhd_tools.pyfhd_utils import idl_argunique, histogram
from pathlib import Path
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

def create_obs(pyfhd_header : dict, params : dict, pyfhd_config : dict, logger : logging.RootLogger) -> dict:
    """
    create_obs takes all the data that has been read in and creates the obs data structure which holds data
    and metadata of the observation we're doing a PyFHD run on. Inside this function the metafits file will
    be read as well.

    Parameters
    ----------
    pyfhd_header : dict
        The data from the UVFITS header
    params : dict
        The data from the UVFITS file
    pyfhd_config : dict
        PyFHD's configuration dictionary
    logger : logging.RootLogger
        The PyFHD logger

    Returns
    -------
    obs : dict
        The observatiopn data structure for PyFHD containing data from the config and metadata from the observation UVFITS and METAFITS files.
    """

    obs = {}
    baseline_info = {}

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
    if antenna_flag:
        # 256 tile upper limit is hard-coded in CASA format
        # these tile numbers have been verified to be correct
        baseline_min = np.min(params['baseline_arr'])
        exponent = np.log(np.min(baseline_min)) / np.log(2)
        antenna_mod_index = 2 ** np.floor(exponent)
        tile_B_test = baseline_min % antenna_mod_index
        # Check if a bad fit and if autocorrelations or the first tile are missing
        if (tile_B_test > 1) and (baseline_min % 2 == 1):
            antenna_mod_index /= 2 ** np.floor(np.log(np.min(tile_B_test)) / np.log(2))
        baseline_info['tile_A'] = np.floor(params['baseline_arr'] / antenna_mod_index)
        baseline_info['tile_B'] = np.fix(params['baseline_arr'] / antenna_mod_index)
        if max(np.max(baseline_info['tile_A']), np.max(baseline_info['tile_B'])) != obs['n_tile']:
            logger.warning(f"Mis-matched n_tiles Header: {obs['n_tile']}, Data: {max(np.max(baseline_info['tile_A']), np.max(baseline_info['tile_B']))}, adjusting n_tiles to be same as data")
            obs['n_tile'] = max(np.max(baseline_info['tile_A']), np.max(baseline_info['tile_B']))
        params['antenna1'] = baseline_info['tile_A']
        params['antenna2'] = baseline_info['tile_B']
    
    baseline_info['freq_use'] = obs['n_freq'] + 1
    baseline_info['tile_use'] = obs['n_tile'] + 1

    # Calculate kx and ky for each baseline at high precision to get most accurate observation information
    kx_arr = np.outer(baseline_info['freq'] ,params['uu'])
    ky_arr = np.outer(baseline_info['freq'], params['vv'])
    kr_arr = np.sqrt(kx_arr ** 2 + ky_arr ** 2)
    max_baseline = max(np.max(np.abs(kx_arr)), np.max(np.abs(ky_arr)))

    # Determine the imaging parameters to use
    if pyfhd_config['FoV'] is not None:
        obs['kbinsize'] =  (180 / pi) / pyfhd_config['FoV']
    if pyfhd_config['kbinsize'] is None:
        obs['kbinsize'] = 0.5
    else:
        obs['kbinsize'] = pyfhd_config['kbinsize']
        
    # Determine observation resolution/extent parameters given number of pixels in x direction (dimension)
    if pyfhd_config['dimension'] is None and pyfhd_config['elements'] is None:
       obs['dimension'] = 2 ** int((log10((2 * max_baseline) / pyfhd_config['kbinsize']) / log10(2)))
       obs['elements'] = obs['dimension']
    elif pyfhd_config['dimension'] is not None and pyfhd_config['elements'] is None:
        obs['dimension'] = pyfhd_config['dimension']
        obs['elements'] = pyfhd_config['dimension']
    elif pyfhd_config['dimension'] is None and pyfhd_config['elements'] is not None:
        obs['dimension'] = pyfhd_config['elements']
        obs['elements'] = pyfhd_config['elements']
    else:
        obs['dimension'] = pyfhd_config['dimension']
        obs['elements'] = pyfhd_config['elements']
    obs['degpix'] = (180 / pi) / (pyfhd_config['kbinsize'] * pyfhd_config['dimension'])

    max_baseline_inds = np.where((np.abs(kx_arr) / obs['kbinsize'] < obs['dimension'] / 2) & (np.abs(ky_arr) / obs['kbinsize'] < obs['elements']/2))
    obs['max_baseline'] = np.max(np.abs(kx_arr[max_baseline_inds]))
    if pyfhd_config['min_baseline'] is None:
        obs['min_baseline'] = np.min(kr_arr[np.nonzero(kr_arr)])
    else:
        obs['min_baseline'] = max(pyfhd_config['min_baseline'], np.min(kr_arr[np.nonzero(kr_arr)]))

    meta = read_metafits(obs, pyfhd_header, params, pyfhd_config, logger)
    return obs

def read_metafits(obs : dict, pyfhd_header : dict, params : dict, pyfhd_config : dict, logger : logging.RootLogger) -> dict:
    """_summary_

    Parameters
    ----------
    obs : dict
        The current obs structure without the metadata
    pyfhd_header : dict
        The data from the UVFITS header
    params : dict
        The data from the UVFITS file
    pyfhd_config : dict
        PyFHD's configuration dictionary
    logger : logging.RootLogger
        PyFHD's logger

    Returns
    -------
    meta : dict
        The dictionary holding the metadata from the UVFITS and METAFITS files
    """
    
    meta = {}
    time = params['time']
    b0i = idl_argunique(time)
    jdate = pyfhd_header['jd0'] + time[b0i]
    meta['obsx'] = obs['dimension'] / 2
    meta['obsy'] = obs['elements'] / 2
    meta_path = Path(pyfhd_config['input_path'], pyfhd_config['obs_id'] + '.metafits')
    if meta_path.is_file():
        metadata = fits.open(meta_path)
        hdr = metadata[0].header
        data = metadata[1].data
        # Sort the data by antenna using a stable sort, astropy Table is required to access Antenna column for sorting
        # Standard Astropy does not do stable sorting, hence use of argsort to do stable sorting
        data = data[np.array(Table(data).argsort('Antenna', kind = 'stable'))]
        single_i = np.where(data['pol'] == data['pol'][0])
        meta['tile_names'] = data['tile'][single_i]
        meta['tile_height'] = data['height'][single_i] - pyfhd_header['alt']
        meta['tile_flag'] = data['flag']
        if np.sum(meta['tile_flag']) == meta['tile_flag'].size - 1:
            if pyfhd_config['run_simulation']:
                logger.warning("All tiles flagged in metadata")
            else:
                logger.error("All tiles flagged in metadata")
                exit()
        meta['obsra'] = hdr['RA']
        meta['obsdec'] = hdr['DEC']
        meta['phasera'] = hdr['RAPHASE']
        meta['phasedec'] = hdr['DECPHASE']
        meta['time_res'] = hdr['INTTIME']
        meta['JD0'] = np.min(jdate)

        epoch = Time(meta['JD0'], format='jd').to_value('decimalyear')
        meta['delays'] = hdr['DELAYS'].split(',')

    else:
        logger.warning("METAFITS file has not been found, Calculating obs meta settings from the uvfits header instead")
    return meta