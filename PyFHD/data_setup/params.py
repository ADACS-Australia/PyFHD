import numpy as np
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
import logging

def extract_header(pyfhd_config : dict, logger : logging.RootLogger) -> np.recarray:
    """_summary_

    Parameters
    ----------
    pyfhd_config : dict
        _description_
    logger : logging.RootLogger
        _description_
    lon : float, optional
        Given in degrees default is from (MWA, from Tingay et al. 2013), by default 116.67081524
    lat : float, optional
        Given in degrees default is from (MWA, from Tingay et al. 2013), by default -26.7033194
    alt : float, optional
        Altitude in metres default is from (MWA, from Tingay et al. 2013), by default 377.827

    Returns
    -------
    np.recarray
        _description_
    """

    data, header = fits.getdata(Path(pyfhd_config['input_path'], pyfhd_config['obs_id'] + '.uvfits'), header=True)

    pyfhd_header = {}
    # Retrieve data from the header
    pyfhd_header['pol_dim'] = 2
    pyfhd_header['freq_dim'] = 4
    pyfhd_header['real_index'] = 0
    pyfhd_header['imaginary_index'] = 1
    pyfhd_header['weights_index'] = 2
    pyfhd_header['n_tile'] = 128.0
    pyfhd_header['naxis'] = header['naxis']
    pyfhd_header['n_params'] = header['pcount']
    pyfhd_header['nbaselines'] = header['gcount']
    pyfhd_header['n_complex'] = header['naxis2']
    pyfhd_header['n_pol'] = header['naxis3']
    pyfhd_header['n_freq'] = header['naxis4']
    pyfhd_header['freq_ref'] = header['crval4']
    pyfhd_header['freq_res'] = header['cdelt4']
    pyfhd_header['date_obs'] = header['date-obs']
    freq_ref_i = header['crpix4'] - 1
    pyfhd_header['freq_array'] = (np.arange(pyfhd_header['n_freq']) - freq_ref_i) * pyfhd_header['freq_res'] + pyfhd_header['freq_ref']
    pyfhd_header['obsra'] = header['obsra']
    pyfhd_header['obsdec'] = header['obsdec']
    # Put in locations of instrument
    pyfhd_header['lon'] = pyfhd_config['lon']
    pyfhd_header['lat'] = pyfhd_config['lat']
    pyfhd_header['alt'] = pyfhd_config['alt']

    # Setup params list and names
    param_list = []
    ptype_list = ['PTYPE{}'.format(i) for i in range(1, pyfhd_header['n_params'] + 1)]
    for ptype in ptype_list:
        param_list.append(header[ptype].strip())
    param_names = []
    for key in list(header.keys()):
        if key.startswith('CTYPE'):
            param_names.append(header[key].strip().lower())
        
    # Validate params list, if anyone of these is not in the list then there will be a ValueError
    params_valid = True
    try:
        pyfhd_header['uu_i'] = param_list.index('UU')
    except ValueError:
        logger.error('Group parameter UU not found within uvfits header PTYPE keywords')
        params_valid = False
    try:
        pyfhd_header['vv_i'] = param_list.index('VV')
    except ValueError:
        logger.error('Group parameter VV not found within uvfits header PTYPE keywords')
        params_valid = False
    try:
        pyfhd_header['ww_i'] = param_list.index('WW')
    except ValueError:
        logger.error('Group parameter WW not found within uvfits header PTYPE keywords')
        params_valid = False
    try:
        if ('ANTENNA1' in param_list and 'ANTENNA2' in param_list):
            pyfhd_header['ant1_i'] = param_list.index('ANTENNA1')
            pyfhd_header['ant2_i'] = param_list.index('ANTENNA2')
        else:
            pyfhd_header['baseline_i'] = param_list.index('BASELINE')
    except ValueError:
        logger.error('Group parameter BASELINE (or ANTENNA1 and ANTENNA2) not found within uvfits header PTYPE keywords')
        params_valid = False
    try:
        pyfhd_header['date_i'] = param_list.index('DATE')
    except ValueError:
        logger.error('Group parameter DATE not found within uvfits header PTYPE keywords')
        params_valid = False
    
    if not params_valid:
        raise KeyError('One of these keys is missing from the FITS file: UU, VV, WW, BASELINE, DATE, check the log to see which one')
    
    if param_list.count('DATE') > 1:
        # This needs testing as Astropy scales automatically, which affects the DATE data read in, this should be the same though
        pyfhd_header['jd0'] = header['PZERO{}'.format(pyfhd_header['date_i'] + 1)] + data['DATE'][0] - data.columns[pyfhd_header['date_i']].bzero
    else:
        pyfhd_header['jd0'] = header['PZERO{}'.format(pyfhd_header['date_i'] + 1)]

    # Take the julian date and use that in date_obs in the fits format
    if 'jd0' in pyfhd_header.keys():
        julian_time = Time(pyfhd_header['jd0'], format = 'jd')
        julian_time.format = 'fits'
        pyfhd_header['date_obs'] = julian_time.value
    # Probably won't reach here, if it does fill in jd0 from date_obs (fits to julian)
    elif 'date_obs' in pyfhd_header.keys():
        fits_time = Time(pyfhd_header['date_obs'], format = 'fits')
        fits_time.format = 'jd'
        pyfhd_header['jd0'] = fits_time.value

    return pyfhd_header

def create_params(pyfhd_header : np.recarray, pyfhd_config : dict, logger : logging.RootLogger) -> np.recarray:
    """_summary_

    Parameters
    ----------
    pyfhd_header : np.recarray
        _description_
    pyfhd_config : dict
        _description_
    logger : logging.RootLogger
        _description_

    Returns
    -------
    np.recarray
        _description_
    """

    pass