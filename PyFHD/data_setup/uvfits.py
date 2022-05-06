import numpy as np
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
import logging
from typing import Tuple

def extract_header(pyfhd_config : dict, logger : logging.RootLogger) -> Tuple[dict, np.recarray]:
    """_summary_

    Parameters
    ----------
    pyfhd_config : dict
        This is the config created from the argprase
    logger : logging.RootLogger
        The PyFHD logger

    Returns
    -------
    pyfhd_header : dict
        The result from the extraction of the header of the UVFITS file
    data : np.recarray
        The data from the UVFITS file.

    Raises
    ------
    KeyError
        If the UVFITS file doesn't contain all the data then a KeyError will be raised
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
        
    # Validate params list
    params_valid = True
    pyfhd_header['uu_i'] = 'UU' in param_list
    if not pyfhd_header['uu_i']:
        logger.error('Group parameter UU not found within uvfits header PTYPE keywords')
        params_valid = False
    
    pyfhd_header['vv_i'] = 'VV' in param_list
    if not pyfhd_header['vv_i']:
        logger.error('Group parameter VV not found within uvfits header PTYPE keywords')
        params_valid = False

    pyfhd_header['ww_i'] = 'WW' in param_list
    if not pyfhd_header['ww_i']:
        logger.error('Group parameter WW not found within uvfits header PTYPE keywords')
        params_valid = False
    
    pyfhd_header['ant1_i'] = 'ANTENNA1' in param_list
    pyfhd_header['ant2_i'] = 'ANTENNA2' in param_list
    if not pyfhd_header['ant1_i'] or not pyfhd_header['ant2_i']:
        pyfhd_header['baseline_i'] = param_list.index('BASELINE')
        if not pyfhd_header['baseline_i']:
            logger.error('Group parameter BASELINE (or ANTENNA1 and ANTENNA2) not found within uvfits header PTYPE keywords')
            params_valid = False
    
    pyfhd_header['date_i'] = param_list.index('DATE')
    if not pyfhd_header['date_i']:
        logger.error('Group parameter DATE not found within uvfits header PTYPE keywords')
        params_valid = False
    
    # Stop PyFHD if its not valid
    if not params_valid:
        raise KeyError('One of these keys is missing from the FITS file: UU, VV, WW, BASELINE, DATE, check the log to see which one')
    
    # Get the Julian Date
    if param_list.count('DATE') > 1:
        # This needs testing as Astropy scales automatically, which affects the DATE data read in, this should be the same though
        pyfhd_header['jd0'] = header['PZERO{}'.format(pyfhd_header['date_i'] + 1)] + data['DATE'][0] - data.columns[pyfhd_header['date_i']].bzero
    else:
        # This is the bzero value used to normalize the value in Astropy for date
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

    return pyfhd_header, data

def create_params(pyfhd_header : dict, fits_data : np.recarray, logger : logging.RootLogger, antenna_mod_index = None) -> dict:
    """_summary_

    Parameters
    ----------
    pyfhd_header : dict
        The resulting header fom the fits file stored in a dictonary 
    fits_data : np.recarray
        The data from the fits file as taken from astropy.io.fits.getdata
    logger : logging.RootLogger
        The PyFHD logger
    antenna_mod_index : int
        TODO: Requires description, by default None

    Returns
    -------
    params : dict
        The PyFHD params stored as a dictionary (instead of recarray as a dict is faster)

    Raises
    ======
    KeyError
        If the FITS data returned doesn't contain the variables then a KeyError will get thrown.
    
    See Also
    ========
    astropy.io.fits.getdata : Retrieves the Header and Data from a FITS file
    extract_header : Extracts the header from the FITS file and returns the header and data 
    """
    params = {}
    # Retrieve params values
    try:
        params['uu'] = fits_data['UU']
        params['vv'] = fits_data['VV']
        params['ww'] = fits_data['WW']
        # Astropy has already normalized the values by PZEROx, time in Julian
        params['time'] = fits_data['DATE']
        # Get baseline and antenna arrays
        if pyfhd_header['baseline_i']:
            params['baseline_arr'] = fits_data['BASELINE']
            params['antenna1'] = params['baseline_arr']
        # The antenna arrays already exist then take those
        if pyfhd_header['ant1_i'] and pyfhd_header['ant2_i']:
            params['antenna1'] = fits_data['ANTENNA1']
            params['antenna2'] = fits_data['ANTENNA2']
        # Else calculate it from the baseline array
        else:
            if antenna_mod_index is None:
                # Calculate antenna_mod_index to check for bad fits
                baseline_min = np.min(params['baseline_arr'])
                exponent = np.log(np.min(baseline_min)) / np.log(2)
                antenna_mod_index = 2 ** np.floor(exponent)
                tile_B_test = np.min(baseline_min) % antenna_mod_index
                if tile_B_test > 1:
                    if baseline_min % 2 == 1:
                        antenna_mod_index /= 2 ** np.floor(np.log(tile_B_test) / np.log(2))
            # Tile numbers start from 1
            params['antenna1'] = np.floor(params['baseline_arr'] / antenna_mod_index)
            params['antenna2'] = np.fix(params['baseline_arr'] % antenna_mod_index)

    except KeyError as error:
        logger.error(f"Validation efforts failed, key not found in data, Traceback : {error}")
        exit()

    return params

def extract_visibilities(pyfhd_header : dict, fits_data : np.recarray, pyfhd_config : dict, logger : logging.RootLogger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the visibilities and their weights from the FITS data.

    Parameters
    ----------
    pyfhd_header : dict
        The resulting header fom the fits file stored in a dictonary
    fits_data : np.recarray
        The data from the fits file as taken from astropy.io.fits.getdata
    pyfhd_config : dict
        This is the config created from the argprase
    logger : logging.RootLogger
        The PyFHD Logger

    Returns
    -------
    vis_arr : np.ndarray
        The visibility array
    vis_weights : np.ndarray
        The visibility weights array

    See Also
    ========
    astropy.io.fits.getdata : Retrieves the Header and Data from a FITS file
    extract_header : Extracts the header from the FITS file and returns the header and data 
    """

    data_array = np.squeeze(fits_data['DATA'])
    # Set the number of polarizations
    if pyfhd_config['n_pol'] == 0:
        n_pol = pyfhd_header['n_pol']
    else:
        n_pol = min(pyfhd_config['n_pol'], pyfhd_header['n_pol'])
    
    if data_array.ndim > 4:
        logger.error("No current support for PyFHD to support spectral dimensions yet")
        exit()
    else:
        vis_arr = np.zeros((pyfhd_header['nbaselines'], pyfhd_header['n_freq'], n_pol), dtype = np.complex128)
        vis_weights = np.zeros((pyfhd_header['nbaselines'], pyfhd_header['n_freq'], n_pol), dtype = np.float64)

    return vis_arr, vis_weights