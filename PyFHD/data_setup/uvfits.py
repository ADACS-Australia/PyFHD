import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.io.fits.hdu.table import BinTableHDU
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
    params_data : np.recarray
        The data from the UVFITS file.
    antenna_table : astropy.io.fits.hdu.table.BinTableHDU
        The layout header and data which will be used in the create_layout function

    Raises
    ------
    KeyError
        If the UVFITS file doesn't contain all the data then a KeyError will be raised
    """

    # Retrieve all data from the observation
    observation = fits.open(Path(pyfhd_config['input_path'], pyfhd_config['obs_id'] + '.uvfits'))
    params_header = observation[0].header
    params_data = observation[0].data

    pyfhd_header = {}
    # Retrieve data from the params_header
    pyfhd_header['pol_dim'] = 2
    pyfhd_header['freq_dim'] = 4
    pyfhd_header['real_index'] = 0
    pyfhd_header['imaginary_index'] = 1
    pyfhd_header['weights_index'] = 2
    pyfhd_header['n_tile'] = 128.0
    pyfhd_header['naxis'] = params_header['naxis']
    pyfhd_header['n_params'] = params_header['pcount']
    pyfhd_header['nbaselines'] = params_header['gcount']
    pyfhd_header['n_complex'] = params_header['naxis2']
    pyfhd_header['n_pol'] = params_header['naxis3']
    pyfhd_header['n_freq'] = params_header['naxis4']
    pyfhd_header['freq_ref'] = params_header['crval4']
    pyfhd_header['freq_res'] = params_header['cdelt4']
    pyfhd_header['date_obs'] = params_header['date-obs']
    freq_ref_i = params_header['crpix4'] - 1
    pyfhd_header['freq_array'] = (np.arange(pyfhd_header['n_freq']) - freq_ref_i) * pyfhd_header['freq_res'] + pyfhd_header['freq_ref']
    pyfhd_header['obsra'] = params_header['obsra']
    pyfhd_header['obsdec'] = params_header['obsdec']
    # Put in locations of instrument
    pyfhd_header['lon'] = pyfhd_config['lon']
    pyfhd_header['lat'] = pyfhd_config['lat']
    pyfhd_header['alt'] = pyfhd_config['alt']

    # Setup params list and names
    param_list = []
    ptype_list = ['PTYPE{}'.format(i) for i in range(1, pyfhd_header['n_params'] + 1)]
    for ptype in ptype_list:
        param_list.append(params_header[ptype].strip())
    param_names = []
    for key in list(params_header.keys()):
        if key.startswith('CTYPE'):
            param_names.append(params_header[key].strip().lower())
        
    # Validate params list
    params_valid = True
    pyfhd_header['uu_i'] = 'UU' in param_list
    if not pyfhd_header['uu_i']:
        logger.error('Group parameter UU not found within uvfits params_header PTYPE keywords')
        params_valid = False
    
    pyfhd_header['vv_i'] = 'VV' in param_list
    if not pyfhd_header['vv_i']:
        logger.error('Group parameter VV not found within uvfits params_header PTYPE keywords')
        params_valid = False

    pyfhd_header['ww_i'] = 'WW' in param_list
    if not pyfhd_header['ww_i']:
        logger.error('Group parameter WW not found within uvfits params_header PTYPE keywords')
        params_valid = False
    
    pyfhd_header['ant1_i'] = 'ANTENNA1' in param_list
    pyfhd_header['ant2_i'] = 'ANTENNA2' in param_list
    if not pyfhd_header['ant1_i'] or not pyfhd_header['ant2_i']:
        pyfhd_header['baseline_i'] = param_list.index('BASELINE')
        if not pyfhd_header['baseline_i']:
            logger.error('Group parameter BASELINE (or ANTENNA1 and ANTENNA2) not found within uvfits params_header PTYPE keywords')
            params_valid = False
    
    pyfhd_header['date_i'] = param_list.index('DATE')
    if not pyfhd_header['date_i']:
        logger.error('Group parameter DATE not found within uvfits params_header PTYPE keywords')
        params_valid = False
    
    # Stop PyFHD if its not valid
    if not params_valid:
        raise KeyError('One of these keys is missing from the FITS file: UU, VV, WW, BASELINE, DATE, check the log to see which one')
    
    # Get the Julian Date
    if param_list.count('DATE') > 1:
        # This needs testing as Astropy scales automatically, which affects the DATE data read in, this should be the same though
        pyfhd_header['jd0'] = params_header['PZERO{}'.format(pyfhd_header['date_i'] + 1)] + params_data['DATE'][0] - params_data.columns[pyfhd_header['date_i']].bzero
    else:
        # This is the bzero value used to normalize the value in Astropy for date
        pyfhd_header['jd0'] = params_header['PZERO{}'.format(pyfhd_header['date_i'] + 1)]

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

    # Keep the layout header and data for the create_layout function
    antenna_table = observation[1]

    return pyfhd_header, params_data, antenna_table

def create_params(pyfhd_header : dict, params_data : np.recarray, logger : logging.RootLogger, antenna_mod_index = None) -> dict:
    """_summary_

    Parameters
    ----------
    pyfhd_header : dict
        The resulting header fom the fits file stored in a dictonary 
    params_data : np.recarray
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
    astropy.io.fits.getdata : https://docs.astropy.org/en/stable/io/fits/api/files.html#getdata
    extract_header : Extracts the header from the FITS file and returns the header and data 
    """
    params = {}
    # Retrieve params values
    try:
        params['uu'] = params_data['UU']
        params['vv'] = params_data['VV']
        params['ww'] = params_data['WW']
        # Astropy has already normalized the values by PZEROx, time in Julian
        params['time'] = params_data['DATE']
        # Get baseline and antenna arrays
        if pyfhd_header['baseline_i']:
            params['baseline_arr'] = params_data['BASELINE']
            params['antenna1'] = params['baseline_arr']
        # The antenna arrays already exist then take those
        if pyfhd_header['ant1_i'] and pyfhd_header['ant2_i']:
            params['antenna1'] = params_data['ANTENNA1']
            params['antenna2'] = params_data['ANTENNA2']
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

def extract_visibilities(pyfhd_header : dict, params_data : np.recarray, pyfhd_config : dict, logger : logging.RootLogger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the visibilities and their weights from the FITS data.

    Parameters
    ----------
    pyfhd_header : dict
        The resulting header fom the fits file stored in a dictonary
    params_data : np.recarray
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
    astropy.io.fits.getdata : https://docs.astropy.org/en/stable/io/fits/api/files.html#getdata
    extract_header : Extracts the header from the FITS file and returns the header and data 
    """

    data_array = np.squeeze(params_data['DATA'])
    # Set the number of polarizations
    if pyfhd_config['n_pol'] == 0:
        n_pol = pyfhd_header['n_pol']
    else:
        n_pol = min(pyfhd_config['n_pol'], pyfhd_header['n_pol'])
    
    if data_array.ndim > 4:
        logger.error("No current support for PyFHD to support spectral dimensions yet")
        exit()
    else:
        polarizations = np.arange(n_pol)
        vis_arr = data_array[:, : , polarizations, pyfhd_header['real_index']] + data_array[:, : , polarizations, pyfhd_header['imaginary_index']] * (1j)
        vis_weights = data_array[:, : , polarizations, pyfhd_header['weights_index']]
       
    return vis_arr, vis_weights

def create_layout(antenna_table : BinTableHDU) -> dict:
    """_summary_

    Parameters
    ----------
    antenna_table : BinTableHDU
        The second table 

    Returns
    -------
    layout : dict
        The layout dictionary which wil enable compatibility with pyuvdata

    See Also
    ---------
    extract_header : Opens the FITS file and extracts the header and data, including the antenna_table.
    """
    antenna_data = antenna_table.data
    antenna_header = antenna_table.header
    layout = {}
    # Extract data from the header
    try: 
        layout['array_center'] = [antenna_header['arrayx'], antenna_header['arrayy'], antenna_header['arrayz']]
    except KeyError:
        # if no center given, assume MWA center (Tingay et al. 2013, converted from lat/lon using pyuvdata)
        layout['array_center'] = [-2559454.07880307,  5095372.14368305, -2849057.18534633]
    try:
        layout['coordinate_frame'] = antenna_header['frame']
    except KeyError:
        layout['coordinate_frame'] = 'IRTF'
    layout['gst0'] = antenna_header['gstia0']
    layout['earth_degpd'] = antenna_header['egpdy']
    layout['refdate'] = antenna_header['rdate']
    layout['time_system'] = antenna_header['timesys']

    print(layout)

    