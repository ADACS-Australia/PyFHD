import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.io.fits.hdu.table import BinTableHDU
from pathlib import Path
import logging
from typing import Tuple
from astropy.coordinates import EarthLocation
import astropy


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
    pyfhd_header['n_tile'] = 128
    pyfhd_header['naxis'] = params_header['naxis']
    pyfhd_header['n_params'] = params_header['pcount']
    pyfhd_header['nbaselines'] = params_header['gcount']
    pyfhd_header['n_complex'] = params_header['naxis2']
    pyfhd_header['n_pol'] = params_header['naxis3']
    pyfhd_header['n_freq'] = params_header['naxis4']
    pyfhd_header['freq_ref'] = params_header['crval4']
    pyfhd_header['freq_res'] = params_header['cdelt4']
    try:
        pyfhd_header['date_obs'] = params_header['date-obs']
    except KeyError:
        pyfhd_header['date_obs'] = params_header['dateobs']
    freq_ref_i = params_header['crpix4'] - 1
    pyfhd_header['frequency_array'] = (np.arange(pyfhd_header['n_freq']) - freq_ref_i) * pyfhd_header['freq_res'] + pyfhd_header['freq_ref']
    pyfhd_header['obsra'] = params_header['obsra']
    pyfhd_header['obsdec'] = params_header['obsdec']
    # Put in locations of instrument from FITS file or from Astropy site data
    try:
        location = EarthLocation.of_site(pyfhd_config['instrument'])
    except astropy.coordinates.errors.UnknownSiteException:
        ##TODO fix this MWA location thing, have some kind of built in
        ##locations in the repo?
        logger.info(f"Failed to load in the {pyfhd_config['instrument']} instrument location from astropy. If lon/lat/alt are not in the UVFITS things will fail.")
        
    try: 
        pyfhd_header['lon'] = params_header['lon']
    except KeyError:
        pyfhd_header['lon'] = location.lon.deg
    try: 
        pyfhd_header['lat'] = params_header['lat']
    except KeyError:
        pyfhd_header['lat'] = location.lat.deg
    try: 
        pyfhd_header['alt'] = params_header['alt']
    except KeyError:
        pyfhd_header['alt'] = location.height.value

    logger.info(f"Setting {pyfhd_config['instrument']} instrument location to: lon {pyfhd_header['lon']:.2f}, lat {pyfhd_header['lat']:.2f}, alt {pyfhd_header['alt']:.2f}")

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
        raise KeyError('One of these keys is missing from the UVFITS file: UU, VV, WW, BASELINE, DATE, check the log to see which one')
    
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

def create_params(pyfhd_header : dict, params_data : np.recarray, logger : logging.RootLogger) -> dict:
    """_summary_

    Parameters
    ----------
    pyfhd_header : dict
        The resulting header fom the fits file stored in a dictonary 
    params_data : np.recarray
        The data from the fits file as taken from astropy.io.fits.getdata
    logger : logging.RootLogger
        The PyFHD logger

    Returns
    -------
    params : dict
        The PyFHD params stored as a dictionary (instead of recarray as a dict is faster)

    Raises
    ======
    KeyError
        If the UVFITS data returned doesn't contain the variables then a KeyError will get thrown.
    
    See Also
    ========
    astropy.io.fits.getdata : https://docs.astropy.org/en/stable/io/fits/api/files.html#getdata
    extract_header : Extracts the header from the UVFITS file and returns the header and data 
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
    Extract the visibilities and their weights from the UVFITS data.

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
    extract_header : Extracts the header from the UVFITS file and returns the header and data 
    """

    data_array = np.squeeze(params_data['DATA'])
    # Set the number of polarizations
    if pyfhd_config['n_pol'] == 0:
        n_pol = pyfhd_header['n_pol']
    else:
        n_pol = pyfhd_config['n_pol']
    
    if data_array.ndim > 4:
        logger.error("No current support for PyFHD to support spectral dimensions yet")
        exit()
    else:
        polarizations = np.arange(n_pol)
        vis_arr = data_array[:, : , polarizations, pyfhd_header['real_index']] + data_array[:, : , polarizations, pyfhd_header['imaginary_index']] * (1j)
        vis_weights = data_array[:, : , polarizations, pyfhd_header['weights_index']]
    # Redo the shape so its the format per polarization, per frequency per baseline.
    return vis_arr.transpose(), vis_weights.transpose()

def _check_layout_valid(layout : dict, key : str, logger : logging.RootLogger, check_min_max = False):
    """
    Check if the key given is a valid part of the layout, if not give an error in the log.
    The errors do not stop the run as it might only affect compatibility with other packages and
    could be solved by editing or fixing the UVFITS file. 

    Parameters
    ----------
    layout : dict
        The current layout
    key : str
        The key we're interested in validating
    logger : logging.RootLogger
        The logger
    check_min_max : bool, optional
        When True check if the min is the same as max, if so changes the value so its only one number, by default False
    """

    if check_min_max:
        if type(layout[key]) == np.ndarray and np.min(layout[key]) == np.max(layout[key]):
            layout[key] = layout[key][0]
    
    if type(layout[key]) == np.ndarray and (layout[key].size != layout['n_antenna']):
        logger.error(f"The layout[{key}] array set is not the same size of the number of antennas. Check the UVFITS file for errors.")
    

def create_layout(antenna_table : BinTableHDU, logger : logging.RootLogger) -> dict:
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
    extract_header : Opens the UVFITS file and extracts the header and data, including the antenna_table.
    """
    antenna_data = antenna_table.data
    antenna_header = antenna_table.header
    layout = {}

    # Extract data from the header
    # array_center
    try: 
        layout['array_center'] = [antenna_header['arrayx'], antenna_header['arrayy'], antenna_header['arrayz']]
    except KeyError:
        # if no center given, assume MWA center (Tingay et al. 2013, converted from lat/lon using pyuvdata)
        logger.info("No center was given in the UVFITS file, assuming MWA is the array and using a default center for MWA")
        layout['array_center'] = [-2559454.07880307,  5095372.14368305, -2849057.18534633]
    
    # Coordinate_frame
    try:
        layout['coordinate_frame'] = antenna_header['frame']
    except KeyError:
        logger.info("Coordinate Frame is missing from the UVFITS file, using IRTF")
        layout['coordinate_frame'] = 'IRTF'
    
    # Greenwich Sidereal Time
    try:
        layout['gst0'] = antenna_header['gstia0']
    except KeyError:
        logger.warning("Greenwich sidereal time missing from UVFITS file gst0 will be -1")
        layout['gst0'] = -1

    # Earth's Rotation
    try:
        layout['earth_degpd'] = antenna_header['degpdy']
    except KeyError:
        logger.info("degpdy is missing from the UVFITS file, using 360.985 for Earth's rotation in degrees")
        layout['earth_degpd'] = 360.985

    # Reference Date
    try:
        layout['refdate'] = antenna_header['rdate']
    except KeyError:
        logger.warning("No refdate supplied in UVFITS file, set ref_date as -1")
        layout['refdate'] = -1

    # Time System
    try:
        layout['time_system'] = antenna_header['timesys'].strip()
    except KeyError:
        try:
            layout['time_system'] = antenna_header['timsys'].strip()
        except KeyError:
            logger.warning("No Time System supplied in UVFITS file setting time system as UTC")
            layout['time_system'] = 'UTC'
    
    # UT1UTC
    try:
        layout['dut1'] = antenna_header['ut1utc']
    except KeyError:
        logger.info("UT1UTC is mising from UVFITS, using 0")
        layout['dut1'] = 0

    # DATUTC
    try:
        layout['diff_utc'] = antenna_header['datutc']
    except:
        logger.info("No difference set between time_system and UTC, set to 0")
        layout['diff_utc'] = 0
    
    # Number of leap seconds
    try:
        layout['nleap_sec'] = antenna_header['iautc']
    except KeyError:
        if layout['time_system'] == 'IAT':
            logger.info("Time System is IAT and leap seconds is missing, using value from diff_utc(layout)/datutc(uvfits)")
            layout['nleap_sec'] = layout['diff_utc']
        else:
            logger.warning("Number of Leap Seconds is missing and the time system isn't IAT so we can't know the leap seconds, setting as -1")
    
    # Polarization Type
    try:
        layout['pol_type'] = antenna_header['poltype']
    except KeyError:
        logger.info("Polarization Type not in UVFITS file, Linear approximation for linear feeds is being used")
        layout['pol_type'] = 'X-Y LIN'

    # Polarization Characteristics
    try:
        layout['n_pol_cal_params'] = antenna_header['nopcal']
    except KeyError:
        logger.info("Polarization Characteristics of the feed not given in UVFITS file, Set n_pol_cal_params to 0")
        layout['n_pol_cal_params'] = 0
    
    # Number of antennas
    try:
        layout['n_antenna'] = antenna_header['naxis2']
    except KeyError:
        logger.info("Number of antennas missing from header, set 128 as per MWA")
        layout['n_antenna'] = 128

    # Extract data from the data table
    # Antenna Names
    try:
        layout['antenna_names'] = antenna_data['anname']
    except KeyError:
        logger.warning("Antenna Names missing, replacing with a string of numbers")
        layout['antenna_names'] = np.arange(layout['n_antenna']).astype(str)
    
    # Antenna Numbers
    try:
        layout['antenna_numbers'] = antenna_data['nosta']
    except KeyError:
        layout.warning("Antenna Numbers missing replacing with an array of numbers of range 1: n_antenna")
        layout['antenna_numbers'] = np.arange(1, layout['n_antenna'])
    
    # Antenna Coordinates
    try:
        layout['antenna_coords'] = antenna_data['stabxyz']
    except KeyError:
        logger.warning("Antenna Coordinates missing, replacing with a zero array of shape n_antenna, 3")
        layout['antenna_coords'] = np.zeros((layout['n_antenna'], 3))
    
    # Mount Type
    try:
        layout['mount_type'] = antenna_data['mntsta']
    except KeyError:
        logger.warning("No Mount Type set, mount_type has been set to 0")
        layout['mount_type'] = 0

    # Axis Offset
    try:
        layout['axis_offset'] = antenna_data['staxof']
    except KeyError:
        logger.warning('Axis Offset is not given, setting to 0')
        layout['axis_offset'] = 0
    
    # Feed Polarization of feed A (Pol A)
    try:
        layout['pola'] = antenna_data['poltya']
    except:
        logger.warning('Pol A polarization not given setting to X')
        layout['pola'] = 'X'
    
    # PolA Orientation
    try:
        layout['pola_orientation'] = antenna_data['polaa']
    except KeyError:
        logging.warning("PolA orientation not given setting to 0")
        layout['pola_orientation'] = 0
    
    # PolA params
    try:
        layout['pola_cal_params'] = antenna_data['polcala']
    except KeyError:
        logger.warning("PolA params is missing from the UVFITS, set to array of zeros of length n_pol_cal_params or 0")
        if layout['n_pol_cal_params'] > 1:
            layout['pola_cal_params'] = np.zeros(layout['n_pol_cal_params'])
        else:
            layout['pola_cal_params'] = 0

     # Feed Polarization of feed B (Pol B)
    try:
        layout['polb'] = antenna_data['poltyb']
    except:
        logger.warning('Pol B polarization not given setting to Y')
        layout['polb'] = 'Y'
    
    # PolB Orientation
    try:
        layout['polb_orientation'] = antenna_data['polab']
    except KeyError:
        logging.warning("PolB orientation not given setting to 0")
        layout['polb_orientation'] = 90
    
    # PolB params
    try:
        layout['polb_cal_params'] = antenna_data['polcalb']
    except KeyError:
        logger.warning("PolB params is missing from the UVFITS, set to array of zeros of length n_pol_cal_params or 0")
        if layout['n_pol_cal_params'] > 1:
            layout['polb_cal_params'] = np.zeros(layout['n_pol_cal_params'])
        else:
            layout['polb_cal_params'] = 0

    # Diameters
    try:
        layout['diameters'] = antenna_data['diameter']
    except KeyError:
        logger.info('Diameters not in UVFITS file continuing.')
    
    # Beam Full Width Half Maximum
    try:
        layout['beam_fwhm'] = antenna_data['beamfwhm']
    except KeyError:
        logger.info("Beam Full Width Half maximum not present in UVFITS continuing.")

    # Layout Validation
    _check_layout_valid(layout, 'antenna_names', logger)
    _check_layout_valid(layout, 'antenna_numbers', logger)
    _check_layout_valid(layout, 'mount_type', logger, check_min_max = True)
    _check_layout_valid(layout, 'axis_offset', logger, check_min_max = True)
    _check_layout_valid(layout, 'pola', logger)
    _check_layout_valid(layout, 'pola_orientation', logger, check_min_max = True)
    _check_layout_valid(layout, 'polb', logger)
    _check_layout_valid(layout, 'polb_orientation', logger, check_min_max = True)

    return layout

    