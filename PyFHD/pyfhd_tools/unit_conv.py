from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.wcs import WCS
from astropy.time import Time
from astropy import units as u
import numpy as np

def altaz_to_radec(alt : float, az : float, lat : float, lon : float, height : float, time : float, time_format = 'jd') -> tuple[float, float]:
    """
    Turn AltAz coordinates into the equatorial/celestial coordinates RA and DEC.
    The exact location and time must given in order for the coordinates to be calculated.

    Parameters
    ----------
    alt : float
        The altitude in degrees
    az : float
        The azimuth in degrees
    lat : float
        The latitude of the location (default is MWA's latitude)
    lon : float
        The longitude of the location (default is MWA's longitude)
    height : float
        The altitude of the location in metres above sea level (default is MWA's altitude above sea level)
    time : float
        The time from the UVFITS file
    time_format : str
        The time format given, as per AstroPy's Time Object, by default jd (Julian)

    Returns
    -------
    ra, dec : float, float
        The corresponding Equatorial Coordinates from the given location and time with altitude and azimuth
    """

    loc = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height = height*u.meter)
    altaz = AltAz(alt = alt*u.deg, az = az*u.deg, location = loc, obstime = Time(time, format=time_format))
    return altaz.transform_to(ICRS()).ra.deg, altaz.transform_to(ICRS()).dec.deg

def radec_to_altaz(ra : float, dec : float, lat : float, lon : float, height : float, time : float, time_format = 'jd') -> tuple[float, float]:
    """
    Turn Celestial/Equatorial coordinates into AltAz at the given location and time.
    Time Format by default is Julian, but you can use any of the formats provided by the AstroPy Time classes.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    lat : float
        The latitude in degrees (default is MWA's latitude)
    lon : float
        The longtitude in degrees (default is MWA's longitude)
    height : float
        The height of the location in metres above sea level (default is MWA's altitude above sea level)
    time : float
        The time
    time_format : str, optional
        The format of the time, by default 'jd' (Julian)

    Returns
    -------
    alt, az
        The corresponding Altitude and Azimuth coordinates for the given location, time and celestial coordinates
    """

    # Create the Earth Location
    loc = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, height = height*u.meter)
    loc_time = Time(time, format = time_format)
    radec = SkyCoord(ra = ra * u.deg, dec = dec * u.deg)
    altaz = radec.transform_to(AltAz(location = loc, obstime = loc_time))
    return altaz.alt.deg, altaz.az.deg

def radec_to_pixel(ra : float, dec : float, astr : dict) -> tuple[float, float]:
    """
    Turn Celestial Coordinates into Pixel coordinates (X & Y). The astr dictionary should contain
    cdelt, ctype, crpix and crval as per the WCS standard when naxis is 2.

    Parameters
    ----------
    ra : float
        The Right Ascension in degrees
    dec : float
        The Declination in degrees
    astr : dict
        The astrometry dictionary used in PyFHD.

    Returns
    -------
    x, y : float, float
        The pixel coordinates for the given celestial coordinate.
    """

    # Create WCS object with astr
    wcs_astr = WCS(naxis  = 2)
    wcs_astr.wcs.cdelt = astr['cdelt']
    wcs_astr.wcs.ctype = astr['ctype']
    wcs_astr.wcs.crpix = astr['crpix']
    wcs_astr.wcs.crval = astr['crval']
    # Now use world_to_pixel function for WCS objects
    x, y = wcs_astr.world_to_pixel(SkyCoord(ra = ra*u.deg, dec = dec*u.deg))
    # AstroPy returns values as an array, cast to float
    return float(x), float(y)

def pixel_to_radec(x: float | np.ndarray, y: float | np.ndarray, astr: dict) -> tuple[float, float]:
    """
    Turn Pixel coordinates (X & Y) into Celestial Coordinates based off a WCS. 
    The astr dictionary should contain cdelt, ctype, crpix and crval as per the 
    WCS standard when naxis is 2.

    Parameters
    ----------
    x : float | np.ndarray
        x coordinate(s)
    y : float | np.ndarray
        y coordinate(s)
    astr : dict
        The astrometry dictionary from an obs dictionary

    Returns
    -------
    ra, dec : tuple[float | np.ndarray, float | np.ndarray]
        The corresponding celestial cooridnates
    """
    wcs_astr = WCS(naxis  = 2)
    wcs_astr.wcs.cdelt = astr['cdelt']
    wcs_astr.wcs.ctype = astr['ctype']
    wcs_astr.wcs.crpix = astr['crpix']
    wcs_astr.wcs.crval = astr['crval']
    radec = wcs_astr.pixel_to_world(x, y)

    return radec.ra, radec.dec