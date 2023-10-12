import numpy as np
import deepdish as dd
from logging import RootLogger
from pathlib import Path
from astropy.io import fits
from datetime import datetime
from PyFHD.data_setup.obs import update_obs
from PyFHD.pyfhd_tools.unit_conv import pixel_to_radec
from PyFHD.pyfhd_tools.pyfhd_utils import meshgrid, rebin, weight_invert, region_grow, crosspol_split_real_imaginary
from PyFHD.gridding.gridding_utils import dirty_image_generate
from PyFHD.healpix.healpix_utils import healpix_cnv_generate
from healpy.pixelfunc import ring2nest

def get_image_renormalization(
    obs: dict, 
    weights: np.ndarray, 
    beam_base: np.ndarray, 
    filter_arr: np.ndarray, 
    pyfhd_config: dict, 
    logger: RootLogger
) -> np.ndarray:
    """
    TODO: _summary_

    Parameters
    ----------
    obs : dict
        _description_
    weights : np.ndarray
        _description_
    beam_base : np.ndarray
        _description_
    filter_arr : np.ndarray
        _description_
    pyfhd_config : dict
        _description_
    logger : RootLogger
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    renorm_factor = np.empty(obs["pol_i"])
    for pol_i in range(obs["n_pol"]):
        renorm_factor[pol_i] = 1 / dirty_image_generate(
            weights[pol_i], 
            pyfhd_config, 
            logger, 
            weights = weights[pol_i], 
            pad_uv_image = pyfhd_config['pad_uv_image'],
            filter = filter_arr[pol_i],
            beam_ptr = beam_base[pol_i]
        )[obs['dimension'] // 2, obs['elements'] // 2]
        # TODO: Check x and y indexing
        renorm_factor[pol_i] *= beam_base[pol_i, obs["obsx"], obs["obsy"]] ** 2
        renorm_factor[pol_i] /= (obs["degpix"] * (np.pi / 180)) ** 2
    return renorm_factor

def quickview(
    obs: dict,
    psf: dict,
    params: dict,
    cal: dict,
    vis_arr: np.ndarray,
    vis_weights: np.ndarray,
    image_uv: np.ndarray,
    weights_uv: np.ndarray,
    variance_uv: np.ndarray,
    uniform_filter_uv: np.ndarray,
    model_uv: np.ndarray,
    pyfhd_config: dict,
    logger: RootLogger
 ) -> None:
    # Save all the things into the output directory
    if pyfhd_config["save_obs"]:
        obs_path = Path(pyfhd_config["output_dir"],'obs.h5')
        logger.info(f"Saving the obs dictionary to {obs_path}")
        dd.io.save(obs_path, obs)
    if pyfhd_config["save_params"]:
        params_path = Path(pyfhd_config["output_dir"],'params.h5')
        logger.info(f"Saving params dictionary to {params_path}")
        dd.io.save(params_path, params)
    if pyfhd_config["save_visibilities"]:
        if pyfhd_config["recalculate-grid"]:
            uv_path = Path(pyfhd_config["output_dir"],'uv.h5')
            logger.info(f"Saving the gridded uv plane to {uv_path}")
            h5_save_dict = {
                "image": image_uv,
                "weights": weights_uv,
                "variance": variance_uv,
                "uniform_filter": uniform_filter_uv,
                "model": model_uv
            }
            dd.io.save(uv_path, h5_save_dict)
        cal_vis_arr_path = Path(pyfhd_config["output_dir"],'calibrated_vis_arr.h5')
        logger.info(f"Saving the calibrated visibilities to {cal_vis_arr_path}")
        dd.io.save(cal_vis_arr_path, {"visibilities": vis_arr})
    if pyfhd_config["save_cal"] and pyfhd_config["calibrate_visibilities"]:
        cal_path = Path(pyfhd_config["output_dir"],"cal.h5")
        logger.info(f"Saving the calibration dictionary to {cal_path}")
        dd.io.save(cal_path, cal)
    if pyfhd_config["save_calibrated_weights"]:
        weights_path = Path(pyfhd_config["output_dir"],"calibrated_vis_weights.h5")
        logger.info(f"Saving the calibrated weights to {weights_path}")
        dd.io.save(weights_path, {"weights": vis_weights})
    
    obs_out = update_obs(obs, obs['dimension'] * pyfhd_config['pad_uv_image'], obs['kpix'])
    horizon_mask = np.ones([obs_out['dimension'], obs_out['elements']])
    ra, _ = pixel_to_radec(
        meshgrid(obs_out["dimension"], obs_out["elements"], 1), 
        meshgrid(obs_out["dimension"], obs_out["elements"], 2),
        obs_out['astr']
    )
    horizon_test = np.isfinite(ra)
    horizon_mask[horizon_test] = 0

    # Calculate the beam mask and beam indexes associated with that mask
    beam_mask = np.ones([obs_out["dimension"], obs_out["elements"]])
    beam_avg = np.zeros([obs_out["dimension"], obs_out["elements"]]) 
    beam_base_out = np.empty([obs_out["n_pol"], obs_out["dimension"], obs_out["elements"]])
    beam_correction_out = np.empty_like(beam_base_out)
    for pol_i in obs["n_pol"]:
        beam_base_out[pol_i] = rebin(psf["beam_ptr"], [obs_out["dimension"], obs_out["elements"]]) * horizon_mask
        beam_correction_out[pol_i] = weight_invert(beam_base_out[pol_i], 1e-3)
        if (pol_i == 0):
            beam_mask_test = beam_base_out[pol_i]
            # Didn't see the option for allow_sidelobe_image_output in FHD dictionary defined or used anywhere?
            beam_i = region_grow(
                beam_mask_test, 
                np.array([obs_out["dimension"] / 2 + obs_out["dimension"] * obs_out["elements"] / 2]),
                low = 0.05 / 2, # This is beam_threshold/2 in FHD
                high = np.max(beam_mask_test)
            )
            beam_mask0 = np.zeros([obs_out["dimension"], obs_out["elements"]])
            beam_mask0.flat[beam_i] = 1
            beam_avg += beam_base_out[pol_i] ** 2
            beam_mask *= beam_mask0
    beam_avg /= min(obs["n_pol"],2)
    beam_avg = np.sqrt(np.maximum(beam_avg, 0)) * beam_mask
    beam_i = np.nonzero(beam_mask)

    if pyfhd_config["save_healpix_fits"]:
        FoV_use = (180 / np.pi) / obs_out["kpix"]
        hpx_cnv, obs_out = healpix_cnv_generate(obs_out, beam_mask, FoV_use / np.sqrt(2), pyfhd_config, logger)
        hpx_inds_nest = ring2nest(hpx_cnv['nside'], hpx_cnv['inds'])
    # Generate our dirty images of the uv planes
    instr_dirty_arr = np.empty([obs["n_pol"], obs["dimension"], obs["elements"]])
    instr_model_arr = np.empty([obs["n_pol"], obs["dimension"], obs["elements"]])
    filter_arr = np.zeros([obs["n_pol"], obs["dimension"], obs["elements"]])
    for pol_i in range(obs['n_pol']):
        complex_flag = pol_i > 1
        filter = np.empty(0)
        # Get the dirty image of the uv plane and the filter from filter_uv_uniform
        instr_dirty_arr[pol_i], filter, _ = dirty_image_generate(
            image_uv[pol_i], 
            pyfhd_config, 
            logger, 
            degpix = obs_out['degpix'],
            weights = vis_weights[pol_i],
            pad_uv_image = pyfhd_config['pad_uv_image'],
            filter = filter,
            not_real = complex_flag,
            beam_ptr = beam_base_out[pol_i]
        )
        filter_arr[pol_i] = filter
        instr_model_arr[pol_i], filter, _ = dirty_image_generate(
            model_uv[pol_i], 
            pyfhd_config, 
            logger, 
            degpix = obs_out['degpix'],
            weights = vis_weights[pol_i],
            pad_uv_image = pyfhd_config['pad_uv_image'],
            filter = filter,
            not_real = complex_flag,
            beam_ptr = beam_base_out[pol_i]
        )
    renorm_factor = get_image_renormalization(obs_out, vis_weights, beam_base_out, filter_arr, pyfhd_config, logger)
    # Reshape renorm factor to multiply per polarization without loop to [obs["n_pol"], 1, 1] 
    renorm_factor = np.expand_dims(renorm_factor.reshape([obs_out["n_pol"],1]), -1)
    instr_dirty_arr *= renorm_factor
    instr_model_arr *= renorm_factor
    instr_residual_arr = instr_dirty_arr - instr_model_arr
    # Get the pol_names
    pol_names = obs["pol_names"]
    # The cross-polarization XY and YX images are both complex, but are conjugate mirrors of each other
    # To make images of these, we simply take the real and imaginary parts separately
    if (obs_out["n_pol"] >= 4):
        logger.info("Peforming Cross Polarization splits for real and imaginary")
        instr_dirty_arr, pol_names = crosspol_split_real_imaginary(instr_dirty_arr, pol_names = pol_names)
        instr_model_arr,_ = crosspol_split_real_imaginary(instr_model_arr)
        instr_residual_arr = crosspol_split_real_imaginary(instr_residual_arr)
        # The weights should have been saved at this point and we only need them like this from here
        vis_weights = crosspol_split_real_imaginary(vis_weights)
    
    # Build a fits header
    logger.info("Building the FITS Header for all the FITS files")
    fits_header = fits.PrimaryHDU(instr_dirty_arr[0])
    # Write in the WCS into the header from the astr dictionary in obs_out
    fits_header.header.set("ctype1", obs_out["astr"]["ctype"][0].decode(), "Coordinate Type")
    fits_header.header.set("ctype2", obs_out["astr"]["ctype"][1].decode(), "Coordinate Type")
    fits_header.header.set("equinox", obs_out["astr"]["equinox"], "Equinox of Ref. Coord.")
    fits_header.header.set("equinox", obs_out["astr"]["equinox"], "Equinox of Ref. Coord.")
    fits_header.header.set("equinox", obs_out["astr"]["equinox"], "Equinox of Ref. Coord.")
    cd = obs_out["astr"]["cd"]
    cd[0,:] = cd[0, :] * obs_out["astr"]["cdelt"][0]
    cd[1, :] = cd[1, :] * obs_out["astr"]["cdelt"][1]
    fits_header.header.set("cd1_1", cd[0,0], "Degrees / Pixel")
    fits_header.header.set("cd1_2", cd[0,1], "Degrees / Pixel")
    fits_header.header.set("cd2_1", cd[1,0], "Degrees / Pixel")
    fits_header.header.set("cd2_2", cd[1,1], "Degrees / Pixel")
    fits_header.header.set("crpix1", int(obs_out["astr"]["crpix"][0]), "Reference Pixel in X")
    fits_header.header.set("crpix2", int(obs_out["astr"]["crpix"][1]), "Reference Pixel in Y")
    fits_header.header.set("crval1", obs_out["astr"]["crval"][0], "R.A. (degrees) of reference pixel")
    fits_header.header.set("crval2", obs_out["astr"]["crval"][1], "Declination of reference pixel")
    fits_header.header.set("pv2_1", obs_out["astr"]["pv2"][0], "Projection parameter 1")
    fits_header.header.set("pv2_2", obs_out["astr"]["pv2"][1], "Projection parameter 2")
    for i in range(obs_out["astr"]["pv1"].size):
        fits_header.header.set(f"pv1_{i}", obs_out["astr"]["pv1"][i], "Projection parameters")
    fits_header.header.set("mjd-obs", obs_out["astr"]["mjd_obs"], "Modified Julian day of observations")
    fits_header.header.set("date-obs", obs_out["astr"]["date_obs"], "Date of observations")
    fits_header.header.set("radecsys", obs_out["astr"]["radecsys"], "Reference Frame")
    fits_header.header.set(
        "history", 
        f"World Coordinate System parameters written by PyFHD: {datetime.datetime.now().strftime('%b %d %Y %H:%M:%S')}"
    )
    # Create the fits header to store the dirty images
    fits_header_apparent = fits_header.copy()
    fits_header_apparent.header.set("bunit", "Jy/sr (apparent)")

    # Create the fits header for the weights
    fits_header_uv = fits.PrimaryHDU(vis_weights[0])
    fits_header_uv.header.set("CD1_1", obs['kpix'], 'Wavelengths / Pixel')
    fits_header_uv.header.set("CD2_1", 0., 'Wavelengths / Pixel')
    fits_header_uv.header.set("CD1_2", 0., 'Wavelengths / Pixel')
    fits_header_uv.header.set("CD2_2", obs['kpix'], 'Wavelengths / Pixel')
    fits_header_uv.header.set("CRPIX1", obs_out['dimension'] / 2 + 1, 'Reference Pixel in X')
    fits_header_uv.header.set("CRPIX2", obs_out['elements'] / 2 + 1, 'Reference Pixel in Y')
    fits_header_uv.header.set("CRVAL1", 0., 'Wavelengths (u)')
    fits_header_uv.header.set("CRVAL2", 0., 'Wavelengths (v)')
    fits_header_uv.header.set("MJD-OBS", obs_out["astr"]["mjd_obs"], 'Modified Julian day of observation')
    fits_header_uv.header.set("DATE-OBS", obs_out["astr"]["date_obs"], 'Date of observation')

    x_inc = beam_i % obs_out["dimension"]
    y_inc = np.floor(beam_i / obs_out["dimension"])
    zoom_low = min(np.min(x_inc), np.min(y_inc))
    zoom_high = max(np.max(x_inc), np.max(y_inc))

    # If you need the beam_contour arrays add them here, lines 369-378 in fhd_quickview.pro

    for pol_i in range(obs["n_pol"]):
        logger.info(f"Saving the FITS files for polarization {pol_i}")
        instr_residual = instr_residual_arr[pol_i] * beam_correction_out[pol_i]
        instr_dirty = instr_dirty_arr[pol_i] * beam_correction_out[pol_i]
        instr_model = instr_model_arr[pol_i] * beam_correction_out[pol_i]
        beam_use = beam_base_out[pol_i]
        
        # Write the fits apparent files
    
        # Write the HEALPix Fits files
    
