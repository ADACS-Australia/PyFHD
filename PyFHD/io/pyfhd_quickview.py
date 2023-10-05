import numpy as np
import deepdish as dd
from logging import RootLogger
from pathlib import Path
from PyFHD.data_setup.obs import update_obs
from PyFHD.pyfhd_tools.unit_conv import pixel_to_radec
from PyFHD.pyfhd_tools.pyfhd_utils import meshgrid, rebin, weight_invert, region_grow

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
    model_uv: np.ndarray|None,
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
    if pyfhd_config["save-cal"] and pyfhd_config["calibrate_visibilities"]:
        cal_path = Path(pyfhd_config["output_dir"],"cal.h5")
        logger.info(f"Saving the calibration dictionary to {cal_path}")
        dd.io.save(cal_path, cal)
    if pyfhd_config["save-calibrated-weights"]:
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
        # hpx_cnv
        # ring2nest

