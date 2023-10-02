import numpy as np
import deepdish as dd
from logging import RootLogger
from pathlib import Path
from PyFHD.data_setup.obs import update_obs

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

    if pyfhd_config["pad_uv_image"]:
        obs = update_obs(obs, obs['dimension'] * pyfhd_config['pad_uv_image'], obs['kpix'])
