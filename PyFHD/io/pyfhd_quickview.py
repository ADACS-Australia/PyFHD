import numpy as np
import deepdish as dd
from logging import RootLogger

def pyfhd_main_io(
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
    pass