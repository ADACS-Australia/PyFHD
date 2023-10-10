import numpy as np
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert
import PyFHD.gridding.gridding_utils as gridding_utils
from logging import RootLogger

def filter_uv_uniform(image_uv, vis_count: np.ndarray | None, 
                      obs: dict|None = None, params: dict|None = None, pyfhd_config: dict|None = None, 
                      logger: RootLogger|None = None,weights: dict|None = None, fi_use: dict|None = None, 
                      bi_use: dict|None = None, mask_mirror_indices: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO:_summary_

    Parameters
    ----------
    image_uv : np.ndarray
        An image, usually a dirty image from dirty_image_generate
    vis_count : np.ndarray | None
        TODO: _description_
    obs : dict | None, optional
        The observation dictonary, by default None
    params : dict | None, optional
        The params dictionary containg uu, vv, by default None
    pyfhd_config : dict | None, optional
        PyFHD's configuration dictionary, by default None
    logger : RootLogger | None, optional
        PyFHD's logger, by default None
    weights : dict | None, optional
        The weights array (aka vis_weights), by default None
    fi_use : dict | None, optional
        The frequency use index array, by default None
    bi_use : dict | None, optional
        The baseline use index array, by default None
    mask_mirror_indices : bool, optional
        TODO: _description_, by default False

    Returns
    -------
    tuple[image_uv_filtered: np.ndarray, filter_use: np.ndarray]
        The filtered image and the filter used to filter the image as NumPy arrays

    Raises
    ------
    TypeError
        In the case obs or params is None and vis_count is also None
    """

    # If you need the name, grab it from pyfhd_config where needed

    # This does not make use of fine-grained flagging, but relies on coarse flags from the obs structure 
    # (i.e. a list of tiles completely flagged, and of frequencies completely flagged)
    if vis_count is None:
        if obs is not None or params is not None:
            vis_count = gridding_utils.visibility_count(
                obs, params, weights, pyfhd_config, logger, fi_use, bi_use, mask_mirror_indices
            )
        else:
            if (weights is not None and np.size(weights) == np.size(image_uv)):
                vis_count = weights / np.min(weights[weights > 0])
            raise TypeError("obs and params must not be None when vis_count is None")
        
    # Get the parts of the filter we're using
    filter_use = weight_invert(vis_count, threshold = 1)
    # Get the weights index as well
    if weights is not None and np.size(weights) == np.size(image_uv):
        wts_i = np.nonzero(weights)
    else:
        wts_i = np.nonzero(filter_use)
    # Apply a mean normalization
    if np.size(wts_i) > 0:
        filter_use /= np.mean(filter_use[wts_i])
    else:
        filter_use /= np.mean(filter_use)
    #Return the filtered
    return image_uv * filter_use, filter_use