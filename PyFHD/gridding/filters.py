import numpy as np
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert
import PyFHD.gridding.gridding_utils as gridding_utils
from logging import RootLogger

def filter_uv_uniform(image_uv, vis_count: np.ndarray | None, 
                      obs: dict|None = None, params: dict|None = None, pyfhd_config: dict|None = None, 
                      logger: RootLogger|None = None,weights: dict|None = None, fi_use: dict|None = None, 
                      bi_use: dict|None = None, mask_mirror_indices: bool = False):
    """
    Perform uniform weighting in {u,v} space.

    Parameters
    ----------
    image_uv : np.ndarray
        A 2D {u,v} gridded plane to be filtered
    vis_count : np.ndarray | None
        2D array of number of contributing visibilities per pixel on the {u,v} grid
    obs : dict | None, optional
        Observation metadata dictionary, by default None
    params : dict | None, optional
        Visibility metadata dictionary, by default None
    pyfhd_config : dict | None, optional
        Run option dictionary, by default None
    logger : RootLogger | None, optional
        PyFHD's logger, by default None
    weights : dict | None, optional
        The weights array (aka vis_weights), by default None
    fi_use : dict | None, optional
        Frequency index array for gridding, i.e. gridding all frequencies for continuum images, by default None
    bi_use : dict | None, optional
        Baseline index array for gridding, i.e even vs odd time stamps, by default None
    mask_mirror_indices : bool, optional
        Exclude baselines mirrored along the v-axis, by default False

    Returns
    -------
    tuple[image_uv_filtered: np.ndarray, filter_use: np.ndarray]
        The filtered 2D {u,v} plane and the filter used

    Raises
    ------
    TypeError
        In the case obs or params is None and vis_count is also None
    """

    # If you need the name, grab it from pyfhd_config where needed

    # This does not make use of fine-grained flagging, but relies on coarse flags from the obs structure 
    # (i.e. a list of tiles completely flagged, and of frequencies completely flagged)
    if vis_count is None:
        if obs is None or params is None:
            raise TypeError("obs and params must not be None when vis_count is None")
        vis_count = gridding_utils.visibility_count(
            obs, params, weights, pyfhd_config, logger, fi_use, bi_use, mask_mirror_indices
        )
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