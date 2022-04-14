import numpy as np
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert
import PyFHD.gridding.gridding_utils as gridding_utils
from PyFHD.io.pyfhd_save import save

def filter_uv_uniform(image_uv, vis_count = None, obs = None, psf = None, params = None, weights = None, fi_use = None, bi_use = None, 
                      mask_mirror_indices = False, name = "uniform", return_name_only = False):
    """[summary]

    Parameters
    ----------
    image_uv : [type]
        [description]
    vis_count : [type], optional
        [description], by default None
    obs : [type], optional
        [description], by default None
    psf : [type], optional
        [description], by default None
    params : [type], optional
        [description], by default None
    weights : [type], optional
        [description], by default None
    fi_use : [type], optional
        [description], by default None
    bi_use : [type], optional
        [description], by default None
    mask_mirror_indices : [type], optional
        [description], by default None
    name : str, optional
        [description], by default "uniform"
    return_name_only : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    TypeError
        [description]
    """

    # Return the name now, if that has been set
    if return_name_only:
        # Why do you want this?
        return image_uv, name
    # This does not make use of fine-grained flagging, but relies on coarse flags from the obs structure 
    # (i.e. a list of tiles completely flagged, and of frequencies completely flagged)
    if vis_count is None:
        if obs is None or psf is None or params is None:
            raise TypeError("obs, psf and params must not be None")
        vis_count = gridding_utils.visibility_count(obs, psf, params, weights, fi_use, bi_use, mask_mirror_indices)
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