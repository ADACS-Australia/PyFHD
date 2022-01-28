from fhd_core.gridding.baseline_grid_locations import baseline_grid_locations
from fhd_core.gridding.conjugate_mirror import conjugate_mirror
import numpy as np

def visibility_count(obs, psf, params, vis_weights, fi_use = None, bi_use = None, mask_mirror_indices = False,
                     file_path_fhd = None, no_conjugate = False, fill_model_visibilities = False):
    """[summary]

    Parameters
    ----------
    obs : [type]
        [description]
    psf : [type]
        [description]
    params : [type]
        [description]
    vis_weights : [type]
        [description]
    xmin : [type]
        [description]
    ymin : [type]
        [description]
    fi_use : [type]
        [description]
    n_freq_use : [type]
        [description]
    bi_use : [type]
        [description]
    mask_miiror_indices : [type]
        [description]
    file_path_fhd : [type], optional
        [description], by default None
    no_conjugate : bool, optional
        [description], by default True
    fill_model_vis : bool, optional
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
    
    if obs is None or psf is None or params is None or vis_weights is None:
        raise TypeError("obs, psf, params or vis_weights should not be None")
    
    #Retrieve info from the data structures
    dimension = int(obs['dimension'][0])
    elements = int(obs['elements'][0])
    psf_dim = psf['dim'][0]

    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, bi_use = bi_use, fi_use = fi_use,
                                             mask_mirror_indices = mask_mirror_indices, fill_model_visibilities = fill_model_visibilities)
    # Retrieve the data we need from baselines_dict
    bin_n = baselines_dict['bin_n']
    bin_i = baselines_dict['bin_i']
    xmin = baselines_dict['xmin']
    ymin = baselines_dict['ymin']
    ri = baselines_dict['ri']
    n_bin_use = baselines_dict['n_bin_use']
    # Remove baselines_dict
    del(baselines_dict)

    uniform_filter = np.zeros((elements, dimension))
    # Get the flat iterations of the xmin and ymin arrays. 
    xmin_iter = xmin.flat
    ymin_iter = ymin.flat
    for bi in range(n_bin_use):
        idx = ri[ri[bin_i[bi]]]
        # Should all be the same, but don't want an array
        xmin_use = xmin_iter[idx] 
        ymin_use = ymin_iter[idx]
        # Please note that due to precision differences, the result will be different compared to IDL FHD
        uniform_filter[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim] += bin_n[bin_i[bi]]
        
    if not no_conjugate:
        uniform_filter = (uniform_filter + conjugate_mirror(uniform_filter)) / 2

    # TODO: Write uniform_filter to file? 
    # fhd_save_io
    
    return uniform_filter



    