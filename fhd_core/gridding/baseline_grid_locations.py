import numpy as np
from fhd_utils.idl_tools.array_match import array_match
from fhd_utils.histogram import histogram

def baseline_grid_locations(obs, psf, params, vis_weights, bi_use = None, fi_use = None, 
                            fill_model_visibilities = False, interp_flag = False, mask_mirror_indices = False):
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
    bi_use : [type], optional
        [description], by default None
    fi_use : [type], optional
        [description], by default None
    fill_model_visibilities : bool, optional
        [description], by default False
    interp_flag : bool, optional
        [description], by default False
    mask_mirror_indices : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    
    Raises
    ------
    TypeError
        In the case obs, psf, params or vis_weights is None.
        TypeError is raised as the function needs to reference
        obs, psf, params and vis_weights.
    """

    #Check if obs, psf, params or vis_weights None, if so raise TypeError
    if obs is None or psf is None or params is None or vis_weights is None:
        raise TypeError("obs, psf, params and vis_weights must not be None")

    # Set up the return dictionary
    baselines_dict = {}

    # Retrieve information from the data structures
    n_tile = obs['n_tile']
    n_freq = obs['n_freq']
    dimension = obs['dimension']
    elements = obs['elements']
    kbinsize = obs['kpix']
    min_baseline = obs['min_baseline']
    max_baseline = obs['max_baseline']
    b_info = obs['baseline_info']
    psf_dim = psf['dim'][0]
    psf_resolution = psf['resolution'][0]

    # Frequency information of the visibilities
    if fill_model_visibilities:
        fi_use = np.arange(n_freq)
    elif fi_use is None:
        fi_use = np.nonzero(b_info[0]['freq_use'][0])[0]
    frequency_array = b_info[0]['freq'][0]
    frequency_array = frequency_array[fi_use]

    # At this point the vis_weight_switch is configured, 
    # but there's no point to us doing that because its never a pointer!
    # Is there something we ought to do instead?
    # What defines vis_weights as valid?
    # Do we bother with vis_weight_switch?

    # Set the bi_use_flag, if we set it here, we need to return it
    bi_use_flag = False
    # Baselines to use
    if bi_use is None:
        bi_use_flag = True
        # if the data is being gridded separately for the even/odd time samples
        # then force flagging to be consistent across even/odd sets
        if not fill_model_visibilities:
            flag_test = np.sum(vis_weights, axis = -1)
            bi_use = np.where(flag_test > 0)[0]
        else:
            tile_use = np.arange(n_tile) + 1
            bi_use, _ = array_match(b_info[0]['tile_a'][0], tile_use, array_2 = b_info[0]['tile_b'][0])
    
    # Calculate indices of visibilities to grid during this call (i.e. specific freqs, time sets)
    # and initialize output arrays
    n_b_use = bi_use.size
    n_f_use = fi_use.size
    # matrix_multiply is not what it seems for 1D arrays, had to do this to replicate!
    vis_inds_use = (np.outer(np.ones(n_b_use), fi_use) + np.outer(bi_use, np.ones(n_f_use)) * n_freq).astype(int)
    
    # Since the indices in vis_inds_use apply to a flattened array, flatten. Leave vis_inds_use as it to have the shape go back to the right shape.
    vis_weights = vis_weights.flatten()[vis_inds_use]

    # Units in pixel/Hz
    kx_arr = params['uu'][0][bi_use] / kbinsize
    ky_arr = params['vv'][0][bi_use] / kbinsize

    if not fill_model_visibilities:
        # Flag baselines on their maximum and minimum extent in the full frequency range of the observation
        # This prevents the sudden dissapearance of baselines along frequency
        dist_test = np.sqrt(kx_arr ** 2 + ky_arr ** 2) * kbinsize
        dist_test_max = np.max(obs['baseline_info'][0]['freq'][0]) * dist_test
        dist_test_min = np.min(obs['baseline_info'][0]['freq'][0]) * dist_test
        flag_dist_baseline = np.where((dist_test_min < min_baseline) | (dist_test_max > max_baseline))
        del(dist_test, dist_test_max, dist_test_min)
    
    # Create the other half of the uv plane via negating the locations
    conj_i = np.where(ky_arr > 0)[0]
    if conj_i.size > 0:
        kx_arr[conj_i] = -kx_arr[conj_i]
        ky_arr[conj_i] = -ky_arr[conj_i]

    # Center of baselines for x and y in units of pixels
    xcen = np.outer(kx_arr, frequency_array)
    ycen = np.outer(ky_arr, frequency_array)

    # Pixel number offset per baseline for each uv-box subset
    x_offset = np.fix(np.floor((xcen - np.floor(xcen)) * psf_resolution) % psf_resolution)
    y_offset = np.fix(np.floor((ycen - np.floor(ycen)) * psf_resolution) % psf_resolution)

    if interp_flag:
        # Derivatives from pixel edge to baseline center for use in interpolation
        dx_arr = (xcen - np.floor(xcen)) * psf_resolution - np.floor((xcen - np.floor(xcen)) * psf_resolution)
        dy_arr = (ycen - np.floor(ycen)) * psf_resolution - np.floor((ycen - np.floor(ycen)) * psf_resolution)
        baselines_dict['dx0dy0_arr'] = (1 - dx_arr) * (1 - dy_arr)
        baselines_dict['dx0dy1_arr'] = (1 - dx_arr) * dy_arr
        baselines_dict['dx1dy0_arr'] = dx_arr * (1 - dy_arr)
        baselines_dict['dx1dy1_arr'] = dx_arr * dy_arr

    # The minimum pixel in the uv-grid (bottom left of the kernel) that each baseline contributes to
    xmin = (np.floor(xcen) + elements / 2 - (psf_dim / 2 - 1)).astype(int)
    ymin = (np.floor(ycen) + dimension / 2 - (psf_dim / 2 - 1)).astype(int)

    # Set the minimum pixel value of baselines which fall outside of the uv-grid tom -1 to exclude them
    range_test_x_i = np.where((xmin <= 0) | (xmin + psf_dim - 1 >= elements - 1))
    range_test_y_i = np.where((ymin <= 0) | (ymin + psf_dim - 1 >= dimension - 1))
    if range_test_x_i[0].size > 0:
        xmin[range_test_x_i] = -1
        ymin[range_test_x_i] = -1
    if range_test_y_i[0].size > 0:
        xmin[range_test_y_i] = -1
        ymin[range_test_y_i] = -1

    # Flag baselines which fall outside the uv plane
    if not fill_model_visibilities:
        if flag_dist_baseline[0].size > 0:
            # If baselines fall outside the desired min/max baseline range at all during the frequency range
            # then set their maximum pixel value to -1 to exclude them
            xmin[flag_dist_baseline, :] = -1
            ymin[flag_dist_baseline, :] = -1
            del(flag_dist_baseline)
    
    # Normally we check vis_weight_switch, but its always true here so... do this
    flag_i = np.where(vis_weights <= 0)
    if fill_model_visibilities:
        n_flag = 0
    else:
        n_flag = flag_i[0].size
    if n_flag > 0:
        xmin[flag_i] = -1
        ymin[flag_i] = -1

    if mask_mirror_indices:
        # Option to exclude v-axis mirrored baselines
        if conj_i.size > 0:
            xmin[conj_i, :] = -1
            ymin[conj_i, :] = -1

    # If xmin or ymin is invalid then adjust the baselines dict as necessary
    if xmin.size == 0 or ymin.size == 0 or np.max([np.max(xmin), np.max(ymin)]) < 0:
        print('WARNING: All data flagged or cut!')
        baselines_dict['bin_n'] = 0
        baselines_dict['n_bin_use'] = 0
        baselines_dict['bin_i'] = -1
        baselines_dict['ri'] = 0
    else:
        # Match all visibilities that map from and to exactly the same pixels and store them as a histogram in bin_n
        # with their respective index ri. Setting min equal to 0, excludes flagged data (data set to -1).
        for_hist = xmin + ymin * dimension
        bin_n, _ , ri = histogram(for_hist, min = 0)
        bin_i = np.nonzero(bin_n)[0]

        # Update the baselines_dict which gets returned
        baselines_dict['bin_n'] = bin_n
        baselines_dict['bin_i'] = bin_i
        baselines_dict['n_bin_use'] = bin_i.size
        baselines_dict['ri'] = ri
    # Add values to baselines dict
    baselines_dict['xmin'] = xmin
    baselines_dict['ymin'] = ymin
    baselines_dict['vis_inds_use'] = vis_inds_use
    baselines_dict['x_offset'] = x_offset
    baselines_dict['y_offset'] = y_offset
    if bi_use_flag:
        baselines_dict['bi_use'] = bi_use
    
    # Return the dictionary
    return baselines_dict