import numpy as np
import PyFHD.gridding.filters
from PyFHD.pyfhd_tools.pyfhd_utils import rebin, histogram, array_match, meshgrid
from scipy.signal import convolve
from astropy.convolution import Box2DKernel
from math import pi
from logging import RootLogger

def interpolate_kernel(kernel_arr: np.ndarray, x_offset: np.ndarray, y_offset: np.ndarray, 
                       dx0dy0: np.ndarray, dx1dy0: np.ndarray, dx0dy1: np.ndarray, dx1dy1: np.ndarray):
    """
    TODO: Description

    Parameters
    ----------
    kernel_arr: np.ndarray
        The array we are applying the kernel too
    x_offset: np.ndarray
        x_offset array
    y_offset: np.ndarray
        y_offset array
    dx0dy0: np.ndarray
        TODO: description
    dx1dy0: np.ndarray
        TODO: Description
    dx0dy1: np.ndarray
        TODO: Description
    dx1dy1: np.ndarray
        TODO: Description

    Returns
    -------
    kernel: np.ndarray
        TODO: Description
    """
    # x_offset and y_offset needed to be swapped around as IDL is column-major, while Python is row-major
    kernel = kernel_arr[y_offset, x_offset] * dx0dy0
    kernel += kernel_arr[y_offset, x_offset + 1] * dx1dy0
    kernel += kernel_arr[y_offset + 1, x_offset] * dx0dy1
    kernel += kernel_arr[y_offset + 1, x_offset + 1] * dx1dy1

    return kernel

def conjugate_mirror(image: np.ndarray):
    """
    This takes a 2D array and mirrors it, shifts it and
    its an array of complex numbers its get the conjugates
    of the 2D array

    Parameters
    ----------
    image: np.ndarray
        A 2D array of real or complex numbers
    
    Returns
    -------
    conj_mirror_image: np.ndarray
        The mirrored and shifted image array
    """
    # Flip image left to right (i.e. flips columns) & Flip image up to down (i.e. flips rows)
    conj_mirror_image = np.flip(image)
    # Shifts columns then rows by 1
    conj_mirror_image = np.roll(conj_mirror_image ,  1, axis = (1,0))
    # If any of the array is complex, or its a complex array, get the conjugates
    if np.iscomplexobj(image):   
        conj_mirror_image = np.conjugate(conj_mirror_image)
    return conj_mirror_image

def baseline_grid_locations(obs: dict, params: dict, vis_weights: np.ndarray, pyfhd_config: dict, logger: RootLogger, 
                            bi_use: np.ndarray|None = None, fi_use: np.ndarray|None = None, fill_model_visibilities: bool = False, 
                            interp_flag: bool = False, mask_mirror_indices: bool = False):
    """
    TODO: Docstring
    [summary]

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
    psf_dim = pyfhd_config['psf_dim']
    psf_resolution = pyfhd_config['psf_resolution']

    # Frequency information of the visibilities
    if fill_model_visibilities:
        fi_use = np.arange(n_freq)
    elif fi_use is None:
        fi_use = np.nonzero(b_info['freq_use'])[0]
    frequency_array = b_info['freq']
    frequency_array = frequency_array[fi_use]

    # Set the bi_use_flag, if we set it here, we need to return it
    bi_use_flag = False
    # Baselines to use
    if bi_use is None:
        bi_use_flag = True
        # if the data is being gridded separately for the even/odd time samples
        # then force flagging to be consistent across even/odd sets
        if not fill_model_visibilities:
            flag_test = np.sum(np.maximum(vis_weights, 0), axis = 0)
            bi_use = np.nonzero(flag_test)[0]
        else:
            tile_use = np.arange(n_tile) + 1
            bi_use = array_match(b_info['tile_a'].astype(int), tile_use, array_2 = b_info['tile_b'].astype(int))
    
    # Rather than calculating the flat indexes we want, lets just index the array
    # by the frequency use and baseline_use indexes
    vis_weights = vis_weights[fi_use, :][:, bi_use]

    # Units in pixel/Hz
    kx_arr = params['uu'][bi_use] / kbinsize
    ky_arr = params['vv'][bi_use] / kbinsize

    if not fill_model_visibilities:
        # Flag baselines on their maximum and minimum extent in the full frequency range of the observation
        # This prevents the sudden dissapearance of baselines along frequency
        dist_test = np.sqrt(kx_arr ** 2 + ky_arr ** 2) * kbinsize
        dist_test_max = np.max(obs['baseline_info']['freq']) * dist_test
        dist_test_min = np.min(obs['baseline_info']['freq']) * dist_test
        flag_dist_baseline = np.where((dist_test_min < min_baseline) | (dist_test_max > max_baseline))
        del(dist_test, dist_test_max, dist_test_min)
    
    # Create the other half of the uv plane via negating the locations
    conj_i = np.where(ky_arr > 0)[0]
    if conj_i.size > 0:
        kx_arr[conj_i] = -kx_arr[conj_i]
        ky_arr[conj_i] = -ky_arr[conj_i]

    # Center of baselines for x and y in units of pixels
    xcen = np.outer(frequency_array, kx_arr)
    ycen = np.outer(frequency_array, ky_arr)

    # Pixel number offset per baseline for each uv-box subset
    x_offset = np.fix(np.floor((xcen - np.floor(xcen)) * psf_resolution) % psf_resolution).astype(np.int64)
    y_offset = np.fix(np.floor((ycen - np.floor(ycen)) * psf_resolution) % psf_resolution).astype(np.int64)

    if interp_flag:
        # Derivatives from pixel edge to baseline center for use in interpolation
        dx_arr = (xcen - np.floor(xcen)) * psf_resolution - np.floor((xcen - np.floor(xcen)) * psf_resolution)
        dy_arr = (ycen - np.floor(ycen)) * psf_resolution - np.floor((ycen - np.floor(ycen)) * psf_resolution)
        baselines_dict['dx0dy0_arr'] = (1 - dx_arr) * (1 - dy_arr)
        baselines_dict['dx0dy1_arr'] = (1 - dx_arr) * dy_arr
        baselines_dict['dx1dy0_arr'] = dx_arr * (1 - dy_arr)
        baselines_dict['dx1dy1_arr'] = dx_arr * dy_arr

    # The minimum pixel in the uv-grid (bottom left of the kernel) that each baseline contributes to
    xmin = (np.floor(xcen) + elements / 2 - (psf_dim / 2 - 1)).astype(np.int64)
    ymin = (np.floor(ycen) + dimension / 2 - (psf_dim / 2 - 1)).astype(np.int64)

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
        if np.size(flag_dist_baseline) > 0:
            # If baselines fall outside the desired min/max baseline range at all during the frequency range
            # then set their maximum pixel value to -1 to exclude them
            xmin[:, flag_dist_baseline] = -1
            ymin[:, flag_dist_baseline] = -1
            del(flag_dist_baseline)
    
    # Normally we check vis_weight_switch, but its always true here so... do this
    flag_i = np.where(vis_weights <= 0)
    if fill_model_visibilities:
        n_flag = 0
    else:
        n_flag = np.size(flag_i)
    if n_flag > 0:
        xmin[flag_i] = -1
        ymin[flag_i] = -1

    if mask_mirror_indices:
        # Option to exclude v-axis mirrored baselines
        if conj_i.size > 0:
            xmin[:, conj_i] = -1
            ymin[:, conj_i] = -1

    # If xmin or ymin is invalid then adjust the baselines dict as necessary
    if xmin.size == 0 or ymin.size == 0 or np.max([np.max(xmin), np.max(ymin)]) < 0:
        logger.warning('All data flagged or cut!')
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
    baselines_dict['x_offset'] = x_offset
    baselines_dict['y_offset'] = y_offset
    if bi_use_flag:
        baselines_dict['bi_use'] = bi_use
    
    # Return the dictionary
    return baselines_dict

def dirty_image_generate(dirty_image_uv, mask = None, baseline_threshold = 0, normalization = None,
                         resize = None, width_smooth = None, degpix = None, not_real = False,
                         image_filter_fn = 'filter_uv_uniform', pad_uv_image = None, filter = None,
                         beam_ptr = None):
    """
    TODO: Docstring
    [summary]

    Parameters
    ----------
    dirty_image_uv : [type]
        [description]
    mask : [type], optional
        [description], by default None
    baseline_threshold : int, optional
        [description], by default 0
    normalization : [type], optional
        [description], by default None
    resize : [type], optional
        [description], by default None
    width_smooth : [type], optional
        [description], by default None
    degpix : [type], optional
        [description], by default None
    real : bool, optional
        [description], by default False
    image_filter_fn : str, optional
        [description], by default 'filter_uv_uniform'
    pad_uv_image : [type], optional
        [description], by default None
    filter : [type], optional
        [description], by default None
    beam_ptr : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """

    # dimension is columns, elements is rows
    elements, dimension = dirty_image_uv.shape
    di_uv_use = dirty_image_uv
    # If the baseline threshold has been set
    if baseline_threshold is not None:
        # If width smooth hasn't been set, set it
        if width_smooth is None:
            width_smooth = np.floor(np.sqrt(dimension * elements) / 100)    
        rarray = np.sqrt((meshgrid(dimension, 1) - dimension / 2) ** 2 + (meshgrid(elements, 2) - elements / 2) ** 2)
        # Get all the values that meet the threshold
        if baseline_threshold >= 0:
            cut_i = np.where(rarray.flatten() < baseline_threshold)
        else:
            cut_i = np.where(rarray.flatten() > np.abs(baseline_threshold))
        # Create the mask array of ones
        mask_bt = np.ones((elements, dimension))
        # If there are values from cut, then use all those here and replace with 0
        if np.size(cut_i) > 0:
            mask_bt_flatiter = mask_bt.flat
            mask_bt_flatiter[cut_i] = 0
        if width_smooth is not None:
            # Get the kernel width
            kernel_width = np.max([width_smooth,1])
            # In IDL if the kernel width is even one is added to make it odd
            if kernel_width % 2 == 0:
                kernel_width += 1
            # Use a box width averaging filter over the mask, use valid so we can insert it in later
            box_averages = convolve(mask_bt, Box2DKernel(kernel_width), mode = 'valid')
            # Since IDL SMOOTH edges by default are the edges of the array used, ignore edges (its the reason why we used a valid convolve)
            start = int(kernel_width // 2)
            end = int(mask_bt.shape[1] - (kernel_width // 2))
            mask_bt[start : end, start : end] = box_averages  
        # Apply boxed mask to the dirty image
        di_uv_use *=  mask_bt
    
    # If a mask was supplied use that too
    if mask is not None:
        di_uv_use *= mask
    
    # If a filter was supplied as a numpy array (we can adjust this to support different formats)
    if filter is not None:
        if isinstance(filter, np.ndarray):
            # If the filter is already the right size, use it
            if np.size(filter) == np.size(di_uv_use):
                di_uv_use *= filter
            # Otherwise use a filter function
            else:
                # TODO: Add if, elif, else block for pyfhd_config["image_filter"] which can have 
                # 'filter_uv_uniform', 'filter_uv_hanning', 'filter_uv_natural', 'filter_uv_radial', 
                # 'filter_uv_tapered_uniform', 'filter_uv_optimal'
                di_uv_use, _ = eval("filters.{}(di_uv_use, vis_count = vis_count, obs = obs, psf = psf, params = params, weights = weights, fi_use = fi_use, bi_use = bi_use, mask_mirror_indices = mask_mirror_indices)".format(image_filter_fn))
    
    # Resize the dirty image by the factor resize    
    if resize is not None:
        dimension *= resize
        elements *= resize
        di_uv_real = di_uv_use.real
        di_uv_img = di_uv_use.imag
        # Use rebin to resize, apply to real and complex separately
        di_uv_real = rebin(di_uv_real, (elements, dimension))
        di_uv_img = rebin(di_uv_img, (elements, dimension))
        # Combine real and complex back together
        di_uv_use = di_uv_real + di_uv_img * 1j
    
    # Apply padding if it was supplied
    if pad_uv_image is not None:
        # dimension_new = int(np.max([np.max([dimension, elements]) * pad_uv_image, np.max([dimension, elements])]))
        # di_uv1 = np.zeros((dimension_new, dimension_new), dtype = "complex")
        # di_uv1[dimension_new // 2 - elements // 2 : dimension_new // 2 + elements // 2,
        #        dimension_new // 2 - dimension // 2 : dimension_new // 2 + dimension // 2] = di_uv_use
        di_uv1 = np.pad(di_uv_use, np.max([dimension, elements]) // 2)
        di_uv_use = di_uv1 * (pad_uv_image ** 2)
    
    # FFT normalization
    if degpix is not None:
        di_uv_use /= np.radians(degpix) ** 2

    # Multivariate Fast Fourier Transform
    dirty_image = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(di_uv_use), norm = "forward"))
    if not not_real:
        dirty_image = dirty_image.real
    
    # filter_uv_optimal produces images that are weighted by one factor of the beam
    # Weight by an additional factor of the beam to align with FHD's convention
    if image_filter_fn == 'filter_uv_optimal' and beam_ptr is not None:
        dirty_image *= beam_ptr
    
    # If we are returning complex, make sure its complex
    if not_real:
        dirty_image = dirty_image.astype("complex")
    else:
        dirty_image = dirty_image.real
    # Normalize by the matrix given, if it was given
    if normalization is not None:
        dirty_image *= normalization
        return dirty_image, normalization
    #Return
    return dirty_image  

def grid_beam_per_baseline(psf, uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,
                           xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot, 
                           image_top, psf_dim3, box_matrix, vis_n, beam_clip_floor = False, beam_int = None, 
                           beam2_int = None, n_grp_use = None, degrid_flag = False, obs = None, params = None,
                           weights = None, fi_use = None, bi_use = None, mask_mirror_indices = False):
    """
    TODO: Docstring

    Parameters
    ----------
    psf : [type]
        [description]
    uu : [type]
        [description]
    vv : [type]
        [description]
    ww : [type]
        [description]
    l_mode : [type]
        [description]
    m_mode : [type]
        [description]
    n_tracked : [type]
        [description]
    frequency_array : [type]
        [description]
    x : [type]
        [description]
    y : [type]
        [description]
    xmin_use : [type]
        [description]
    ymin_use : [type]
        [description]
    freq_i : [type]
        [description]
    bt_index : [type]
        [description]
    polarization : [type]
        [description]
    fbin : [type]
        [description]
    image_bot : [type]
        [description]
    image_top : [type]
        [description]
    psf_dim3 : [type]
        [description]
    box_matrix : [type]
        [description]
    vis_n : [type]
        [description]
    beam_int : [type], optional
        [description], by default None
    beam2_int : [type], optional
        [description], by default None
    n_grp_use : [type], optional
        [description], by default None
    degrid_flag : bool, optional
        [description], by default False
    beam_clip_floor : bool, optional
        [description], by default False
    
    Returns
    -------
    box_matrix: array
        [description]
    """

    # Make the beams on the fly with corrective phases given the baseline location. 
    # Will need to be rerun for every baseline, so speed is key.
    # For more information, see Jack Line's thesis

    # Loop over all visibilities that fall within the chosen visibility box
    for ii in range(vis_n):
        # Pixel center offset phases
        deltau_l = l_mode * (uu[bt_index[ii]] * frequency_array[freq_i[ii]] - x[xmin_use + psf['dim'][0] // 2])
        deltav_m = m_mode * (vv[bt_index[ii]] * frequency_array[freq_i[ii]] - y[ymin_use + psf['dim'][0] // 2])
        # w term offset phase
        w_n_tracked = n_tracked * ww[bt_index[ii]] * frequency_array[freq_i[ii]]

        # Generate a UV beam from the image space beam, offset by calculated phases
        psf_base_superres = dirty_image_generate(
            psf['image_info'][0]['image_power_beam_arr'][fbin[ii]][polarization] * \
            np.exp(2 * pi * (0 + 1j) * \
            (-w_n_tracked + deltau_l + deltav_m)),
            not_real = True,
            obs = obs,
            params = params, 
            weights = weights, 
            fi_use = fi_use, 
            bi_use = bi_use,
            mask_mirror_indices = mask_mirror_indices
        )
        psf_base_superres = psf_base_superres[image_bot: image_top + 1, image_bot : image_top + 1]

        # A quick way to sum down the image by a factor of 2 in both dimensions.
        # A 4x4 example where we sum down by a factor of 2
        # 
        # 1  2  3  4           1  2           1  2           1  2  5  6            14 46           14 22
        # 5  6  7  8    -->    3  4    -->    5  6    -->    9  10 13 14    -->    22 54    -->    46 54
        # 9  10 11 12                         9  10
        # 13 14 15 16          5  6           13 14          3  4  7  8
        #                      7  8                          11 12 15 16
        #                                     3  4
        #                      9  10          7  8
        #                      11 12          11 12
        #                                     15 16
        #                      13 14
        #                      15 16   
        d = psf_base_superres.shape
        # Note columns and rows are swapped from IDL so nx is now rows!
        nx = d[0] // psf['resolution'][0]
        ny = d[1] // psf['resolution'][0]
        # The same result of IDL in numpy is np.reshape, with shape swapping rows and columns, then doing transpose of this shape
        psf_base_superres = np.reshape(psf_base_superres,[psf['resolution'][0] * ny, nx, psf['resolution'][0]])
        psf_base_superres = np.transpose(psf_base_superres, [1,0,2])
        psf_base_superres = np.reshape(psf_base_superres, [ny, nx, psf['resolution'][0] ** 2])
        psf_base_superres = np.sum(psf_base_superres, -1)
        psf_base_superres = np.transpose(psf_base_superres)

        psf_base_superres = np.reshape(psf_base_superres, psf['dim'] ** 2)
        start = psf_dim3 * ii
        end = start + psf_base_superres.size
        box_matrix_iter = box_matrix.flat
        box_matrix_iter[start : end] = psf_base_superres
    
    # Subtract off a small clip, set negative indices to 0, and renomalize.
    # This is a modification of the look-up-table beam using a few assumptions
    # to make it faster/feasible to run.
    # Modifications: done per group of baselines that fit within the current box, 
    # rather than individually. region_grow is not used to find a contiguous
    # edge around the beam to cut because it is too slow.
    if beam_clip_floor:
        psf_val_ref = np.sum(box_matrix, 1)
        psf_amp = np.abs(box_matrix)
        psf_mask_threshold_use = np.max(psf_amp) / psf['beam_mask_threshold']
        psf_amp -= psf_mask_threshold_use
        psf_phase = np.arctan2(box_matrix.imag, box_matrix.real)
        psf_amp = np.maximum(psf_amp, np.zeros_like(psf_amp))
        box_matrix = psf_amp * np.cos(psf_phase) + (0 + 1j) * psf_amp * np.sin(psf_phase)
        ref_temp = np.sum(box_matrix, -1)
        box_matrix[:vis_n, :] *=  np.reshape(psf_val_ref / ref_temp, (psf_val_ref.size, 1))
    
    if degrid_flag and beam_int is not None and beam2_int is not None and n_grp_use is not None:
        # Calculate the beam and beam^2 integral (degridding)
        psf_resolution = psf['resolution']
        beam_int_temp = np.sum(box_matrix, 0) / psf_resolution ** 2
        beam2_int_temp = np.sum(np.abs(box_matrix) ** 2, 0) / psf_resolution ** 2
        for ii in range(np.size(freq_i)):
            beam_int[freq_i[ii]] += beam_int_temp[ii]
            beam2_int[freq_i[ii]] += beam2_int_temp[ii]
            n_grp_use[freq_i[ii]] += 1
    
    return box_matrix

def visibility_count(obs: dict, params: dict, vis_weights: np.ndarray, pyfhd_config: dict, logger: RootLogger, 
                     fi_use: np.ndarray|None = None, bi_use: np.ndarray|None = None, 
                     mask_mirror_indices: bool = False, no_conjugate: bool = False, fill_model_visibilities: bool = False):
    """
    TODO: Docstring
    [summary]

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
    
    #Retrieve info from the data structures
    dimension = int(obs['dimension'])
    elements = int(obs['elements'])
    psf_dim = pyfhd_config['psf_dim']

    baselines_dict = baseline_grid_locations(
        obs, 
        params, 
        vis_weights, 
        pyfhd_config, 
        logger, 
        bi_use = bi_use, 
        fi_use = fi_use,
        mask_mirror_indices = mask_mirror_indices, 
        fill_model_visibilities = fill_model_visibilities
    )
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

def holo_mapfn_convert(map_fn, psf_dim, dimension, elements = None, norm = 1, threshold = 0):
    """
    TODO: Description

    Parameters
    ----------
    map_fn: ndarray
        TODO: Description
    psf_dim: int, float
        TODO: Description
    dimension: int
        TODO: Description
    elements: None, optional
        TODO: Description
    norm: int
        TODO: Description
    threshold: int, float, optional
        TODO: Description
    Returns
    -------
    
    """
    # Set up all the necessary arrays and numbers
    if elements is None:
        elements = dimension
    psf_dim2 = 2 * psf_dim
    psf_n = psf_dim ** 2
    sub_xv = meshgrid(psf_dim2, 1) - psf_dim
    sub_yv = meshgrid(psf_dim2, 2) - psf_dim
    # Generate an array of shape elements x dimension
    n_arr = np.zeros((elements, dimension))
    # Replace the values of n_arr with the number from each array in mapfn that exceeds the threshold
    for xi in range(dimension - 1):
        for yi in range(elements - 1):
            temp_arr = map_fn[xi, yi]
            n1 = np.size(np.where(abs(temp_arr) > threshold))
            n_arr[xi,yi] = n1
    # Get the ones to use
    i_use = np.where(n_arr)
    # Get the amount we're using
    i_use_size = np.size(i_use)
    # If we aren't using any then return 0
    if i_use_size == 0:
        return 0
    
    # Get the reverse indices
    _, _, ri = histogram(i_use, min = 0)
    # Create zeros of the same size as what we're using
    sa = np.zeros(i_use_size)
    ija = np.zeros(i_use_size)
    # Fill in the sa and ija arrays
    for index in range(i_use_size):
        i = i_use[index]
        xi = i % dimension
        yi = np.floor(i / dimension)
        map_fn_sub = map_fn[xi, yi]
        j_use = np.where(np.abs(map_fn_sub) > threshold)

        xii_arr = sub_xv[j_use] + xi
        yii_arr = sub_yv[j_use] + yi
        sa[index] = map_fn_sub[j_use]
        ija[index] = ri[ri[xii_arr + dimension * yii_arr]]
    # Return as record array
    mapfn = np.recarray(ija, sa, i_use, norm, 1, \
                        dtype=[(('ija', 'IJA'), 'int'), (('sa', 'SA'), 'int'), \
                               (('i_use', 'I_USE'), 'int'), (('norm', 'NORM'), 'float'),\
                               (('indexed', 'INDEXED'), '>i2')])
    return mapfn