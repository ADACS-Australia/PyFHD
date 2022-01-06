import numpy as np
from fhd_utils.FFT.deriv_coefficients import deriv_coefficients
from fhd_utils.histogram import histogram
from baseline_grid_locations import baseline_grid_locations
from fhd_utils.l_m_n import l_m_n
from interpolate_kernel import interpolate_kernel
from grid_beam_per_baseline import grid_beam_per_baseline
from fhd_utils.rebin import rebin
from fhd_utils.weight_invert import weight_invert

def visibility_degrid(image_uv, vis_weights, obs, psf, params, polarization = 0,
                      fill_model_visibilities = False, vis_input = None, spectral_model_uv_arr = None,
                      beam_per_baseline = False, uv_grid_phase_only = True, conserve_memory = False, memory_threshold = 1e8):
    """[summary]

    Parameters
    ----------
    image_uv : [type]
        [description]
    vis_weight : [type]
        [description]
    obs : [type]
        [description]
    psf : [type]
        [description]
    params : [type]
        [description]
    polarixation : int, optional
        [description], by default 0
    fill_model_visibilities : bool, optional
        [description], by default False
    vis_input : [type], optional
        [description], by default None
    spectral_model_uv_arr : [type], optional
        [description], by default None
    beam_per_baseline : bool, optional
        [description], by default False
    uv_grid_phase_only : bool, optional
        [description], by default True
    conserve_memory : bool, optional
        [description], by default False
    """

    complex = psf['complex_flag']
    n_spectral = obs['degrid_spectral_terms']
    interp_flag = psf['interpolate_kernel']
    if conserve_memory:
        # memory threshold is in bytes
        if memory_threshold < 1e6:
            memory_threshold = 1e8
    
    # For each unflagged baseline, get the minimum contributing pixel number for gridding 
    # and the 2D derivatives for bilinear interpolation
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, fill_model_visibilities = fill_model_visibilities, interp_flag = interp_flag)

    # Extract data from the data structures
    bin_n = baselines_dict['bin_n']
    bin_i = baselines_dict['bin_i']
    n_bin_use = baselines_dict['n_bin_use']
    ri = baselines_dict['ri']
    xmin = baselines_dict['xmin']
    ymin = baselines_dict['ymin']
    vis_inds_use = baselines_dict['vis_inds_use']
    x_offset = baselines_dict['x_offset']
    y_offset = baselines_dict['y_offset']
    if interp_flag:
        dx0dy0_arr = baselines_dict['dx0dy0']
        dx0dy1_arr = baselines_dict['dx0dy1']
        dx1dy0_arr = baselines_dict['dx1dy0']
        dx1dy1_arr = baselines_dict['dx1dy1']
    bi_use = baselines_dict['bi_use']
    dimension = obs['dimension']
    elements = obs['elements']
    kbinsize = obs['kpix']
    freq_bin_i = obs['baseline_info']['fbin_i']
    frequency_array = obs['baseline_info']['freq']
    freq_delta = (frequency_array - obs['freq_center']) / obs['freq_center']
    psf_dim = psf['dim']
    psf_resolution = psf['resolution']
    psf_dim3 = psf_dim ** 2
    nbaselines = obs['nbaselines']
    n_samples = obs['n_time']
    n_freq_use = frequency_array.size
    n_freq = obs['n_freq']
    psf_dim2 = 2 * psf_dim
    group_arr = np.squeeze(psf['id'][:, freq_bin_i, polarization])
    beam_arr = psf['beam_ptr']

    if beam_per_baseline:
        uu = params['uu']
        vv = params['vv']
        ww = params['ww']
        psf_image_dim = psf['image_info']['psf_image_dim']
        psf_intermediate_res = np.min(np.ceil(np.sqrt(psf_resolution) / 2) * 2, psf_resolution)
        image_bot = -(psf_dim / 2) * psf_intermediate_res + psf_image_dim / 2
        image_top = (psf_dim * psf_resolution - 1) - (psf_dim / 2) * psf_intermediate_res + psf_image_dim / 2

        l_mode, m_mode, n_tracked = l_m_n(obs, psf)

        # w-terms have not been tested, thus they've been turned off for now
        if uv_grid_phase_only:
            n_tracked = np.zeros_like(n_tracked)
        
        beam_int = beam2_int = n_grp_use = primary_beam_area = primary_beam_sq_area =  np.zeros(n_freq)

        x = y = (np.arange(dimension) - dimension / 2) * kbinsize
    
    conj_i = np.where(params['vv'] > 0)
    if conj_i.size > 0:
        if beam_per_baseline:
            uu[conj_i] = -uu[conj_i]
            vv[conj_i] = -vv[conj_i]
            ww[conj_i] = -ww[conj_i]
    
    # Create the correct size visibility array
    vis_dimension = nbaselines * n_samples
    visibility_array = np.zeros((vis_dimension, n_freq), dtype = complex)
    
    ind_ref = np.arange(max(bin_n))

    if complex:
        arr_type = complex
    else:
        arr_type = float
    
    if n_spectral:
        prefactor = [0] * n_spectral
        for s_i in range(n_spectral):
            prefactor[s_i] = deriv_coefficients(s_i + 1, divide_factorial = True)
        box_arr_ptr = np.zeros(n_spectral)

    for bi in range(n_bin_use):
        vis_n = bin_n[bin_i[bi]]
        inds = ri[ri[bin_i[bi]] : ri[bin_i[bi] + 1] - 1]

        # if constraining memory usage, then est number of loops needed
        if conserve_memory:
            required_bytes = 8 * vis_n * psf_dim3
            mem_iter = np.ceil(required_bytes / memory_threshold)
        else:
            mem_iter = 1
        # These variables will get used in the case of mem_iter > 1
        vis_n_full = vis_n
        inds_full = inds
        vis_n_per_iter = np.ceil(vis_n_full/mem_iter)
        # loop over chunks of visibilities to grid to conserve memory
        for mem_i in range(mem_iter):
            if mem_iter > 1:
                # calculate the indices of this visibility chunk if split into multiple chunks
                if vis_n_per_iter * (mem_i + 1) > vis_n_full:
                    max_ind = vis_n_full
                else:
                    max_ind = vis_n_per_iter * (mem_i + 1)
                inds = inds_full[vis_n_per_iter * mem_i : max_ind - 1]
                vis_n = max_ind - vis_n_per_iter * mem_i
            
            ind0 = inds[0]
            x_off = x_offset[inds]
            y_off = y_offset[inds]
            xmin_use = xmin[ind0]
            ymin_use = ymin[ind0]
            freq_i = inds % n_freq_use
            fbin = freq_bin_i[freq_i]
            baseline_inds = (inds / n_freq_use) % nbaselines

            box_matrix = np.zeros((vis_n, psf_dim3), dtype = arr_type)
            box_arr = np.flatten(image_uv[ymin_use : ymin_use + psf_dim - 1, xmin_use : xmin_use + psf_dim - 1])

            if interp_flag:
                dx0dy0 = dx0dy0_arr[inds]    
                dx0dy1 = dx0dy1_arr[inds]    
                dx1dy0 = dx1dy0_arr[inds]    
                dx1dy1 = dx1dy1_arr[inds]
                ind_remap_flag = False
                bt_index = inds / n_freq_use
                for ii in range(vis_n):
                    box_matrix[psf_dim3 * ii] = interpolate_kernel(beam_arr[baseline_inds[ii], fbin[ii], polarization], 
                                                                   x_off[ii], y_off[ii], dx0dy0[ii], dx1dy0[ii], dx0dy1[ii],
                                                                   dx1dy1[ii])
            else:
                group_id = group_arr[inds]
                group_max = np.max(group_id) + 1
                xyf_i = (x_off + y_off * psf_resolution + fbin * psf_resolution ** 2) * group_max + group_id
                xyf_si = np.sort(xyf_i)
                xyf_i = xyf_i[xyf_si]
                xyf_ui = np.unique(xyf_i)
                n_xyf_bin = xyf_ui.size

                # There might be a better selection criteria to determine which is more efficient 
                if vis_n > np.ceil(1.1 * n_xyf_bin) and not beam_per_baseline:
                    ind_remap_flag = True
                    inds = inds[xyf_si]
                    inds_use = xyf_si[xyf_ui]
                    freq_i = freq_i[inds_use]
                    x_off = x_off[inds_use]
                    y_off = y_off[inds_use]
                    fbin = fbin[inds_use]
                    baseline_inds = baseline_inds[inds]

                    if n_xyf_bin == 1:
                        ind_remap = np.arange(vis_n, dtype = int)
                    else:
                        hist_inds_u, _, ri_xyf = histogram(xyf_ui, bin_size = 1, min = 0)
                        ind_remap = ind_ref[ri_xyf[0 : hist_inds_u.size] - ri_xyf[0]]
                    
                    vis_n = n_xyf_bin
                else:
                    ind_remap_flag = False
                    bt_index = inds / n_freq_use
                
                if beam_per_baseline:
                    box_matrix = grid_beam_per_baseline(psf, uu, vv, ww, l_mode, m_mode, n_tracked,
                                                        frequency_array, x, y, xmin_use, ymin_use,
                                                        freq_i, bt_index, polarization, fbin, image_bot,
                                                        image_top, psf_dim3, box_matrix, vis_n, 
                                                        beam_int = beam_int, beam2_int = beam2_int, 
                                                        n_grp_use = n_grp_use, degrid_flag = True,
                                                        obs = obs, params = params, weights = vis_weights)
                else:
                    for ii in range(vis_n):
                        # more efficient array subscript notation 
                        box_matrix[psf_dim3 * ii] = beam_arr[baseline_inds[ii], fbin[ii], polarization][y_off[ii], x_off[ii]]
            
            if n_spectral:
                vis_box = np.dot(box_arr, np.transpose(box_matrix))
                freq_term_arr = rebin(np.transpose(freq_delta[freq_i]), (vis_n, psf_dim3), sample = True)
                for s_i in range(n_spectral):
                    # s_i loop is over terms of the Taylor expansion, starting from the lowest-order term
                    prefactor_use = prefactor[s_i]
                    box_matrix *= freq_term_arr
                    box_arr_ptr[s_i] = np.flatten(spectral_model_uv_arr[s_i][ymin_use : ymin_use + psf_dim - 1, xmin_use : xmin_use + psf_dim - 1])

                    for s_i_i in range(s_i):
                        # s_i_i loop is over powers of the model x alpha^n, n=s_i_i+1
                        box_arr = prefactor_use[s_i_i] * box_arr_ptr[s_i_i]
                        vis_box += np.dot(box_arr, np.transpose(box_matrix))
                del(box_arr_ptr)
            else:
                vis_box = np.dot(box_arr, np.transpose(box_matrix))
            if ind_remap_flag:
                vis_box = vis_box[ind_remap]
            visibility_array[inds] = vis_box
    
    if beam_per_baseline:
        # factor of kbinsize^2 is FFT units normalization
        beam2_int *= weight_invert(n_grp_use) / kbinsize ** 2
        beam_int *= weight_invert(n_grp_use) / kbinsize ** 2
        primary_beam_area = beam2_int
        primary_beam_area = beam_int
        # Returning obs in this case?
        obs['primary_beam_area'][polarization] = primary_beam_area
        obs['primary_beam_sq_area'][polarization] = primary_beam_sq_area

    del(x_offset, y_offset, xmin, ymin, bin_n)
    if conj_i.size > 0:
        visibility_array[conj_i, :] = np.conj(visibility_array[conj_i, :])

    if vis_input.size > 0:
        return vis_input + visibility_array
    else:
        return visibility_array 

