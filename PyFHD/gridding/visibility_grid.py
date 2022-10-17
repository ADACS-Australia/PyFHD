import numpy as np
import warnings
from PyFHD.gridding.gridding_utils import interpolate_kernel, baseline_grid_locations, grid_beam_per_baseline, conjugate_mirror, holo_mapfn_convert
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert, rebin, l_m_n, idl_argunique

def visibility_grid(visibility, vis_weights, obs, status_str, psf, params,
                    file_path_fhd= "/.", weights_flag = False, variance_flag = False, polarization = 0,
                    map_flag = False, uniform_flag = False, grid_uniform = False, fi_use = None, bi_use = None, 
                    no_conjugate = False, mask_mirror_indices = False, model = None, grid_spectral = False, 
                    beam_per_baseline = False, uv_grid_phase_only = True) :
    """
    [summary]
    TODO: docstring

    Parameters
    ----------
    visibility : [type]
        [description]
    vis_weights : [type]
        [description]
    obs : [type]
        [description]
    status_str : [type]
        [description]
    psf : [type]
        [description]
    params : [type]
        [description]
    file_path_fhd : str, optional
        [description], by default "/."
    weights_flag : bool, optional
        [description], by default False
    variance_flag : bool, optional
        [description], by default False
    polarization : int, optional
        [description], by default 0
    map_flag : bool, optional
        [description], by default False
    uniform_flag : bool, optional
        [description], by default False
    fi_use : [type], optional
        [description], by default None
    bi_use : [type], optional
        [description], by default None
    no_conjugate : bool, optional
        [description], by default False
    return_mapfn : bool, optional
        [description], by default False
    mask_mirror_indices : bool, optional
        [description], by default False
    no_save : bool, optional
        [description], by default False
    model : [type], optional
        [description], by default None
    preserve_visibilities : bool, optional
        [description], by default False
    error : bool, optional
        [description], by default False
    grid_spectral : bool, optional
        [description], by default False
    spectral_model_uv : int, optional
        [description], by default 0
    beam_per_baseline : bool, optional
        [description], by default False
    uv_grid_phase_only : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """

    # Get information from the data structures
    dimension = int(obs['dimension'][0])
    elements = int(obs['elements'][0])
    interp_flag = psf['interpolate_kernel']
    alpha = obs['alpha']
    freq_bin_i = obs['baseline_info'][0]['fbin_i'][0]
    n_freq = obs['n_freq']
    if fi_use is None:
        fi_use = np.nonzero(obs['baseline_info'][0]['freq_use'][0])[0]
    n_f_use = fi_use.size
    freq_bin_i = freq_bin_i[fi_use]
    n_vis_arr = obs['nf_vis'][0].copy()
    # Getting a read-only error for some reason, setting as writeable
    n_vis_arr.flags.writeable = True

    # If both beam and interp_flag leave a warning, prioritise beam_per_baseline
    if beam_per_baseline and interp_flag:
        warnings.warn("Cannot have beam per baseline and interpolation at the same time, turning off interpolation")
        interp_flag = False

    # For each unflagged baseline, get the minimum contributing pixel number for gridding 
    # and the 2D derivatives for bilinear interpolation
    baselines_dict = baseline_grid_locations(
        obs, 
        psf, 
        params, 
        vis_weights,
        fi_use = fi_use, 
        bi_use = bi_use,
        mask_mirror_indices = mask_mirror_indices, 
        interp_flag = interp_flag
    )
    # Extract data from the returned dictionary 
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
        dx0dy0_arr = baselines_dict['dx0dy0_arr']
        dx0dy1_arr = baselines_dict['dx0dy1_arr']
        dx1dy0_arr = baselines_dict['dx1dy0_arr']
        dx1dy1_arr = baselines_dict['dx1dy1_arr']
    if bi_use is None:
        bi_use = baselines_dict['bi_use']

    # Instead of checking the visibilitity pointer we just take the vis_inds_use from visibility
    vis_arr_use = visibility.flat[vis_inds_use]
    # Model_flag has been removed in favor of just the model taking advantage that the model default is None
    # If it has been specified at all with anything other than None or False, then it should be a numpy array
    # if it isn't exit
    if model is not None:
        if isinstance(model, np.ndarray):
            model_use = model.flat[vis_inds_use]
            model_return = np.zeros((elements, dimension), dtype = np.complex128)
        else:
            raise ValueError('Your model must be a numpy array when used as an argument')

    # Now with the information we need, retrieve more data from the structures
    frequency_array = obs['baseline_info'][0]['freq'][0]
    frequency_array = frequency_array[fi_use]
    complex_flag = psf['complex_flag']
    psf_dim = psf['dim'][0]
    psf_resolution = psf['resolution'][0]
    nbaselines = obs['nbaselines'][0]
    n_samples = obs['n_time'][0]
    # New group_arr code that is consistent with the FHD version
    group_arr = np.squeeze(psf['id'][0][:, freq_bin_i, polarization])
    # REBIN in IDL when expanding dimensions repeats 2D arrays after expanding
    group_arr = np.expand_dims(rebin(group_arr, (nbaselines, n_f_use)), axis = 0)
    group_arr = np.repeat(group_arr, n_samples, axis = 0) 
    group_arr = np.reshape(group_arr, (n_samples*nbaselines ,n_f_use))
    beam_arr = psf['beam_ptr'][0]
    n_freq_use = frequency_array.size
    psf_dim2 = 2 * psf_dim
    psf_dim3 = psf_dim ** 2
    bi_use_reduced = bi_use % nbaselines
    
    # Flags have been defined in the function definition
    # Instead of reading the flags and then setting them.

    if beam_per_baseline:
        # Initialization for gridding operation via a low-res beam kernel, calculated per
        # baseline using offsets from image-space delays
        uu = params['uu'][0][bi_use]
        vv = params['vv'][0][bi_use]
        ww = params['ww'][0][bi_use]
        x = (np.arange(dimension) - dimension / 2) * obs['kpix'][0]
        y = x.copy()
        psf_intermediate_res = np.min([np.ceil(np.sqrt(psf_resolution) / 2) * 2, psf_resolution])
        psf_image_dim = psf['image_info'][0]['psf_image_dim'][0]
        image_bot = int(-(psf_dim / 2) * psf_intermediate_res + psf_image_dim / 2)
        image_top = int((psf_dim * psf_resolution - 1) - (psf_dim / 2) * psf_intermediate_res + psf_image_dim / 2)
        l_mode, m_mode, n_tracked = l_m_n(obs, psf)
        if uv_grid_phase_only:
            n_tracked = np.zeros_like(n_tracked)
    
    # Initialize uv-arrays 
    image_uv = np.zeros((elements, dimension), dtype = np.complex128)
    weights = np.zeros((elements, dimension), dtype = np.complex128)
    variance = np.zeros((elements, dimension))
    uniform_filter = np.zeros((elements, dimension))

    # If the uniform gridding has been activated we need to activate the uniform filter and switch off mapping if it has been activated
    if grid_uniform:
        if map_flag:
            map_flag = False
            warnings.warn("You cannot turn on uniform gridding and the mapping functions at the same time, map functions have been disabled")
        uniform_flag = True
    
    conj_i = np.where(params['vv'][0].flat[bi_use] > 0)[0]
    if conj_i.size > 0:
        if beam_per_baseline:
            uu[conj_i] = -uu[conj_i]
            vv[conj_i] = -vv[conj_i]
            ww[conj_i] = -ww[conj_i]
        vis_arr_use[conj_i, :] = np.conj(vis_arr_use[conj_i, :])
        if model is not None:
            model_use[conj_i, :] = np.conj(model_use[conj_i, :])
    
    # Return if all baselines have been flagged
    if n_bin_use == 0:
        print("All data has been flagged")
        return np.zeros([elements, dimension], dtype = np.complex128)
    
    n_vis = np.sum(bin_n)
    for fi in range(n_f_use):
        n_vis_arr[fi_use[fi]] = np.sum(xmin[:, fi] > 0)
    obs['nf_vis'][0] = n_vis_arr

    index_arr = np.reshape(np.arange(elements * dimension), (elements, dimension))
    if complex_flag:
        init_arr = np.zeros([psf_dim2, psf_dim2], dtype = np.complex128)
    else:
        init_arr = np.zeros([psf_dim2, psf_dim2])
    arr_type = init_arr.dtype
    if grid_spectral:
        # Spectral B and Spectral D shouldn't reference each other just in case
        spectral_A = np.zeros([elements, dimension], dtype = np.complex128)
        spectral_B = np.zeros([elements, dimension])
        spectral_D = np.zeros([elements, dimension])
        if model is not None:
            spectral_model_A = np.zeros([elements, dimension], dtype = np.complex128)
    
    # In the IDL visibility_grid, map_fn is set up as a 2D array of null pointers
    # In the case of Ptr_Valid, null pointers return False (0)
    # Meaning that every value between [ymin1:ymin1+psf_dim-1, xmin1:xmin1+psf_dim-1]
    # is set as init_arr, which is size psf_dim2 x psf_dim2
    if map_flag:
        map_fn = np.zeros((elements, dimension, psf_dim2, psf_dim2))
        # TODO: IDL visibility_grid.pro code from 164-182 goes here

    for bi in range(n_bin_use):
        # Cycle through sets of visibilities which contribute to the same data/model uv-plane pixels, and perform
        # the gridding operation per set using each visibilities' hyperresolved kernel 

        # Select the indices of the visibilities which contribute to the same data/model uv-plane pixels
        inds = ri[ri[bin_i[bi]] : ri[bin_i[bi] + 1]]
        ind0 = inds[0]

        # Select the pixel offsets of the hyperresolution uv-kernel of the selected visibilities 
        x_off = x_offset.flat[inds]
        y_off = y_offset.flat[inds]

        # Since all selected visibilities have the same minimum x,y pixel they contribute to,
        # reduce the array
        xmin_use = xmin.flat[ind0]
        ymin_use = ymin.flat[ind0]

        # Find the frequency group per index
        freq_i = inds % n_freq_use
        fbin = freq_bin_i[freq_i]

        # Calculate the number of selected visibilities and their baseline index
        vis_n = bin_n[bin_i[bi]]
        baseline_inds = bi_use_reduced[((inds / n_f_use) % nbaselines).astype(int)]

        if interp_flag:
            # Calculate the interpolated kernel on the uv-grid given the derivatives to baseline locations
            # and the hyperresolved pre-calculated beam kernel

            # Select the 2D derivatives to baseline locations
            dx1dy1 = dx1dy1_arr.flat[inds]
            dx1dy0 = dx1dy0_arr.flat[inds]
            dx0dy1 = dx0dy1_arr.flat[inds]
            dx0dy0 = dx0dy0_arr.flat[inds]

            # Select the model/data visibility values of the set, each with a weight of 1
            rep_flag = False
            if model is not None:
                model_box = model_use.flat[inds]
            vis_box = vis_arr_use.flat[inds]
            psf_weight = np.ones(vis_n)

            box_matrix = np.zeros((vis_n, psf_dim3), dtype = arr_type)
            for ii in range(vis_n):
                # For each visibility, calculate the kernel values on the static uv-grid given the
                # hyperresolved kernel and an interpolation involving the derivatives
                box_matrix[ii] = interpolate_kernel(beam_arr[baseline_inds[ii], fbin[ii], polarization],
                                                    x_off[ii], y_off[ii], dx0dy0[ii], dx1dy0[ii], dx0dy1[ii], dx1dy1[ii])
        else:
            # Calculate the beam kernel at each baseline location given the hyperresolved pre-calculated
            # beam kernel

            # Calculate a unique index for each kernel location and kernel type in order to reduce 
            # operations if there are repeats
            group_id = group_arr.flat[inds]
            group_max = np.max(group_id) + 1
            xyf_i = (x_off + y_off * psf_resolution + fbin * psf_resolution ** 2) * group_max + group_id

            # Calculate the unique number of kernel locations/types 
            xyf_si = xyf_i.argsort(kind='stable')
            xyf_i = np.sort(xyf_i)
            xyf_ui = idl_argunique(xyf_i)
            n_xyf_bin = xyf_ui.size

            # There might be a better selection criteria to determine which is most efficient
            if vis_n > 1.1 * n_xyf_bin and not beam_per_baseline:
                # If there are any baselines which use the same beam kernel and the same discretized location
                # given the hyperresolution, then reduce the number of gridding operations to only 
                # non-repeated baselines
                rep_flag = True
                inds = inds[xyf_si]
                inds_use = xyf_si[xyf_ui]
                freq_i = freq_i[inds_use]

                x_off = x_off[inds_use]
                y_off = y_off[inds_use]
                fbin = fbin[inds_use]
                baseline_inds = baseline_inds[inds_use]
                if n_xyf_bin > 1:
                    xyf_ui0 = np.insert(xyf_ui[0: n_xyf_bin - 1] + 1, 0, 0)
                else:
                    xyf_ui0 = np.array([0])
                psf_weight = xyf_ui - xyf_ui0 + 1

                vis_box1 = vis_arr_use.flat[inds]
                vis_box = vis_box1.flat[xyf_ui]
                if model is not None:
                    model_box1 = model_use.flat[inds]
                    model_box = model_box1.flat[xyf_ui]
                
                # For the baselines which map to the same pixels and use the same beam,
                # add the underlying data/model pixels such that the gridding operation
                # only needs to be performed once for the set
                if xyf_ui0.size == 1:
                    repeat_i = np.array([0])
                else:
                    repeat_i = np.where(psf_weight > 1)[0]
                xyf_ui = xyf_ui[repeat_i]
                xyf_ui0 = xyf_ui0[repeat_i]
                for rep_ii in range(repeat_i.size):
                    vis_box[repeat_i[rep_ii]] = np.sum(vis_box1[xyf_ui0[rep_ii]:xyf_ui[rep_ii] + 1])
                    if model is not None:
                        model_box[repeat_i[rep_ii]] = np.sum(model_box1[xyf_ui0[rep_ii]:xyf_ui[rep_ii] + 1])
                vis_n = n_xyf_bin
            else:
                # If there are not enough baselines which use the same beam kernel and discretized
                # location to warrent reduction, then perform the gridding operation per baseline
                rep_flag = False
                if model is not None:
                    model_box = model_use.flat[inds]
                vis_box = vis_arr_use.flat[inds]
                psf_weight = np.ones(vis_n)
                # IDL had integer / integer i.e. 2015 / 336 == 5, used flooring divider instead in python
                # ALso do take note that each were very close always within 0.01 of their next number.
                # e.g. 2015 / 336 = 5.99, should ceiling be be used instead?
                bt_index = inds // n_freq_use
            
            box_matrix = np.zeros((vis_n, psf_dim3), dtype = arr_type)
            if beam_per_baseline:
                # Make the beams on the fly with corrective phases given the baseline location for each visibility
                # to the static uv-grid
                box_matrix = grid_beam_per_baseline(psf, uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array,
                                                    x, y, xmin_use, ymin_use, freq_i, bt_index, polarization,
                                                    fbin, image_bot, image_top, psf_dim3, box_matrix, vis_n,
                                                    obs = obs, params = params, weights = vis_weights, 
                                                    fi_use = fi_use, bi_use = bi_use, mask_mirror_indices = mask_mirror_indices)
            else:
                for ii in range(vis_n):
                    # For each visibility, calculate the kernel values on the static uv-grid given the
                    # hyperresolved kernel
                    box_matrix[ii, :] = beam_arr[baseline_inds[ii], fbin[ii], polarization][y_off[ii], x_off[ii]]
        
        #  Calculate the conjugate transpose (dagger) of the uv-pixels that the current beam kernel contributes to
        #  Calculate the conjugate transpose (dagger) of the uv-pixels that the current beam kernel contributes to
        if complex_flag:
            box_matrix_dag = np.conj(box_matrix)
        else:
            box_matrix_dag = box_matrix.real
        if map_flag and rep_flag:
            box_matrix *= rebin(np.transpose(psf_weight), (vis_n, psf_dim3))
        
        if grid_spectral:
            term_A_box = np.dot(np.transpose(box_matrix_dag), np.transpose((freq_i * vis_box) / n_vis))
            term_B_box = np.dot(np.transpose(box_matrix_dag), np.transpose(freq_i / n_vis))
            term_D_box = np.dot(np.transpose(box_matrix_dag), np.transpose(freq_i ** 2 / n_vis))
            spectral_A[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += term_A_box
            spectral_B[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += term_B_box.real
            spectral_D[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += term_D_box.real
            #del(term_A_box, term_B_box, term_D_box)
            if model is not None:
                term_Am_box = np.dot(np.transpose(box_matrix_dag), np.transpose((freq_i * model_box) / n_vis))
                spectral_model_A[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim] += term_Am_box
        
        if model is not None:
            # If model visibilities are being gridded, calculate the product of the model vis and the beam kernel
            # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane 

            # Ensure model_box is flat, sometimes odd shapes can come in from metadata
            model_box = model_box.flatten()
            box_arr = np.dot(np.transpose(box_matrix_dag), np.transpose(model_box / n_vis))
            model_return[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += box_arr
        
        # Calculate the product of the data vis and the beam kernel
        # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane

        # Ensure vis_box is flat, sometimes odd shapes can come in from metadata
        vis_box = vis_box.flatten()
        box_arr = np.dot(np.transpose(box_matrix_dag), vis_box / n_vis)
        image_uv[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += box_arr
        del(box_arr)

        if weights_flag:
            # If weight visibilities are being gridded, calculate the product the weight (1 per vis) and the beam kernel
            # for all vis which contribute to the same static uv-pixels, and add to the static uv-plane
            wts_box = np.dot(np.transpose(box_matrix_dag), np.transpose(psf_weight / n_vis))
            weights[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += wts_box
        
        if variance_flag:
            # If variance visibilities are being gridded, calculate the product the weight (1 per vis) and the square
            # of the beam kernel for all vis which contribute to the same static uv-pixels, and add to the static uv-plane
            var_box = np.dot(np.transpose(np.abs(box_matrix_dag) ** 2), np.transpose(psf_weight / n_vis))
            variance[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim].flat += var_box
        
        if uniform_flag:
            uniform_filter[ymin_use : ymin_use + psf_dim, xmin_use : xmin_use + psf_dim] += bin_n[bin_i[bi]]
        
        if map_flag:
            # If the mapping function is being calculated, then calculate the beam mapping for the current
            # set of uv-pixels and add to the full mapping function
            box_arr_map = np.dot(np.transpose(box_matrix_dag), box_matrix)
            for i in range(psf_dim):
                for j in range(psf_dim):
                    ij = i + j * psf_dim
                    # TODO: access map function
        
    # Free Up Memory
    del(vis_arr_use, xmin, ymin, ri, inds, x_offset, y_offset, bin_i, bin_n)

    if model is not None:
        del(model_use)

    if map_flag:
        map_fn = holo_mapfn_convert(map_fn, psf_dim, dimension, norm = n_vis)
    
    # Option to use spectral index information to scale the uv-plane 
    if grid_spectral:
        spectral_uv = (spectral_A - n_vis * spectral_B * image_uv) * weight_invert(spectral_D - spectral_B ** 2)
        if model is not None:
            spectral_model_uv = (spectral_model_A - n_vis * spectral_B * model_return) * weight_invert(spectral_D - spectral_B ** 2)
        if not no_conjugate:
            spectral_uv = (spectral_uv + conjugate_mirror(spectral_uv)) / 2
            if model is not None:
                spectral_model_uv = (spectral_model_uv + conjugate_mirror(spectral_model_uv)) / 2
    
    # Option to apply a uniform weighted filter to all uv-planes
    if grid_uniform:
        filter_use = weight_invert(uniform_filter, threshold = 1)
        wts_i = np.nonzero(filter_use)
        if wts_i[0].size > 0:
            filter_use /= np.mean(filter_use[wts_i])
        else:
            filter_use /= np.mean(filter_use)
        image_uv *= weight_invert(filter_use)
        if weights_flag:
            weights *= weight_invert(filter_use)
        if variance_flag:
            variance *= weight_invert(filter_use)
        if model is not None:
            model_return *= weight_invert(filter_use)
    
    if not no_conjugate:
        # The uv-plane is its own conjugate mirror about the x-axis, so fill in the rest of the uv-plane
        # using simple maths instead of extra gridding
        image_uv = (image_uv + conjugate_mirror(image_uv)) / 2
        if weights_flag:
            weights = (weights + conjugate_mirror(weights)) / 2
        if variance_flag:
            variance = (variance + conjugate_mirror(variance)) / 4
        if model is not None:
            model_return = (model_return + conjugate_mirror(model_return)) / 2
        if uniform_flag:
            uniform_filter = (uniform_filter + conjugate_mirror(uniform_filter)) / 2

    # Arrange returns into a dictionary
    gridding_dict = {
        'image_uv' : image_uv,
        'weights' : weights,
        'variance' : variance,
        'uniform_filter' : uniform_filter,
        'obs' : obs,
    }

    if grid_spectral:
        gridding_dict['spectral_uv'] = spectral_uv

    if model is not None:
        gridding_dict['model_return'] = model_return
        if grid_spectral:
            gridding_dict['spectral_model_uv'] = spectral_model_uv
    
    return gridding_dict
