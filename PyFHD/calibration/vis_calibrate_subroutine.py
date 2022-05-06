import numpy as np
import warnings
from PyFHD.calibration.calibration_utils import calculate_adaptive_gain
from PyFHD.pyfhd_tools.pyfhd_utils import weight_invert, histogram

def vis_calibrate_subroutine(vis_ptr, vis_model_ptr, vis_weight_ptr, obs, cal, 
                             calibration_weights = False,  no_ref_tile = False):
    """
    TODO: Docstring
    [summary]

    Parameters
    ----------
    vis_ptr : [type]
        [description]
    vis_model_ptr : [type]
        [description]
    vis_weight_ptr : [type]
        [description]
    obs : [type]
        [description]
    cal : [type]
        [description]
    preserve_visibilities : bool, optional
        [description], by default False
    calibration_weights : bool, optional
        [description], by default False
    no_ref_tile : bool, optional
        [description], by default True
    """
    # Retrieve values from data structures
    reference_tile = cal['ref_antenna'][0]
    min_baseline = obs['min_baseline'][0]
    max_baseline = obs['max_baseline'][0]
    dimension = obs['dimension'][0]
    elements = obs['elements'][0]
    min_cal_baseline = cal['min_cal_baseline'][0]
    max_cal_baseline = cal['max_cal_baseline'][0]
    # minimum number of calibration equations needed to solve for the gain of one baseline
    min_cal_solutions = cal['min_solns'][0]
    # average the visibilities across time steps before solving for the gains
    time_average = cal['time_avg'][0]
    # maximum iterations to perform for the linear least-squares solver
    max_cal_iter = cal['max_iter'][0]
    # Leave a warning if its less than 5 iterations, or an Error if its less than 1
    if max_cal_iter < 5:
        warnings.warn("At Least 5 calibrations iterations is recommended.\nYou're currently using {} iterations".format(int(max_cal_iter)))
    elif max_cal_iter < 1:
        raise ValueError("max_cal_iter should be 1 or more. A max_cal_iter of 5 or more is recommended")
    conv_thresh = cal['conv_thresh'][0]
    use_adaptive_gain = cal['adaptive_gain'][0]
    base_gain = cal['base_gain'][0]
    # halt if the strict convergence is worse than most of the last x iterations
    divergence_history = 3
    # halt if the convergence gets significantly worse by a factor of x in one iteration
    divergence_factor = 1.5
    n_pol = cal['n_pol'][0]
    n_freq = cal['n_freq'][0]
    n_tile = cal['n_tile'][0]
    n_time = cal['n_time'][0]
    # weights WILL be over-written! (Only for NAN gain solutions)
    vis_weight_ptr_use = vis_weight_ptr
    # tile_A & tile_B contribution indexed from 0
    tile_A_i = cal['tile_a'][0] - 1
    tile_B_i = cal['tile_b'][0] - 1
    freq_arr = cal['freq'][0]
    n_baselines = obs['nbaselines'][0]
    if 'phase_iter' in cal.dtype.names:
        phase_fit_iter = cal['phase_iter'][0]
    else:
        phase_fit_iter = np.min([np.floor(max_cal_iter / 4), 4])
    kbinsize = obs['kpix'][0]
    cal_return = cal.copy()

    for pol_i in range(n_pol):
        convergence = np.zeros((n_tile, n_freq))
        conv_iter_arr = np.zeros((n_tile, n_freq))
        gain_arr = cal['gain'][0][pol_i]

        # Average the visibilities over the time steps before solving for the gains solutions
        # This is not recommended, as longer baselines will be downweighted artifically.
        if time_average:
            # The visibilities have dimension nfreq x (n_baselines x n_time),
            # which can be reformed to nfreq x n_baselines x n_time
            tile_A_i = tile_A_i[0 : n_baselines]
            tile_B_i = tile_B_i[0 : n_baselines]
            # So IDL does reforms as REFORM(x, cols, rows, num_of_col_row_arrays)
            # Python is row-major, so we need to flip that shape that is used in REFORM
            shape = np.flip(np.array([n_freq, n_baselines, n_time]))
            vis_weight_use = np.maximum(np.reshape(vis_weight_ptr_use[pol_i], shape), 0)
            vis_weight_use = np.minimum(vis_weight_use, 1)
            vis_model = np.reshape(vis_model_ptr[pol_i], shape)
            vis_model = np.sum(vis_model * vis_weight_use, axis = 0)
            vis_measured = np.reshape(vis_ptr[pol_i], shape)
            vis_avg = np.sum(vis_measured * vis_weight_use, axis = 0)
            weight = np.sum(vis_weight_use, axis = 0)

            kx_arr = cal['uu'][0][0 : n_baselines] / kbinsize
            ky_arr = cal['vv'][0][0 : n_baselines] / kbinsize
        else:
            # In the case of not using a time_average do the following setup instead for weight and vis_avg
            vis_weight_use = np.min([np.max([0, vis_weight_ptr_use[pol_i]]), 1])
            vis_model = vis_model * vis_weight_use
            vis_avg = vis_ptr[pol_i] * vis_weight_use
            weight = vis_weight_use

            kx_arr = cal['uu'][0] / kbinsize
            ky_arr = cal['vv'][0] / kbinsize
        # Now use the common code from the two possibilities in vis_calibrate_subroutine.pro 
        kr_arr = np.sqrt(kx_arr ** 2 + ky_arr ** 2)
        # When IDL does a matrix multiply on two 1D vectors it does the outer product.
        dist_arr = np.outer(kr_arr, freq_arr) * kbinsize
        xcen = np.outer(abs(kx_arr), freq_arr)
        ycen = np.outer(abs(ky_arr), freq_arr)
        if calibration_weights:
            flag_dist_cut = np.where((dist_arr.flat < min_baseline) | (xcen.flat > (elements / 2)) | (ycen.flat > (dimension / 2)))[0]
            if min_cal_baseline > min_baseline:
                taper_min = np.max((np.sqrt(2) * min_cal_baseline - dist_arr) / min_cal_baseline, 0)
            else:
                taper_min = 0
            if max_cal_baseline < max_baseline:
                taper_max = np.max((dist_arr - max_cal_baseline) / min_cal_baseline, 0)
            else:
                taper_max = 0
            baseline_weights = np.max(1 - (taper_min + taper_max) ** 2, 0)
        else:
            flag_dist_cut = np.where((dist_arr.flat < min_cal_baseline) | (dist_arr.flat > max_cal_baseline) | (xcen.flat > elements / 2) | (ycen.flat > dimension / 2))[0]
        # Remove kx_arr, ky_arr and dist_arr from the namespace, allow garbage collector to do its work
        del(kx_arr,ky_arr,dist_arr)

        if np.size(flag_dist_cut) > 0:
            weight.flat[flag_dist_cut] = 0
        vis_avg *= weight_invert(weight)
        vis_model *= weight_invert(weight)

        tile_use_flag = obs['baseline_info'][0]['tile_use'][0]
        freq_use_flag = obs['baseline_info'][0]['freq_use'][0]

        freq_weight = np.sum(weight, axis = 0)
        baseline_weight = np.sum(weight, axis = 1)
        freq_use = np.where((freq_weight > 0) & (freq_use_flag > 0))[0]
        baseline_use = np.nonzero(baseline_weight)
        hist_tile_A, _, riA = histogram(tile_A_i[baseline_use], min = 0, max = n_tile - 1)
        hist_tile_B, _, riB = histogram(tile_B_i[baseline_use], min = 0, max = n_tile - 1)
        tile_use = np.where(((hist_tile_A + hist_tile_B) > 0) & (tile_use_flag > 0))[0]

        tile_A_i_use = np.zeros(np.size(baseline_use))
        tile_B_i_use = np.zeros(np.size(baseline_use))
        for tile_i in range(np.size(tile_use)):
            if hist_tile_A[tile_use[tile_i]] > 0:
                # Calculate tile contributions for each non-flagged baseline
                tile_A_i_use[riA[riA[tile_use[tile_i]] : riA[tile_use[tile_i] + 1]]] = tile_i
            if hist_tile_B[tile_use[tile_i]] > 0:
                # Calculate tile contributions for each non-flagged baseline
                tile_B_i_use[riB[riB[tile_use[tile_i]] : riB[tile_use[tile_i] + 1]]] = tile_i

        ref_tile_use = np.where(reference_tile == tile_use)
        if ref_tile_use[0].size == 0:
            ref_tile_use = 0
            # Are we returning cal?
            cal['ref_antenna'] = tile_use[ref_tile_use]
            cal['ref_antenna_name'] = obs['baseline_info']['tile_names'][cal['ref_antenna']]
        
        # Replace all NaNs with 0's
        vis_model[np.isnan(vis_model)] = 0

        conv_test = np.zeros((max_cal_iter, freq_use.size))
        n_converged = 0
        for fii in range(freq_use.size):
            fi = freq_use[fii]
            gain_curr = np.squeeze(gain_arr[tile_use, fi])
            # Set up data and model arrays of the original and conjugated versions. This
            # provides twice as many equations into the linear least-squares solver.
            vis_data2 = np.squeeze(vis_avg[baseline_use, fi])
            vis_data2 = np.array([vis_data2, np.conj(vis_data2)])
            vis_model2 = np.squeeze(vis_model[baseline_use, fi])
            vis_model2 = np.array([vis_model2, np.conj(vis_model2)])
            weight2 = np.squeeze(weight[baseline_use, fi])
            weight2 = np.array([weight2, weight2])
            if calibration_weights:
                baseline_wts2 = np.squeeze(baseline_weights[baseline_use, fi])
                baseline_wts2 = [baseline_wts2, baseline_wts2]
            
            b_i_use = np.where(weight2 > 0)
            weight2 = weight2[b_i_use]
            vis_data2 = vis_data2[b_i_use]
            vis_model2 = vis_model2[b_i_use]

            A_ind = np.array([tile_A_i_use, tile_B_i_use], dtype = np.int64)
            A_ind = A_ind[b_i_use]
            B_ind = np.array([tile_B_i_use, tile_A_i_use], dtype = np.int64)
            B_ind = B_ind[b_i_use]

            A_ind_arr = []
            n_arr = np.zeros(tile_use.size)
            for tile_i in range(tile_use.size):
                inds = np.where(A_ind == tile_i)[0]
                if inds.size > 1:
                    A_ind_arr.append(np.reshape(inds, (inds.size, 1)))
                else:
                    A_ind_arr.append(-1)
                # NEED SOMETHING MORE IN CASE INDIVIDUAL TILES ARE FLAGGED FOR ONLY A FEW FREQUENCIES!!
                n_arr[tile_i] = inds.size
            A_ind_arr = np.array(A_ind_arr, dtype=object)
            # For tiles which don't satisfy the minimum number of solutions, pre-emptively set them to 0
            # in order to prevent certain failure in meeting strict convergence threshold
            inds_min_cal = np.where(n_arr < min_cal_solutions)[0]
            if inds_min_cal.size > 0:
                gain_curr[inds_min_cal] = 0
            gain_new = np.zeros(tile_use.size, dtype = np.complex128)
            convergence_list = np.zeros(max_cal_iter)
            conv_gain_list = np.zeros(max_cal_iter)
            convergence_loose = 0
            for i in range(max_cal_iter):
                convergence_loose_prev = convergence_loose
                divergence_flag = 0
                vis_use = vis_data2

                vis_model_matrix = vis_model2 * np.conj(gain_curr[B_ind])
                for tile_i in range(tile_use.size):
                    if n_arr[tile_i] >= min_cal_solutions:
                        if calibration_weights:
                            xmat = vis_model_matrix[A_ind_arr[tile_i]]
                            xmat_dag = np.conj(xmat) * baseline_wts2
                            gain_new[tile_i] = 1 / (np.dot(np.transpose(xmat), xmat_dag) * np.dot(np.transpose(vis_use[A_ind_arr[tile_i]]), xmat_dag))
                        else:
                            gain_new[tile_i] = np.linalg.lstsq(vis_model_matrix[A_ind_arr[tile_i]], vis_use[A_ind_arr[tile_i]], rcond=None)[0][0][0]
                
                gain_old = gain_curr
                if np.sum(np.abs(gain_new)) == 0:
                    gain_curr = gain_new
                    # Break the loop
                    break
                if phase_fit_iter - i > 0:
                    # fit only phase at first
                    gain_new *= np.abs(gain_old) * weight_invert(np.abs(gain_new))
                if use_adaptive_gain:
                    conv_gain = calculate_adaptive_gain(conv_gain_list, convergence_list, i, base_gain, final_convergence_estimate = 0)
                else:
                    conv_gain = base_gain
                gain_curr = (gain_new * conv_gain + gain_old * base_gain) / (base_gain + conv_gain)
                dgain = np.abs(gain_curr) * weight_invert(np.abs(gain_old))
                diverge_i = np.where(dgain < np.abs(gain_old) / 2)[0]
                if diverge_i.size > 0:
                    gain_curr[diverge_i] = (gain_new[diverge_i] + gain_old[diverge_i] * 2) / 3
                if np.size(np.where(np.isnan(gain_curr))) > 0:
                    gain_curr[np.where(np.isnan(gain_curr))] = gain_old[np.where(np.isnan(gain_curr))]
                if not no_ref_tile:
                    gain_curr *= np.conj(gain_curr[ref_tile_use]) / np.abs(gain_curr[ref_tile_use])
                convergence_strict = np.max(np.abs(gain_curr - gain_old) * weight_invert(np.abs(gain_old)))
                convergence_loose = np.mean(np.abs(gain_curr - gain_old) * weight_invert(np.abs(gain_old)))
                convergence_list[i] = convergence_strict
                conv_test[i, fii] = convergence_strict
                if i > phase_fit_iter + divergence_history:
                    if convergence_strict < conv_thresh:
                        # Stop if the solution has converged to the specified threshold
                        n_converged += 1
                        break
                    if convergence_loose >= convergence_loose_prev:
                        # Stop if the solutions are no longer converging
                        if (convergence_loose <= conv_thresh) or (convergence_loose_prev <= conv_thresh):
                            n_converged += 1
                            if convergence_loose <= conv_thresh:
                                # If the previous solution met the threshold, but the current one did not, then
                                # back up one iteration and use the previous solution
                                gain_curr = gain_old
                                convergence[tile_use, fi] = conv_test[i - 1, fii]
                                conv_iter_arr[tile_use, fi] = i - 1
                            break
                    else:
                        # Halt if the strict convergence is worse than most of the recent iterations
                        divergence_test_1 = convergence_strict >= np.median(conv_test[i - divergence_history - 1 : i - 1, fii])
                        # Also halt if the convergence gets significantly worse in one iteration
                        divergence_test_2 = convergence_strict >= np.min(conv_test[0: i - 1, fii]) * divergence_factor
                        if divergence_test_1 or divergence_test_2:
                            # If both measures of convergence are getting worse, we need to stop.
                            print("Calibration diverged at iteration: {}\nfor pol_i: {}\nfreq_i:\
                                 {}\nConvergence was: {}\nthreshold was: {}".format(i, pol_i, fi, conv_test[i - 1, fii], conv_thresh))
                            divergence_flag = True
                            break
            if divergence_flag:
                # If the solution diverged, back up one iteration and use the previous solution
                gain_curr = gain_old
                convergence[tile_use, fi] = conv_test[i - 1, fii]
                conv_iter_arr[tile_use, fi] = i - 1
            else:
                convergence[tile_use, fi] = np.abs(gain_curr - gain_old) * weight_invert(np.abs(gain_old))
                conv_iter_arr[tile_use, fi] = i
            if i == max_cal_iter:
                print("Calibration reach max iterations before converging for pol_i: {}\nfreq_i:\
                    {}\nConvergence was: {}\nthreshold was: {}".format(pol_i, fi, conv_test[i - 1, fii], conv_thresh))
            del A_ind_arr
            gain_arr[tile_use, fi] = gain_curr
        nan_i = np.where(np.isnan(gain_curr))[0]
        if nan_i.size > 0:
            # any gains with NANs -> all tiles for that freq will have NANs
            freq_nan_i = nan_i % n_freq
            freq_nan_i = freq_nan_i[np.unique(freq_nan_i)]
            vis_weight_ptr_use[pol_i][:, freq_nan_i] = 0
            weight[:, freq_nan_i] = 0
            gain_arr[nan_i] = 0
        cal_return['gain'][0][pol_i] = gain_arr
        cal_return['convergence'][0][pol_i] = convergence
        cal_return['n_converged'][0][pol_i] = n_converged
        cal_return['conv_iter'][0][pol_i] = conv_iter_arr

    n_vis_cal = np.size(np.nonzero(weight))
    cal_return['n_vis_cal'] = n_vis_cal
    return cal_return
