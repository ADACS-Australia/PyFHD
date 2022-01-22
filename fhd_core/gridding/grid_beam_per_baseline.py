import numpy as np
from PyFHD.fhd_core.gridding.dirty_image_generate import dirty_image_generate
from math import pi

def grid_beam_per_baseline(psf, uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,
                           xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot, 
                           image_top, psf_dim3, box_matrix, vis_n, beam_clip_floor = False, beam_int = None, 
                           beam2_int = None, n_grp_use = None, degrid_flag = False):
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
            not_real = True
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