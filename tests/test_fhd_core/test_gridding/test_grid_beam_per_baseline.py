import pytest
import numpy as np
from glob import glob
from fhd_core.gridding.grid_beam_per_baseline import grid_beam_per_baseline
from tests.test_utils import get_data, get_data_items

@pytest.fixture
def data_dir():
    return glob('**/grid_beam_per_baseline/', recursive = True)[0]

def test_grid_beam_one(data_dir):
    psf, extras = get_data(
        data_dir,
        'visibility_grid_psf_input_1.npy',
        'visibility_grid_extra_input_1.npy'
    )
    uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
    xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
    image_top, psf_dim3, box_matrix, vis_n, expected_box_matrix = get_data_items(
        data_dir,
        'visibility_grid_uu_input_1.npy',
        'visibility_grid_vv_input_1.npy',
        'visibility_grid_ww_input_1.npy',
        'visibility_grid_l_mode_input_1.npy',
        'visibility_grid_m_mode_input_1.npy',
        'visibility_grid_n_tracked_input_1.npy',
        'visibility_grid_frequency_array_input_1.npy',
        'visibility_grid_x_input_1.npy',
        'visibility_grid_y_input_1.npy',
        'visibility_grid_xmin_use_input_1.npy',
        'visibility_grid_ymin_use_input_1.npy',
        'visibility_grid_freq_i_input_1.npy',
        'visibility_grid_bt_index_input_1.npy',
        'visibility_grid_polarization_input_1.npy',
        'visibility_grid_fbin_input_1.npy',
        'visibility_grid_image_bot_input_1.npy',
        'visibility_grid_image_top_input_1.npy',
        'visibility_grid_psf_dim3_input_1.npy',
        'visibility_grid_box_matrix_input_1.npy',
        'visibility_grid_vis_n_input_1.npy',
        'visibility_grid_box_matrix_output_1.npy',
    )
    beam_clip_floor = extras['beam_clip_floor'][0]
    output_box_matrix = grid_beam_per_baseline(
        psf, 
        uu, 
        vv,
        ww,
        l_mode,
        m_mode,
        n_tracked,
        frequency_array,
        x,
        y,
        xmin_use,
        ymin_use,
        freq_i,
        bt_index,
        polarization,
        fbin,
        int(image_bot),
        int(image_top),
        psf_dim3,
        box_matrix,
        vis_n,
        beam_clip_floor = beam_clip_floor,
    )
    '''
    The explanation for this threshold comes from line 142
    box_matrix[:vis_n, :] *=  np.reshape(psf_val_ref / ref_temp, (psf_val_ref.size, 1))
    Up until this point in this test, all maximum error values are less than single precision
    (1e-8, in fact most are below 1e-10) however after this point, which is the last code applied to box_matrix, 
    the values are out by 1e-5 after a simple division on two small vectors. 
    That error of 1e-5 is then multiplied to box_matrix.
    The 1.5e-4 is due to rounding errors from that precision error.
    '''
    assert np.max(np.abs(expected_box_matrix.real - output_box_matrix.real)) < 1.5e-4
    assert np.max(np.abs(expected_box_matrix.imag - output_box_matrix.imag)) < 1.5e-4

def test_grid_beam_two(data_dir):
    psf, extras = get_data(
        data_dir,
        'visibility_grid_psf_input_2.npy',
        'visibility_grid_extra_input_2.npy'
    )
    uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
    xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
    image_top, psf_dim3, box_matrix, vis_n, expected_box_matrix = get_data_items(
        data_dir,
        'visibility_grid_uu_input_2.npy',
        'visibility_grid_vv_input_2.npy',
        'visibility_grid_ww_input_2.npy',
        'visibility_grid_l_mode_input_2.npy',
        'visibility_grid_m_mode_input_2.npy',
        'visibility_grid_n_tracked_input_2.npy',
        'visibility_grid_frequency_array_input_2.npy',
        'visibility_grid_x_input_2.npy',
        'visibility_grid_y_input_2.npy',
        'visibility_grid_xmin_use_input_2.npy',
        'visibility_grid_ymin_use_input_2.npy',
        'visibility_grid_freq_i_input_2.npy',
        'visibility_grid_bt_index_input_2.npy',
        'visibility_grid_polarization_input_2.npy',
        'visibility_grid_fbin_input_2.npy',
        'visibility_grid_image_bot_input_2.npy',
        'visibility_grid_image_top_input_2.npy',
        'visibility_grid_psf_dim3_input_2.npy',
        'visibility_grid_box_matrix_input_2.npy',
        'visibility_grid_vis_n_input_2.npy',
        'visibility_grid_box_matrix_output_2.npy',
    )
    beam_clip_floor = extras['beam_clip_floor'][0]
    output_box_matrix = grid_beam_per_baseline(
        psf, 
        uu, 
        vv,
        ww,
        l_mode,
        m_mode,
        n_tracked,
        frequency_array,
        x,
        y,
        xmin_use,
        ymin_use,
        freq_i,
        bt_index,
        polarization,
        fbin,
        int(image_bot),
        int(image_top),
        psf_dim3,
        box_matrix,
        vis_n,
        beam_clip_floor = beam_clip_floor,
    )
    assert np.max(np.abs(expected_box_matrix.real - output_box_matrix.real)) < 1e-8
    assert np.max(np.abs(expected_box_matrix.imag - output_box_matrix.imag)) < 1e-8

def test_grid_beam_three(data_dir):
    psf, extras = get_data(
        data_dir,
        'visibility_grid_psf_input_3.npy',
        'visibility_grid_extra_input_3.npy'
    )
    uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
    xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
    image_top, psf_dim3, box_matrix, vis_n, expected_box_matrix = get_data_items(
        data_dir,
        'visibility_grid_uu_input_3.npy',
        'visibility_grid_vv_input_3.npy',
        'visibility_grid_ww_input_3.npy',
        'visibility_grid_l_mode_input_3.npy',
        'visibility_grid_m_mode_input_3.npy',
        'visibility_grid_n_tracked_input_3.npy',
        'visibility_grid_frequency_array_input_3.npy',
        'visibility_grid_x_input_3.npy',
        'visibility_grid_y_input_3.npy',
        'visibility_grid_xmin_use_input_3.npy',
        'visibility_grid_ymin_use_input_3.npy',
        'visibility_grid_freq_i_input_3.npy',
        'visibility_grid_bt_index_input_3.npy',
        'visibility_grid_polarization_input_3.npy',
        'visibility_grid_fbin_input_3.npy',
        'visibility_grid_image_bot_input_3.npy',
        'visibility_grid_image_top_input_3.npy',
        'visibility_grid_psf_dim3_input_3.npy',
        'visibility_grid_box_matrix_input_3.npy',
        'visibility_grid_vis_n_input_3.npy',
        'visibility_grid_box_matrix_output_3.npy',
    )
    beam_clip_floor = extras['beam_clip_floor'][0]
    output_box_matrix = grid_beam_per_baseline(
        psf, 
        uu, 
        vv,
        ww,
        l_mode,
        m_mode,
        n_tracked,
        frequency_array,
        x,
        y,
        xmin_use,
        ymin_use,
        freq_i,
        bt_index,
        polarization,
        fbin,
        int(image_bot),
        int(image_top),
        psf_dim3,
        box_matrix,
        vis_n,
        beam_clip_floor = beam_clip_floor,
    )
    '''
    The explanation for this threshold comes from line 142
    box_matrix[:vis_n, :] *=  np.reshape(psf_val_ref / ref_temp, (psf_val_ref.size, 1))
    Up until this point in this test, all maximum error values are less than single precision
    (1e-8, in fact most are below 1e-10) however after this point, which is the last code applied to box_matrix, 
    the values are out by 1e-5 after a simple division on two small vectors. 
    That error of 1e-5 is then multiplied to box_matrix.
    The 1.5e-4 is due to rounding errors from that precision error.
    '''
    assert np.max(np.abs(expected_box_matrix.real - output_box_matrix.real)) < 1e-8
    assert np.max(np.abs(expected_box_matrix.imag - output_box_matrix.imag)) < 1e-8