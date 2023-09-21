import pytest
from os import environ as env
from pathlib import Path
from PyFHD.gridding.gridding_utils import grid_beam_per_baseline
from PyFHD.pyfhd_tools.test_utils import get_data, get_data_items, recarray_to_dict
import deepdish as dd
from numpy.testing import assert_allclose
from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'grid_beam_per_baseline')

@pytest.fixture(scope="function", params=[1,2,3])
def number(request):
    return request.param

@pytest.fixture
def before_grid_per_baseline(data_dir, number):
    before_grid_per_baseline = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if before_grid_per_baseline.exists():
        return before_grid_per_baseline
    
    # For now put psf into dict
    psf, extra = get_data(
        data_dir,
        f'visibility_grid_psf_input_{number}.npy',
        f'visibility_grid_extra_input_{number}.npy'
    )
    psf = recarray_to_dict(psf)
    extra = recarray_to_dict(extra)
    pyfhd_config = {
        "psf_dim": psf["dim"],
        "psf_resolution": psf["resolution"],
        "beam_mask_threshold": psf["beam_mask_threshold"],
        "beam_clip_floor": True if extra['beam_clip_floor'] else False,
        "image_filter": "filter_uv_uniform",
    }
    uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
    xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
    image_top, psf_dim3, box_matrix, vis_n = get_data_items(
        data_dir,
        f'visibility_grid_uu_input_{number}.npy',
        f'visibility_grid_vv_input_{number}.npy',
        f'visibility_grid_ww_input_{number}.npy',
        f'visibility_grid_l_mode_input_{number}.npy',
        f'visibility_grid_m_mode_input_{number}.npy',
        f'visibility_grid_n_tracked_input_{number}.npy',
        f'visibility_grid_frequency_array_input_{number}.npy',
        f'visibility_grid_x_input_{number}.npy',
        f'visibility_grid_y_input_{number}.npy',
        f'visibility_grid_xmin_use_input_{number}.npy',
        f'visibility_grid_ymin_use_input_{number}.npy',
        f'visibility_grid_freq_i_input_{number}.npy',
        f'visibility_grid_bt_index_input_{number}.npy',
        f'visibility_grid_polarization_input_{number}.npy',
        f'visibility_grid_fbin_input_{number}.npy',
        f'visibility_grid_image_bot_input_{number}.npy',
        f'visibility_grid_image_top_input_{number}.npy',
        f'visibility_grid_psf_dim3_input_{number}.npy',
        f'visibility_grid_box_matrix_input_{number}.npy',
        f'visibility_grid_vis_n_input_{number}.npy',
    )

    h5_save_dict = {
        "psf": psf,
        "pyfhd_config": pyfhd_config,
        "uu": uu, 
        "vv": vv, 
        "ww": ww, 
        "l_mode": l_mode, 
        "m_mode": m_mode, 
        "n_tracked": n_tracked, 
        "frequency_array": frequency_array, 
        "x": x, 
        "y": y, 
        "xmin_use": xmin_use, 
        "ymin_use": ymin_use, 
        "freq_i": freq_i, 
        "bt_index": bt_index, 
        "polarization": polarization, 
        "fbin": fbin, 
        "image_bot": int(image_bot), 
        "image_top": int(image_top), 
        "psf_dim3": psf_dim3, 
        "box_matrix": box_matrix, 
        "vis_n": vis_n, 
    }

    dd.io.save(before_grid_per_baseline, h5_save_dict)
    
    return before_grid_per_baseline

@pytest.fixture
def after_grid_per_baseline(data_dir, number):
    after_grid_per_baseline = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if after_grid_per_baseline.exists():
        return after_grid_per_baseline
    
    h5_save_dict = {
        "box_matrix": get_data_items(data_dir, f'visibility_grid_box_matrix_output_{number}.npy',)
    }
    
    dd.io.save(after_grid_per_baseline, h5_save_dict)

    return after_grid_per_baseline

def test_grid_per_baseline(before_grid_per_baseline: Path, after_grid_per_baseline: Path):
    h5_before = dd.io.load(before_grid_per_baseline)
    h5_after = dd.io.load(after_grid_per_baseline)

    output_box_matrix = grid_beam_per_baseline(
        h5_before["psf"], 
        h5_before["pyfhd_config"],
        RootLogger(1),
        h5_before["uu"], 
        h5_before["vv"],
        h5_before["ww"],
        h5_before["l_mode"],
        h5_before["m_mode"],
        h5_before["n_tracked"],
        h5_before["frequency_array"],
        h5_before["x"],
        h5_before["y"],
        h5_before["xmin_use"],
        h5_before["ymin_use"],
        h5_before["freq_i"],
        h5_before["bt_index"],
        h5_before["polarization"],
        h5_before["image_bot"],
        h5_before["image_top"],
        h5_before["psf_dim3"],
        h5_before["box_matrix"],
        h5_before["vis_n"],
    )

    assert_allclose(output_box_matrix, h5_after["box_matrix"], atol = 1e-8)

# def test_grid_beam_one(data_dir):
#     psf, extras = get_data(
#         data_dir,
#         'visibility_grid_psf_input_1.npy',
#         'visibility_grid_extra_input_1.npy'
#     )
#     uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
#     xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
#     image_top, psf_dim3, box_matrix, vis_n, expected_box_matrix = get_data_items(
#         data_dir,
#         'visibility_grid_uu_input_1.npy',
#         'visibility_grid_vv_input_1.npy',
#         'visibility_grid_ww_input_1.npy',
#         'visibility_grid_l_mode_input_1.npy',
#         'visibility_grid_m_mode_input_1.npy',
#         'visibility_grid_n_tracked_input_1.npy',
#         'visibility_grid_frequency_array_input_1.npy',
#         'visibility_grid_x_input_1.npy',
#         'visibility_grid_y_input_1.npy',
#         'visibility_grid_xmin_use_input_1.npy',
#         'visibility_grid_ymin_use_input_1.npy',
#         'visibility_grid_freq_i_input_1.npy',
#         'visibility_grid_bt_index_input_1.npy',
#         'visibility_grid_polarization_input_1.npy',
#         'visibility_grid_fbin_input_1.npy',
#         'visibility_grid_image_bot_input_1.npy',
#         'visibility_grid_image_top_input_1.npy',
#         'visibility_grid_psf_dim3_input_1.npy',
#         'visibility_grid_box_matrix_input_1.npy',
#         'visibility_grid_vis_n_input_1.npy',
#         'visibility_grid_box_matrix_output_1.npy',
#     )
#     beam_clip_floor = extras['beam_clip_floor'][0]
#     output_box_matrix = grid_beam_per_baseline(
#         psf, 
#         uu, 
#         vv,
#         ww,
#         l_mode,
#         m_mode,
#         n_tracked,
#         frequency_array,
#         x,
#         y,
#         xmin_use,
#         ymin_use,
#         freq_i,
#         bt_index,
#         polarization,
#         fbin,
#         int(image_bot),
#         int(image_top),
#         psf_dim3,
#         box_matrix,
#         vis_n,
#         beam_clip_floor = beam_clip_floor,
#     )
#     assert np.max(np.abs(expected_box_matrix.real - output_box_matrix.real)) < 1.5e-4
#     assert np.max(np.abs(expected_box_matrix.imag - output_box_matrix.imag)) < 1.5e-4

# def test_grid_beam_two(data_dir):
#     psf, extras = get_data(
#         data_dir,
#         'visibility_grid_psf_input_2.npy',
#         'visibility_grid_extra_input_2.npy'
#     )
#     uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
#     xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
#     image_top, psf_dim3, box_matrix, vis_n, expected_box_matrix = get_data_items(
#         data_dir,
#         'visibility_grid_uu_input_2.npy',
#         'visibility_grid_vv_input_2.npy',
#         'visibility_grid_ww_input_2.npy',
#         'visibility_grid_l_mode_input_2.npy',
#         'visibility_grid_m_mode_input_2.npy',
#         'visibility_grid_n_tracked_input_2.npy',
#         'visibility_grid_frequency_array_input_2.npy',
#         'visibility_grid_x_input_2.npy',
#         'visibility_grid_y_input_2.npy',
#         'visibility_grid_xmin_use_input_2.npy',
#         'visibility_grid_ymin_use_input_2.npy',
#         'visibility_grid_freq_i_input_2.npy',
#         'visibility_grid_bt_index_input_2.npy',
#         'visibility_grid_polarization_input_2.npy',
#         'visibility_grid_fbin_input_2.npy',
#         'visibility_grid_image_bot_input_2.npy',
#         'visibility_grid_image_top_input_2.npy',
#         'visibility_grid_psf_dim3_input_2.npy',
#         'visibility_grid_box_matrix_input_2.npy',
#         'visibility_grid_vis_n_input_2.npy',
#         'visibility_grid_box_matrix_output_2.npy',
#     )
#     beam_clip_floor = extras['beam_clip_floor'][0]
#     output_box_matrix = grid_beam_per_baseline(
#         psf, 
#         uu, 
#         vv,
#         ww,
#         l_mode,
#         m_mode,
#         n_tracked,
#         frequency_array,
#         x,
#         y,
#         xmin_use,
#         ymin_use,
#         freq_i,
#         bt_index,
#         polarization,
#         fbin,
#         int(image_bot),
#         int(image_top),
#         psf_dim3,
#         box_matrix,
#         vis_n,
#         beam_clip_floor = beam_clip_floor,
#     )
#     assert np.max(np.abs(expected_box_matrix.real - output_box_matrix.real)) < 1e-8
#     assert np.max(np.abs(expected_box_matrix.imag - output_box_matrix.imag)) < 1e-8

# def test_grid_beam_three(data_dir):
#     psf, extras = get_data(
#         data_dir,
#         'visibility_grid_psf_input_3.npy',
#         'visibility_grid_extra_input_3.npy'
#     )
#     uu, vv, ww, l_mode, m_mode, n_tracked, frequency_array, x, y,\
#     xmin_use, ymin_use, freq_i, bt_index, polarization, fbin, image_bot,\
#     image_top, psf_dim3, box_matrix, vis_n, expected_box_matrix = get_data_items(
#         data_dir,
#         'visibility_grid_uu_input_3.npy',
#         'visibility_grid_vv_input_3.npy',
#         'visibility_grid_ww_input_3.npy',
#         'visibility_grid_l_mode_input_3.npy',
#         'visibility_grid_m_mode_input_3.npy',
#         'visibility_grid_n_tracked_input_3.npy',
#         'visibility_grid_frequency_array_input_3.npy',
#         'visibility_grid_x_input_3.npy',
#         'visibility_grid_y_input_3.npy',
#         'visibility_grid_xmin_use_input_3.npy',
#         'visibility_grid_ymin_use_input_3.npy',
#         'visibility_grid_freq_i_input_3.npy',
#         'visibility_grid_bt_index_input_3.npy',
#         'visibility_grid_polarization_input_3.npy',
#         'visibility_grid_fbin_input_3.npy',
#         'visibility_grid_image_bot_input_3.npy',
#         'visibility_grid_image_top_input_3.npy',
#         'visibility_grid_psf_dim3_input_3.npy',
#         'visibility_grid_box_matrix_input_3.npy',
#         'visibility_grid_vis_n_input_3.npy',
#         'visibility_grid_box_matrix_output_3.npy',
#     )
#     beam_clip_floor = extras['beam_clip_floor'][0]
#     output_box_matrix = grid_beam_per_baseline(
#         psf, 
#         uu, 
#         vv,
#         ww,
#         l_mode,
#         m_mode,
#         n_tracked,
#         frequency_array,
#         x,
#         y,
#         xmin_use,
#         ymin_use,
#         freq_i,
#         bt_index,
#         polarization,
#         fbin,
#         int(image_bot),
#         int(image_top),
#         psf_dim3,
#         box_matrix,
#         vis_n,
#         beam_clip_floor = beam_clip_floor,
#     )
#     assert np.max(np.abs(expected_box_matrix.real - output_box_matrix.real)) < 1e-8
#     assert np.max(np.abs(expected_box_matrix.imag - output_box_matrix.imag)) < 1e-8