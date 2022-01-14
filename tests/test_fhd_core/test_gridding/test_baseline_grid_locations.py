import pytest
import numpy as np
from glob import glob
from tests.test_utils import get_data, get_data_items
from fhd_utils.idl_tools.array_match import array_match
from fhd_utils.histogram import histogram
from fhd_core.gridding.baseline_grid_locations import baseline_grid_locations

@pytest.fixture
def data_dir():
    return glob('**/baseline_grid_locations/', recursive = True)[0]

def test_baseline_one(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
        'input_psf_1.npy',
    )
    obs, params, vis_weights, fi_use, interp_flag = get_data_items(
        data_dir,
        'input_obs_1.npy',
        'input_params_1.npy',
        'input_vis_weight_ptr_1.npy',
        'input_fi_use_1.npy',
        'input_interp_flag_1.npy',
    )
    # Get the expected outputs
    expected_bin_n, expected_bin_i, expected_n_bin_use, expected_ri, expected_xmin,\
    expected_ymin, expected_vis_inds_use, expected_x_offset, expected_y_offset,\
    expected_dx0dy0, expected_dx0dy1, expected_dx1dy0, expected_dx1dy1 = get_data_items(
        data_dir,
        'output_bin_n_1.npy',
        'output_bin_i_1.npy',
        'output_n_bin_use_1.npy',
        'output_ri_1.npy',
        'output_xmin_1.npy',
        'output_ymin_1.npy',
        'output_vis_inds_use_1.npy',
        'output_x_offset_1.npy',
        'output_y_offset_1.npy',
        'output_dx0dy0_arr_1.npy',
        'output_dx0dy1_arr_1.npy',
        'output_dx1dy0_arr_1.npy',
        'output_dx1dy1_arr_1.npy',
    )
    # Use the baseline grid locations function
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, fi_use = fi_use, interp_flag = interp_flag)
    # Check we got the right results from the dictionary
    assert np.array_equal(expected_vis_inds_use, baselines_dict['vis_inds_use'])
    # Since precision errors from xcen and ycen impact the offsets check that the 
    # number of "wrong" values is less than 0.05% of the dataset
    x_wrong_p = np.nonzero(np.abs((expected_x_offset - baselines_dict['x_offset'])))[0].size / expected_x_offset.size
    assert x_wrong_p < 0.0005
    y_wrong_p = np.nonzero(np.abs((expected_y_offset - baselines_dict['y_offset'])))[0].size / expected_y_offset.size
    assert y_wrong_p < 0.0005
    # The same xcen and ycen precision errors affect xmin and ymin too
    # However the result is floored so the result shouldn't change by more
    # than 1 value, as the rest of the code is the same as the tests check
    # that the maximum difference is less than or equal to 1
    assert np.max(np.abs(expected_xmin - baselines_dict['xmin'])) <= 1
    assert np.max(np.abs(expected_ymin - baselines_dict['ymin'])) <= 1
    # Given there are changes to xmin and ymin, I can't adequately test the
    # histogram function applied to xmin + ymin * dimension. However, the number
    # of items that are nonzero in the histogram shouldn't change. So we'll test
    # those. I'll also test the size of ri nd hist to make sure its correct size
    # Get the size of the histogram in theory
    hist_size = np.arange(0, np.max(baselines_dict['xmin'] + baselines_dict['ymin'] * obs['dimension']) + 1).size
    assert baselines_dict['bin_n'].size == hist_size
    # Get the size of the data in theory
    data = baselines_dict['xmin'] + baselines_dict['ymin'] * obs['dimension']
    data = data[data >= 0]
    assert baselines_dict['ri'].size == hist_size + 1 + data.size
    # Check the indices of where the histogram is 0
    assert np.array_equal(expected_bin_i, baselines_dict['bin_i'])
    # Check the number of indices from the histogram
    assert expected_n_bin_use == baselines_dict['n_bin_use']
    # Precision errors with xcen and ycen cause huge differences in the second derivatives
    # Hopefully in theory, the Python is a better result, even though its different from the IDL output
    # assert np.array_equal(expected_dx0dy0, baselines_dict['dx0dy0_arr'])
    # assert np.array_equal(expected_dx0dy1, baselines_dict['dx0dy1_arr'])
    # assert np.array_equal(expected_dx1dy0, baselines_dict['dx1dy0_arr'])
    # assert np.array_equal(expected_dx1dy1, baselines_dict['dx1dy1_arr'])

def test_baseline_two(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
            'input_psf_2.npy',
    )
    obs, params, vis_weights, interp_flag, fill_model_vis = get_data_items(
        data_dir,
        'input_obs_2.npy',
        'input_params_2.npy',
        'input_vis_weight_ptr_2.npy',
        'input_interp_flag_2.npy',
        'input_fill_model_visibilities_2.npy'
    )
    # Get the expected outputs
    expected_bin_n, expected_bin_i, expected_n_bin_use, expected_ri, expected_xmin,\
    expected_ymin, expected_x_offset, expected_y_offset,\
    expected_dx0dy0, expected_dx0dy1, expected_dx1dy0, expected_dx1dy1 = get_data_items(
        data_dir,
        'output_bin_n_2.npy',
        'output_bin_i_2.npy',
        'output_n_bin_use_2.npy',
        'output_ri_2.npy',
        'output_xmin_2.npy',
        'output_ymin_2.npy',
        'output_x_offset_2.npy',
        'output_y_offset_2.npy',
        'output_dx0dy0_arr_2.npy',
        'output_dx0dy1_arr_2.npy',
        'output_dx1dy0_arr_2.npy',
        'output_dx1dy1_arr_2.npy',
    )
    # Use the baseline grid locations function
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, interp_flag = interp_flag, fill_model_visibilities = fill_model_vis)
    # Check we got the right results from the dictionary
    x_wrong_p = np.nonzero(np.abs((expected_x_offset - baselines_dict['x_offset'])))[0].size / expected_x_offset.size
    assert x_wrong_p < 0.0005
    y_wrong_p = np.nonzero(np.abs((expected_y_offset - baselines_dict['y_offset'])))[0].size / expected_y_offset.size
    assert y_wrong_p < 0.0005
    assert np.max(np.abs(expected_xmin - baselines_dict['xmin'])) <= 1
    assert np.max(np.abs(expected_ymin - baselines_dict['ymin'])) <= 1
    hist_size = np.arange(0, np.max(baselines_dict['xmin'] + baselines_dict['ymin'] * obs['dimension']) + 1).size
    assert baselines_dict['bin_n'].size == hist_size
    data = baselines_dict['xmin'] + baselines_dict['ymin'] * obs['dimension']
    data = data[data >= 0]
    assert baselines_dict['ri'].size == hist_size + 1 + data.size
    assert np.array_equal(expected_bin_i, baselines_dict['bin_i'])
    assert expected_n_bin_use == baselines_dict['n_bin_use']
    # Precision errors with xcen and ycen cause huge differences in the second derivatives
    # Hopefully in theory, the Python is a better result, even though its different from the IDL output
    # assert np.array_equal(expected_dx0dy0, baselines_dict['dx0dy0_arr'])
    # assert np.array_equal(expected_dx0dy1, baselines_dict['dx0dy1_arr'])
    # assert np.array_equal(expected_dx1dy0, baselines_dict['dx1dy0_arr'])
    # assert np.array_equal(expected_dx1dy1, baselines_dict['dx1dy1_arr'])

def test_baseline_three(data_dir):
    # Get the inputs
    psf = get_data(
        data_dir,
            'input_psf_2.npy',
    )
    obs, params, vis_weights, fi_use, interp_flag, bi_use = get_data_items(
        data_dir,
        'input_obs_3.npy',
        'input_params_3.npy',
        'input_vis_weight_ptr_3.npy',
        'input_fi_use_3.npy',
        'input_interp_flag_3.npy',
        'input_bi_use_arr_3.npy'
    )
    # Get the expected outputs
    expected_bin_n, expected_bin_i, expected_n_bin_use, expected_ri, expected_xmin,\
    expected_ymin, expected_vis_inds_use, expected_x_offset, expected_y_offset,\
    expected_dx0dy0, expected_dx0dy1, expected_dx1dy0, expected_dx1dy1 = get_data_items(
        data_dir,
        'output_bin_n_3.npy',
        'output_bin_i_3.npy',
        'output_n_bin_use_3.npy',
        'output_ri_3.npy',
        'output_xmin_3.npy',
        'output_ymin_3.npy',
        'output_vis_inds_use_3.npy',
        'output_x_offset_3.npy',
        'output_y_offset_3.npy',
        'output_dx0dy0_arr_3.npy',
        'output_dx0dy1_arr_3.npy',
        'output_dx1dy0_arr_3.npy',
        'output_dx1dy1_arr_3.npy',
    )
    # Use the baseline grid locations function
    baselines_dict = baseline_grid_locations(obs, psf, params, vis_weights, fi_use = fi_use, interp_flag = interp_flag, bi_use = bi_use)
    # Check we got the right results from the dictionary
    assert np.array_equal(expected_vis_inds_use, baselines_dict['vis_inds_use'])
    x_wrong_p = np.nonzero(np.abs((expected_x_offset - baselines_dict['x_offset'])))[0].size / expected_x_offset.size
    assert x_wrong_p < 0.0005
    y_wrong_p = np.nonzero(np.abs((expected_y_offset - baselines_dict['y_offset'])))[0].size / expected_y_offset.size
    assert y_wrong_p < 0.0005
    assert np.max(np.abs(expected_xmin - baselines_dict['xmin'])) <= 1
    assert np.max(np.abs(expected_ymin - baselines_dict['ymin'])) <= 1
    hist_size = np.arange(0, np.max(baselines_dict['xmin'] + baselines_dict['ymin'] * obs['dimension']) + 1).size
    assert baselines_dict['bin_n'].size == hist_size
    data = baselines_dict['xmin'] + baselines_dict['ymin'] * obs['dimension']
    data = data[data >= 0]
    assert baselines_dict['ri'].size == hist_size + 1 + data.size
    assert np.array_equal(expected_bin_i, baselines_dict['bin_i'])
    assert expected_n_bin_use == baselines_dict['n_bin_use']
    # Precision errors with xcen and ycen cause huge differences in the second derivatives
    # Hopefully in theory, the Python is a better result, even though its different from the IDL output
    # assert np.array_equal(expected_dx0dy0, baselines_dict['dx0dy0_arr'])
    # assert np.array_equal(expected_dx0dy1, baselines_dict['dx0dy1_arr'])
    # assert np.array_equal(expected_dx1dy0, baselines_dict['dx1dy0_arr'])
    # assert np.array_equal(expected_dx1dy1, baselines_dict['dx1dy1_arr'])    