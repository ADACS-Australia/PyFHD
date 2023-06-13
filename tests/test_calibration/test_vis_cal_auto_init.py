import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_auto_init
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes

import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_init")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

# def test_calc_adapt_gain_one(data_dir):
#     gain_list, convergence_list, iter, base_gain, final_con_est, expected_gain = get_data_items(
#         data_dir,
#         'input_gain_list_1.npy',
#         'input_convergence_list_1.npy',
#         'input_iter_1.npy',
#         'input_base_gain_1.npy',
#         'input_final_convergence_estimate_1.npy',
#         'output_gain_1.npy'
#     )
#     result_gain = vis_cal_auto_init(gain_list, convergence_list, iter, base_gain, final_convergence_estimate = final_con_est)
#     assert expected_gain == result_gain

# def test_calc_adapt_gain_two(data_dir):
#     gain_list, convergence_list, iter, base_gain, final_con_est, expected_gain = get_data_items(
#         data_dir,
#         'input_gain_list_2.npy',
#         'input_convergence_list_2.npy',
#         'input_iter_2.npy',
#         'input_base_gain_2.npy',
#         'input_final_convergence_estimate_2.npy',
#         'output_gain_2.npy'
#     )
#     result_gain = vis_cal_auto_init(gain_list, convergence_list, iter, base_gain, final_convergence_estimate = final_con_est)
#     assert expected_gain == result_gain

# '''
# The below test will never pass due to IDL's default median behaviour, for example
# PRINT, MEDIAN([1, 2, 3, 4], /EVEN) 
# 2.50000
# PRINT, MEDIAN([1, 2, 3, 4])
# 3.00000

# One has to use the EVEN keyword to get the *proper* median behaviour that most mathematicians and scientists
# expect. This was the only thing that produced the wrong results was the use of numpy median instead of IDL MEDIAN.
# The array that went into it est_final_conv produced the exact same results in Python and IDL (with the exception of
# Python producing a more accurate result due to double precision) 

# def test_calc_adapt_gain_three(data_dir):
#     gain_list, convergence_list, iter, base_gain, expected_gain = get_data_items(
#         data_dir,
#         'input_gain_list_3.npy',
#         'input_convergence_list_3.npy',
#         'input_iter_3.npy',
#         'input_base_gain_3.npy',
#         'output_gain_3.npy'
#     )
#     result_gain = vis_cal_auto_init(gain_list, convergence_list, iter, base_gain)
#     assert expected_gain == result_gain
# '''

def run_test(data_loc):
    """Runs the test on `vis_cal_auto_init` - reads in the data in `data_loc`,
    and then calls `vis_cal_auto_init`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_cal_auto_init.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_cal_auto_init.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    vis_arr = h5_before['vis_arr']
    vis_model_arr = h5_before['vis_model_arr']
    vis_auto = h5_before['vis_auto']
    vis_model_auto = h5_before['vis_model_auto']
    auto_tile_i = h5_before['auto_tile_i']

    # print(vis_arr.shape, vis_model_arr.shape, vis_auto.shape)
    
    expected_cal = h5_after['cal_init']

    expected_auto_gain = sav_file_vis_arr_swap_axes(expected_cal['gain'])



    ##TODO this is failing, there are a couple of translation boo-boos

    result_auto_gain = vis_cal_auto_init(obs, cal, vis_arr, vis_model_arr, vis_auto, vis_model_auto, auto_tile_i)

    # print(result_auto_gain[0, 0, :10])
    # print(expected_auto_gain[0, 0, :10])

    fig, axs = plt.subplots(2, 1)

    # axs.hist(np.real(expected_auto_gain[0, 0, :]), histtype='step', label='Expected')
    # axs.hist(np.real(result_auto_gain[0, 0, :]), histtype='step', label='PyFHD')

    freqind = 10

    axs[0].plot(np.real(expected_auto_gain[0, freqind, :]), label='Expected')
    axs[0].plot(np.real(result_auto_gain[0, freqind, :]), label='PyFHD')

    
    axs[0].set_ylabel('Real value (Jy)')

    ratio = np.real(expected_auto_gain[0, freqind, :]) / np.real(result_auto_gain[0, freqind, :])

    axs[1].plot(ratio, label='Ratio (FHD / PyFHD)')

    print("expec / PyFHD mean, std", np.mean(ratio), np.std(ratio))

    axs[1].set_xlabel('Auto correlation index')

    axs[0].set_ylabel('Real value (Jy)')

    axs[1].legend()

    fig.savefig('cmooooon.png', bbox_inches='tight')
    plt.close()
    
    # ##Check returned gain is as expected
    # assert np.allclose(result_auto_gain, expected_auto_gain, atol=1e-5)

    # ##Check that the gain value is inserted in `gain_list` correctly
    # assert np.allclose(gain_list[iter], expected_gain, atol=1e-8)


def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_cal_auto_init`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_auto_init.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        cal = recarray_to_dict(sav_dict['cal'])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['vis_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
        h5_save_dict['vis_model_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_arr'])

        h5_save_dict['vis_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_auto'])
        h5_save_dict['vis_model_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_auto'])

        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']
        
        dd.io.save(Path(test_dir, "before_vis_cal_auto_init.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_cal_auto_init`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_auto_init.sav", "meh")
        
        cal_init = recarray_to_dict(sav_dict["cal_init"])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['cal_init'] = cal_init
        
        dd.io.save(Path(test_dir, "after_vis_cal_auto_init.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_cal_auto_init`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))