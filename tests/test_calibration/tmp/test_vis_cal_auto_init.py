import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_auto_init
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
import numpy as np

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_init")

@pytest.fixture
def base_dir():
    return '/home/jline/Dropbox/ADACS_work/semester_two/test_PyFHD_data/'

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

def convert_before_sav_to_py(test_dir):
    """Takes the before .sav file out of FHD function `vis_cal_auto_init`
    and converts into python formats"""

    sav_dict = convert_sav_to_dict(f"{test_dir}/before_vis_cal_auto_init.sav", "meh")


    # save_cal = {}
    # save_cal["n_pol"] = sav_dict["cal"][0]["n_pol"]

    # print(sav_dict['vis_arr'].shape)
    
    np.save("before_vis_cal_auto_init_vis_arr.npy", sav_dict['vis_arr'][0])
    np.save("before_vis_cal_auto_init_vis_model_arr.npy", sav_dict['vis_model_arr'][0])
    
    
    np.savez(f"{test_dir}/before_vis_cal_auto_init.npz",
                gain_list=sav_dict['gain_list'],
                convergence_list=sav_dict['convergence_list'],
                iter=sav_dict['iter'],
                base_gain=sav_dict['base_gain'],
                final_convergence_estimate=sav_dict['final_convergence_estimate'])
    
def convert_after_sav_to_npz(test_dir):
    """Takes the after .sav file out of FHD function `vis_cal_auto_init`
    and converts into a numpy format"""

    sav_dict = convert_sav_to_dict(f"{test_dir}/after_vis_cal_auto_init.sav", "meh")
    
    # np.savez(f"{test_dir}/after_vis_cal_auto_init.npz",
    #             gain_list=sav_dict['gain_list'],
    #             gain=sav_dict['gain'])

    np.savez(f"{test_dir}/after_vis_cal_auto_init.npz",
                gain_list=sav_dict['gain_list'],
                gain=sav_dict['gain'],    
                final_convergence_estimate = sav_dict['final_convergence_estimate'])
    
def convert_sav_to_npz(test_dir):
    """Load the inputs and outputs needed for testing `vis_cal_auto_init`"""
    convert_before_sav_to_npz(test_dir)
    convert_after_sav_to_npz(test_dir)


def get_data_npz(npz_file):
    """Load in a number of objects from a .npz file into a list"""

    data = np.load(npz_file)
    output = [data[key] for key in data.files]

    return output

def run_test(data_loc):
    """Runs the test on `vis_cal_auto_init` - reads in the data in `data_loc`,
    and then calls `vis_cal_auto_init`, checking the outputs match expectations"""

    gain_list, convergence_list, iter, base_gain, final_convergence_estimate_in = get_data_npz(f"{data_loc}/before_vis_cal_auto_init.npz")
    expected_gain_list, expected_gain, final_convergence_estimate_out = get_data_npz(f"{data_loc}/after_vis_cal_auto_init.npz")

    result_gain = vis_cal_auto_init(gain_list, convergence_list, iter, base_gain,
                                          final_convergence_estimate_in)
    
    ##Check returned gain is as expected
    assert np.allclose(expected_gain, result_gain, atol=1e-8)

    ##Check that the gain value is inserted in `gain_list` correctly
    assert np.allclose(gain_list[iter], expected_gain, atol=1e-8)


def test_pointsource1(base_dir):

    run_test(base_dir + '/pointsource1')

    


    
if __name__ == "__main__":

    base_dir = '/home/jline/Dropbox/ADACS_work/semester_two/test_PyFHD_data/'

    convert_sav_to_npz(base_dir + 'pointsource1')

    test_pointsource1(base_dir)