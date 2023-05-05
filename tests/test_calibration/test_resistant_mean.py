import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import resistant_mean
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "resistant_mean")

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `resistant_mean` - reads in the data in `data_loc`,
    and then calls `resistant_mean`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_resistant_mean.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_resistant_mean.h5"))

    input_array = h5_before['input_array']
    deviations = h5_before['deviations']
    
    expected_res_mean = h5_after['res_mean_data']

    result_res_mean = resistant_mean(input_array, deviations)
    
    print(result_res_mean, expected_res_mean)

    assert np.allclose(result_res_mean, expected_res_mean, atol=1e-4)

    # ##Check that the gain value is inserted in `gain_list` correctly
    # assert np.allclose(gain_list[iter], expected_gain, atol=1e-8)

def test_pointsource1_vary(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `resistant_mean`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/before_resistant_mean.sav", "meh")

        input_array = sav_dict['input_array']
        deviations = sav_dict['deviations']
        
        print(input_array.shape)

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['input_array'] = input_array
        h5_save_dict['deviations'] = deviations
        
        dd.io.save(Path(test_dir, "before_resistant_mean.h5"), h5_save_dict)
        
    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `resistant_mean`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(f"{test_dir}/after_resistant_mean.sav", "meh")
        
        res_mean_data = sav_dict["res_mean_data"]

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['res_mean_data'] = res_mean_data
        
        dd.io.save(Path(test_dir, "after_resistant_mean.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `resistant_mean`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))

        # convert_before_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))