import pytest
from os import environ as env
from pathlib import Path
from PyFHD.calibration.calibration_utils import vis_extract_autocorr
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict
import deepdish as dd
import numpy as np
# from copy import deepcopy

@pytest.fixture
def base_dir():
    return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_loc):
    """Runs the test on `vis_extract_autocorr` - reads in the data in `data_loc`,
    and then calls `vis_extract_autocorr`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_loc, "before_vis_extract_autocorr.h5"))
    h5_after = dd.io.load(Path(data_loc, "after_vis_extract_autocorr.h5"))

    obs = h5_before['obs']
    vis_arr = h5_before['vis_arr']
    time_average = h5_before['time_average']
    
    expected_auto_corr = h5_after['auto_corr']
    expected_auto_tile_i = h5_after['auto_tile_i']

    result_auto_corr, result_auto_tile_i = vis_extract_autocorr(obs,
                                                    vis_arr,
                                                    time_average=time_average)

    ##Check returned values
    assert np.allclose(expected_auto_corr, result_auto_corr, atol=1e-8)

    assert result_auto_tile_i == expected_auto_tile_i

def test_pointsource1_vary1(base_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(Path(base_dir, "pointsource1_vary1"))

    
if __name__ == "__main__":

    """Bunch of functions used to convert the IDL FHD .sav files into
    something pythonic. This should only need to be run once for each test data
    set (a data set being one run of FHD on a given set of inputs with specific
    options)"""

    def convert_before_sav(test_dir):
        """Takes the before .sav file out of FHD function `vis_extract_autocorr`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(str(Path(test_dir, "before_vis_extract_autocorr.sav")), "meh")

        ##convert obs from a recarray into a dictionary
        obs = recarray_to_dict(sav_dict['obs'][0])
        
        ##fix the shape of vis_arr
        ##TODO might need to swap axis 1 and 2 here depending on what happens
        ##inside vis_extract_autocorr
        ##Currently I'm reading into a shape of (n_pol, n_freq, n_baselines)
        vis_arr = np.empty((obs["n_pol"], sav_dict['vis_arr'][0].shape[1],
                                          sav_dict['vis_arr'][0].shape[0]))
        for pol in range(obs["n_pol"]):
            vis_arr[pol] = sav_dict['vis_arr'][pol].transpose()
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['vis_arr'] = vis_arr
        h5_save_dict['time_average'] = sav_dict['time_average']

        ##save the thing 
        dd.io.save(Path(test_dir, "before_vis_extract_autocorr.h5"), h5_save_dict)

    def convert_after_sav(test_dir):
        """Takes the after .sav file out of FHD function `vis_extract_autocorr`
        and converts into an hdf5 format"""

        sav_dict = convert_sav_to_dict(str(Path(test_dir, "after_vis_extract_autocorr.sav")), "meh")

        ##super dictionary
        h5_save_dict = {}
        h5_save_dict['auto_corr'] = sav_dict['auto_corr']
        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']

        ##save the thing 
        dd.io.save(Path(test_dir, "after_vis_extract_autocorr.h5"), h5_save_dict)
        
    def convert_sav(test_dir):
        """Load the inputs and outputs needed for testing `vis_extract_autocorr`"""
        convert_before_sav(test_dir)
        convert_after_sav(test_dir)

    ##Start doing the conversion from IDL to Python

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'))

    ##Each test_set contains a run with a different set of inputs/options
    for test_set in ['pointsource1_vary1']:
        convert_sav(Path(base_dir, test_set))
        # run_test(Path(base_dir, test_set))