import pytest
from os import environ as env
from pathlib import Path
import deepdish as dd
import numpy as np

from PyFHD.calibration.calibration_utils import vis_extract_autocorr
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'vis_extract_autocorr')

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run2', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"]]

@pytest.fixture()
def before_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    ##convert obs from a recarray into a dictionary
    obs = recarray_to_dict(sav_dict['obs'])
    
    ##fix the shape of vis_arr
    ##inside vis_extract_autocorr
    ##Currently I'm reading into a shape of (n_pol, n_freq, n_baselines)
    vis_arr = np.empty((obs["n_pol"], sav_dict['vis_arr'][0].shape[1],
                                        sav_dict['vis_arr'][0].shape[0]),
                                        dtype=complex)
    for pol in range(obs["n_pol"]):
        vis_arr[pol, :, :] = sav_dict['vis_arr'][pol].transpose()
    
    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = obs
    h5_save_dict['vis_arr'] = vis_arr
    if ('time_average' in sav_dict):
        h5_save_dict['time_average'] = sav_dict['time_average']
    else:
        # Set to default
        h5_save_dict['time_average'] = True
    dd.io.save(before_file, h5_save_dict)

    return before_file

@pytest.fixture()
def after_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file
    
    sav_file = after_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    ##super dictionary
    h5_save_dict = {}
    h5_save_dict['auto_corr'] = sav_dict['auto_corr']
    h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_points_offzenith_zenith_1088716296(before_file, after_file):
    """Runs the test on `vis_extract_autocorr` - reads in the data in before_file and after_file,
    and then calls `vis_extract_autocorr`, checking the outputs match expectations"""
    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}")

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    obs = h5_before['obs']
    vis_arr = h5_before['vis_arr']
    time_average = h5_before['time_average']
    
    expected_auto_corr = h5_after['auto_corr']
    expected_auto_tile_i = h5_after['auto_tile_i']

    result_auto_corr, result_auto_tile_i = vis_extract_autocorr(obs,
                                                    vis_arr,
                                                    time_average=time_average)

    ##Outputs from .sav file can be an a 1D array containging 2D arrays
    ##PyFHD is making 3D arrays. So test things match as a loop
    ##over polarisations. Furthermore, FHD and PyFHD have different
    ##axes orders, so transpose the expected values

    for pol_i in range(obs['n_pol']):
        assert np.allclose(expected_auto_corr[pol_i].transpose(),
                           result_auto_corr[pol_i], atol=1e-8)
        
    assert np.array_equal(result_auto_tile_i, expected_auto_tile_i)

    
# if __name__ == "__main__":

#     """Bunch of functions used to convert the IDL FHD .sav files into
#     something pythonic. This should only need to be run once for each test data
#     set (a data set being one run of FHD on a given set of inputs with specific
#     options)"""

#     def convert_before_sav(test_dir):
#         """Takes the before .sav file out of FHD function `vis_extract_autocorr`
#         and converts into an hdf5 format"""

#         sav_dict = convert_sav_to_dict(str(Path(test_dir, "before_vis_extract_autocorr.sav")), "meh")

#         ##convert obs from a recarray into a dictionary
#         obs = recarray_to_dict(sav_dict['obs'])
        
#         ##fix the shape of vis_arr
#         ##TODO might need to swap axis 1 and 2 here depending on what happens
#         ##inside vis_extract_autocorr
#         ##Currently I'm reading into a shape of (n_pol, n_freq, n_baselines)
#         vis_arr = np.empty((obs["n_pol"], sav_dict['vis_arr'][0].shape[1],
#                                           sav_dict['vis_arr'][0].shape[0]),
#                                           dtype=complex)
#         for pol in range(obs["n_pol"]):
#             vis_arr[pol, :, :] = sav_dict['vis_arr'][pol].transpose()
        
#         ##super dictionary to save everything in
#         h5_save_dict = {}
#         h5_save_dict['obs'] = obs
#         h5_save_dict['vis_arr'] = vis_arr
#         h5_save_dict['time_average'] = sav_dict['time_average']

#         ##save the thing 
#         dd.io.save(Path(test_dir, "before_vis_extract_autocorr.h5"), h5_save_dict)

#     def convert_after_sav(test_dir):
#         """Takes the after .sav file out of FHD function `vis_extract_autocorr`
#         and converts into an hdf5 format"""

#         sav_dict = convert_sav_to_dict(str(Path(test_dir, "after_vis_extract_autocorr.sav")), "meh")

#         ##super dictionary
#         h5_save_dict = {}
#         h5_save_dict['auto_corr'] = sav_dict['auto_corr']
#         h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']

#         ##save the thing 
#         dd.io.save(Path(test_dir, "after_vis_extract_autocorr.h5"), h5_save_dict)
        
#     def convert_sav(test_dir):
#         """Load the inputs and outputs needed for testing `vis_extract_autocorr`"""
#         convert_before_sav(test_dir)
#         convert_after_sav(test_dir)

#     ##Start doing the conversion from IDL to Python

#     ##Where be all of our data
#     base_dir = Path(env.get('PYFHD_TEST_PATH'))

#     ##Each test_set contains a run with a different set of inputs/options
#     for test_set in ['pointsource1_vary1', 'pointsource1_standard']:
#         convert_sav(Path(base_dir, test_set))
#         # run_test(Path(base_dir, test_set))