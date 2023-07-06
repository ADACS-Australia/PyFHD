import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.pyfhd_tools.pyfhd_utils import vis_noise_calc
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
from logging import RootLogger
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_noise_calc")

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run3'])
def run(request):
    return request.param

skip_tests = []

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

    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
    h5_save_dict['vis_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
    h5_save_dict['vis_weights'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weights'])
    # print('vis-weights shape', h5_save_dict['vis_weights'].shape)

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

    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['noise_arr'] = sav_dict['noise_arr']

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_points_zenith_and_offzenith(before_file, after_file):
    """Runs the test on `vis_noise_calc` - reads in the data in `data_loc`,
    and then calls `vis_noise_calc`, checking the outputs match expectations"""

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    obs = h5_before['obs']
    vis_arr = h5_before['vis_arr']
    vis_weights = h5_before['vis_weights']

    expected_noise_arr = h5_after['noise_arr'].transpose()

    result_noise_arr = vis_noise_calc(obs, vis_arr, vis_weights)

    npt.assert_allclose(expected_noise_arr, result_noise_arr, atol=1e-8)