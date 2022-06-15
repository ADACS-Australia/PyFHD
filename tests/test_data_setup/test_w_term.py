import pytest
import numpy as np
import numpy.testing as npt
from os import environ as env
from pathlib import Path
import logging
from PyFHD.pyfhd_tools.pyfhd_utils import simple_deproject_w_term
from PyFHD.pyfhd_tools.test_utils import get_savs

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'simple_deproject_w_term/')

def test_w_term_1(data_dir):
    dummy_log = logging.getLogger('dummy')
    inputs = get_savs(data_dir,'input_1.sav')
    # Adjust baseline_info so its the same as accessing a python dictionary
    inputs['obs'] = inputs['obs'][0]
    inputs['obs']['baseline_info'] = inputs['obs']['baseline_info'][0]
    # Create new params dictionary with just ww for testing.
    params = {'ww' : inputs['params']['ww'][0]}
    vis_arr = np.zeros((inputs['vis_arr'][0].shape[0],inputs['vis_arr'][0].shape[1], 2), dtype = np.complex64)
    # Change the visibility array to make it inline with the PyFHD visibility array shape
    for i in range(inputs['obs']['n_pol']):
        print(inputs['vis_arr'][i].shape)
        vis_arr[:, :, i] = inputs['vis_arr'][i]
    vis_arr = simple_deproject_w_term(inputs['obs'], params, vis_arr, inputs['direction'], dummy_log)
    expected_output = get_savs(data_dir,'output_1.sav')
    for pol_i in range(inputs['obs']['n_pol']):
        npt.assert_allclose(vis_arr[:, :, pol_i], expected_output['vis_arr'][pol_i])

def test_w_term_2(data_dir):
    dummy_log = logging.getLogger('dummy')
    inputs = get_savs(data_dir,'input_2.sav')
    vis_arr = simple_deproject_w_term(inputs['obs'], inputs['params'], inputs['vis_arr'], inputs['direction'], dummy_log)
    expected_output = get_savs(data_dir,'output_2.sav')
    for pol_i in range(inputs['obs']['n_pol']):
        npt.assert_allclose(vis_arr[:, :, pol_i], expected_output['vis_arr'][pol_i])