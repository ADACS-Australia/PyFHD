import pytest
import numpy as np
import numpy.testing as npt
from os import environ as env
from pathlib import Path
import logging
from PyFHD.data_setup.uvfits import extract_header, extract_visibilities
from PyFHD.pyfhd_tools.test_utils import get_savs

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'uvfits_read/')

def test_1061316296(data_dir):
    dummy_log = logging.getLogger('dummy')
    pyfhd_config = np.load(Path(data_dir, 'config.npy'), allow_pickle=True)
    pyfhd_header, fits_data = extract_header(pyfhd_config, dummy_log)
    vis_arr, vis_weights = extract_visibilities(pyfhd_header, fits_data, pyfhd_config, dummy_log)

    output = get_savs(data_dir,'output.sav')

    npt.assert_allclose(vis_arr, output['vis_arr'], atol = 1e-8)
    npt.assert_allclose(vis_weights, output['vis_weights'], atol = 1e-8)
