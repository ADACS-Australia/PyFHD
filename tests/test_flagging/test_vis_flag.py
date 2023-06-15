from PyFHD.flagging.flagging import vis_flag
import numpy.testing as npt
import pytest
from pathlib import Path
from os import environ as env
import deepdish as dd

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'vis_flag')

def test_pointsource1_vary1(data_dir):
    # h5_before = dd.io.load(Path(data_dir, "pointsource1_vary1_before_vis_flag.h5"))
    # h5_after = dd.io.load(Path(data_dir, "pointsource1_vary1_after_vis_flag.h5"))
    pass

def test_pointsource1_standard(data_dir):
    # h5_before = dd.io.load(Path(data_dir, "pointsource1_standard_before_vis_flag.h5"))
    # h5_after = dd.io.load(Path(data_dir, "pointsource1_standard_after_vis_flag.h5"))
    pass