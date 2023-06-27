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

def run_test(data_dir, tag_name):
    """Runs the test on `vis_noise_calc` - reads in the data in `data_loc`,
    and then calls `vis_noise_calc`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_vis_noise_calc.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_vis_noise_calc.h5"))

    obs = h5_before['obs']
    vis_arr = h5_before['vis_arr']
    vis_weights = h5_before['vis_weights']

    expected_noise_arr = h5_after['noise_arr'].transpose()

    result_noise_arr = vis_noise_calc(obs, vis_arr, vis_weights)

    atol = 1e-8

    ##how do noise, you good?
    npt.assert_allclose(expected_noise_arr, result_noise_arr, atol=atol)


def test_pointsource1_standard(data_dir):
    """Test using the `pointsource1_standard` set of inputs"""

    run_test(data_dir, "pointsource1_standard")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_noise_calc`
        and converts into an hdf5 format"""

        func_name = 'vis_noise_calc'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
        h5_save_dict['vis_arr'] = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
        h5_save_dict['vis_weights'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weights'])
        print('vis-weights shape', h5_save_dict['vis_weights'].shape)

        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_noise_calc`
        and converts into an hdf5 format"""

        func_name = 'vis_noise_calc'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['noise_arr'] = sav_dict['noise_arr']

        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_noise_calc`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_noise_calc')

    tag_names = ['pointsource1_standard']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)