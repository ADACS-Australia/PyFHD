import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.pyfhd_tools.pyfhd_utils import split_vis_weights
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
from logging import RootLogger
import numpy.testing as npt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "split_vis_weights")

def run_test(data_dir, tag_name):
    """Runs the test on `split_vis_weights` - reads in the data in `data_loc`,
    and then calls `split_vis_weights`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_split_vis_weights.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_split_vis_weights.h5"))

    obs = h5_before['obs']
    vis_weights = h5_before['vis_weights']

    expected_vis_weights = h5_after['vis_weights_use']
    expected_bi_use = h5_after['bi_use']

    result_vis_weights, result_bi_use = split_vis_weights(obs, vis_weights)

    atol = 1e-8

    ##Check the returned weights have been spli correctly
    npt.assert_allclose(expected_vis_weights, result_vis_weights,
                        atol=atol)

    ##Annoying shape mis-match, so test each polarisation result
    ##invidually
    for pol in range(obs['n_pol']):
        npt.assert_allclose(expected_bi_use[pol], result_bi_use[pol], atol=atol)


def test_pointsource1_standard(data_dir):
    """Test using the `pointsource1_standard` set of inputs"""

    run_test(data_dir, "pointsource1_standard")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `split_vis_weights`
        and converts into an hdf5 format"""

        func_name = 'split_vis_weights'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = recarray_to_dict(sav_dict['obs'])
        h5_save_dict['vis_weights'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weights'])

        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `split_vis_weights`
        and converts into an hdf5 format"""

        func_name = 'split_vis_weights'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['vis_weights_use'] = sav_file_vis_arr_swap_axes(sav_dict['vis_weights_use'])
        h5_save_dict['bi_use'] = sav_dict['bi_use']

        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `split_vis_weights`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'split_vis_weights')

    tag_names = ['pointsource1_standard']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)
        