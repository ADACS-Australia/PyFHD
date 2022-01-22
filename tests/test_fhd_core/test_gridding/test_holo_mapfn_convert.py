import pytest
import numpy as np
from glob import glob
from fhd_core.gridding.holo_mapfn_convert import holo_mapfn_convert
from tests.test_utils import get_data_items

@pytest.fixture
def data_dir():
    return glob("**/holo_mapfn_convert", recursive = True)[0]

def not_yet_test_holo_mapfn_one(data_dir):
    map_fn, psf_dim, n_vis, dimension, expected_map_fn  = get_data_items(data_dir,
                                                                         'visibility_grid_input_mapfn.npy',
                                                                         'visibility_grid_input_psf_dim.npy',
                                                                         'visibility_grid_input_n_vis.npy',
                                                                         'visibility_grid_input_dimension.npy',
                                                                         'visibility_grid_output_mapfn.npy')
    print(np.where(map_fn))
    output_map_fn = holo_mapfn_convert(map_fn, psf_dim, int(dimension), norm = n_vis)
    assert np.array_equal(output_map_fn['ija'], expected_map_fn['ija'][0])
