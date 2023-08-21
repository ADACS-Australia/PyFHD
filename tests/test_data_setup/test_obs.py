import pytest
from logging import RootLogger
from pathlib import Path
from os import environ as env
from PyFHD.data_setup.uvfits import extract_header, create_params, create_layout
from PyFHD.data_setup.obs import create_obs
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict
import deepdish as dd
import numpy.testing as npt
import numpy as np
from scipy.io import readsav

@pytest.fixture(
    scope="function", 
    params=[
        '1088716296', 
        'point_offzenith_8s_80kHz_analy_autos+gain_errors',
        'point_zenith_8s_80kHz_analy_autos+gain_errors'
    ]
)
def obs_id(request):
    # Ensure you have created the symbolic links for the point_offzenith and point_zenith
    # metafits files from 1088716176.metafits
    return request.param

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "uvfits")

@pytest.fixture
def obs_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "fhd_struct_init_obs")

def check_sav_file(path: Path, pyfhd_config: dict) -> Path:
    """
    Checks if the h5 file for obs testing exists, if it does return it's path
    if it does not exist, create it fro the sav file, then then return the path.

    Parameters
    ----------
    path : Path
        Path to the obs testing directory containing the reuslting FHD obs structures
    pyfhd_config : dict
        A mock example of PyFHD's configuration given to the test

    Returns
    -------
    h5_file: Path
        The path to the h5_file
    """
    if ('_' in pyfhd_config['obs_id']):
        obs_id_split = pyfhd_config['obs_id'].split('_')
        tag = f"{obs_id_split[0]}_{obs_id_split[1]}"
    else:
        tag = pyfhd_config['obs_id']
    h5_file = Path(path, f"{tag}_run1_after_{path.name}.h5")
    if h5_file.exists():
        return h5_file
    sav_file = h5_file.with_suffix('.sav')
    sav_dict = readsav(sav_file, python_dict=True)
    obs = recarray_to_dict(sav_dict['obs'])
    h5_save_dict = {}
    h5_save_dict['obs'] = obs
    dd.io.save(h5_file, h5_save_dict)

    return h5_file


def test_obs_creation(obs_id, data_dir, obs_dir):
    # The obs creation test is more of an integration test, since we will be
    # using the extract_header, create_params, and create_layout to create the obs dictionary. 
    # If this test pass then it essentially means that the dictionaries are almost identical 
    # to that of the IDL structures in the ways that matter for a PyFHD run.
    # In this case we're only going to test the obs structure from run1 of each test.
    logger = RootLogger(1)
    pyfhd_config = {
        'obs_id': obs_id,
        'input_path': data_dir,
        'n_pol': 2,
        'instrument': 'mwa',
        'FoV': None,
        'dimension': 2048,
        'elements' : 2048,
        'kbinsize': 0.5,
        'min_baseline': 1,
        'time_cut': None,
        'beam_nfreq_avg' : 16,
        'dft_threshold' : False,
        'restrict_hpx_inds' : True,
    }
    pyfhd_header, params_data, antenna_header, antenna_data = extract_header(pyfhd_config, logger)
    params = create_params(pyfhd_header, params_data, logger)
    layout = create_layout(antenna_header, antenna_data, logger)
    obs = create_obs(pyfhd_header, params, layout, pyfhd_config, logger)
    obs_fhd_result_path = check_sav_file(obs_dir, pyfhd_config)
    obs_fhd = dd.io.load(obs_fhd_result_path)['obs']

    # Check the basic obs info
    assert(obs['n_pol'] == obs_fhd['n_pol'])
    assert(obs['n_tile'] == obs_fhd['n_tile'])
    assert(obs['n_freq'] == obs_fhd['n_freq'])
    assert(obs['n_time'] == obs_fhd['n_time'])
    assert(obs['kpix'] == obs_fhd['kpix'])
    assert(obs['dimension'] == obs_fhd['dimension'])
    assert(obs['elements'] == obs_fhd['elements'])
    npt.assert_almost_equal(obs['degpix'], obs_fhd['degpix'])
    npt.assert_almost_equal(obs['max_baseline'], obs_fhd['max_baseline'])
    npt.assert_almost_equal(obs['min_baseline'], obs_fhd['min_baseline'])
    npt.assert_array_equal(obs['pol_names'], obs_fhd['pol_names'].astype('str'))

    # Check baseline_info
    npt.assert_array_equal(obs['baseline_info']['time_use'], obs_fhd['baseline_info']['time_use'])
    assert(obs['n_time_flag'] == obs_fhd['n_time_flag'])
    npt.assert_array_equal(obs['baseline_info']['tile_use'], obs_fhd['baseline_info']['tile_use'])
    # tile_flag is a little weird given it wants pointers from tile_flag
    # The indexes provided to tile_flag also go beyond the index range of the metadata
    # is this a bug in FHD? Thankfully it's not used elsewhere, and I can get the same behavior
    # with two polarizations, but I suspect there could be a difference with 4 polarizations
    # in the tile_use?
    # npt.assert_array_equal(obs['baseline_info']['tile_flag'], obs_fhd['baseline_info']['tile_flag'])
    assert(obs['n_tile_flag'] == obs_fhd['n_tile_flag'])
    npt.assert_array_equal(obs['baseline_info']['freq_use'], obs_fhd['baseline_info']['freq_use'])
    assert(obs['dft_threshold'] == obs_fhd['dft_threshold'])
    npt.assert_array_equal(obs['baseline_info']['tile_a'], obs_fhd['baseline_info']['tile_a'])
    npt.assert_array_equal(obs['baseline_info']['tile_b'], obs_fhd['baseline_info']['tile_b'])
    npt.assert_array_equal(obs['baseline_info']['tile_names'], np.char.strip((obs_fhd['baseline_info']['tile_names'].astype('str'))).astype(int))
   
    # Check healpix
    assert(obs['healpix']['nside'] == obs_fhd['healpix']['nside'])
    assert(obs['healpix']['n_pix'] == obs_fhd['healpix']['n_pix'])
    assert(obs['healpix']['ind_list'] == int(obs_fhd['healpix']['ind_list']))
    assert(obs['healpix']['n_zero'] == obs_fhd['healpix']['n_zero'])


