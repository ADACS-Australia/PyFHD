from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from logging import Logger
from PyFHD.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from PyFHD.healpix.healpix_utils import healpix_cnv_generate

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "healpix_cnv_generate")

@pytest.fixture(scope="function", params=['1088285600','1088716296', 'point_zenith', 'point_offzenith'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run3', 'run4'])
def run(request):
    return request.param

skip_tests = [['1088285600', 'run4'], ["point_zenith", "run3"], ["point_offzenith", "run3"]]

@pytest.fixture
def before_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix('.sav')
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")
    sav_dict = recarray_to_dict(sav_dict)

    sav_dict['obs']['dimension'] = int(sav_dict['obs']['dimension'])
    sav_dict['obs']['elements'] = int(sav_dict['obs']['elements'])

    sav_dict["pyfhd_config"] = {
        "restrict_healpix_inds" : sav_dict["restrict_hpx_inds"],
        "healpix_inds": None
    }

    save(before_file, sav_dict, "sav_dict")

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
    sav_dict = recarray_to_dict(sav_dict)
    # sav_dict['hpx_cnv']['ija'] = np.array(sav_dict['hpx_cnv']['ija'], dtype = object)
    largest_arr_size = -1
    for arr in sav_dict['hpx_cnv']['ija']:
        if np.size(arr) > largest_arr_size:
            largest_arr_size = np.size(arr)
    # Create an array of the shape ija and largest_arr_size filling in the values from
    # the original ija array and filling the rest with -1, as you can't use NaN with ints
    ija = np.full((len(sav_dict['hpx_cnv']['ija']), largest_arr_size), -1, dtype = np.int64)
    for i, arr in enumerate(sav_dict['hpx_cnv']['ija']):
        ija[i, :np.size(arr)] = arr
    sav_dict['hpx_cnv']['ija'] = ija
    # Do the same with sa, but we can use NaNs as it's a float array
    largest_arr_size = -1
    for arr in sav_dict['hpx_cnv']['sa']:
        if np.size(arr) > largest_arr_size:
            largest_arr_size = np.size(arr)
    sa = np.full((len(sav_dict['hpx_cnv']['sa']), largest_arr_size), np.nan, dtype = np.float64)
    for i, arr in enumerate(sav_dict['hpx_cnv']['sa']):
        sa[i, :np.size(arr)] = arr
    sav_dict['hpx_cnv']['sa'] = sa
    sav_dict['hpx_cnv']['i_use'] = sav_dict['hpx_cnv']['i_use'].astype(np.int64)

    save(after_file, sav_dict['hpx_cnv'], "hpx_cnv")
    
    return after_file

def test_healpix_cnv_generate(before_file, after_file):
    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}")
    
    h5_before = load(before_file)
    expected_hpx_cnv = load(after_file)

    # Astropy strictly expects the ctype to be a string, not a numpy string
    h5_before["obs"]["astr"]["ctype"] = [str(x) for x in h5_before["obs"]["astr"]["ctype"]]

    hpx_cnv, obs = healpix_cnv_generate(
        h5_before["obs"], 
        h5_before["mask"], 
        h5_before["hpx_radius"],
        h5_before["pyfhd_config"],
        Logger(1),
        nside = int(h5_before["nside"]) if "nside" in h5_before else None
    )

    for key in hpx_cnv:
        npt.assert_allclose(hpx_cnv[key], expected_hpx_cnv[key], atol = 1e-8)

    npt.assert_equal(obs['healpix']['nside'], hpx_cnv['nside'])
    # restrict_healpix_inds is always True for the tests so ind_list is false
    assert obs['healpix']['ind_list'] is None
    npt.assert_equal(obs['healpix']['n_pix'], np.size(hpx_cnv["inds"]))