from PyFHD.io.pyfhd_io import recarray_to_dict
import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.flagging.flagging import vis_flag
from PyFHD.io.pyfhd_io import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import sav_file_vis_arr_swap_axes
import numpy as np
from PyFHD.io.pyfhd_io import save, load
from logging import Logger
import numpy.testing as npt


@pytest.fixture
def data_dir():
    return Path(env.get("PYFHD_TEST_PATH"), "vis_flag")


@pytest.fixture(scope="function", params=["point_zenith", "point_offzenith"])
def tag(request):
    return request.param


@pytest.fixture(scope="function", params=["run3"])
def run(request):
    return request.param


skip_tests = []


@pytest.fixture()
def before_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file

    sav_file = before_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    # super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])
    h5_save_dict["params"] = recarray_to_dict(sav_dict["params"])
    h5_save_dict["vis_arr"] = sav_file_vis_arr_swap_axes(sav_dict["vis_arr"])
    h5_save_dict["vis_weight_ptr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_weight_ptr"]
    )

    save(before_file, h5_save_dict, "before_file")

    return before_file


@pytest.fixture()
def after_file(tag, run, data_dir):
    if [tag, run] in skip_tests:
        return None
    after_file = Path(data_dir, f"{tag}_{run}_after_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if after_file.exists():
        return after_file

    sav_file = after_file.with_suffix(".sav")
    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    h5_save_dict = {}
    h5_save_dict["vis_weight_ptr"] = sav_file_vis_arr_swap_axes(
        sav_dict["vis_weight_ptr"]
    )
    h5_save_dict["obs"] = recarray_to_dict(sav_dict["obs"])
    save(after_file, h5_save_dict, "after_file")

    return after_file


def test_zenith_offzenith(before_file, after_file):
    """Runs the test on `vis_flag` - reads in the data in `data_loc`,
    and then calls `vis_flag`, checking the outputs match expectations"""

    h5_before = load(before_file)
    h5_after = load(after_file)

    obs = h5_before["obs"]
    orig_n_vis = obs["n_vis"]
    params = h5_before["params"]
    vis_arr = h5_before["vis_arr"]
    vis_weight_ptr = h5_before["vis_weight_ptr"]

    expected_obs = h5_after["obs"]
    expected_vis_weight_ptr = h5_after["vis_weight_ptr"]

    logger = Logger(1)

    result_vis_weights, result_obs = vis_flag(
        vis_arr, vis_weight_ptr, obs, params, logger
    )

    # Weights should be setup
    npt.assert_allclose(result_vis_weights, expected_vis_weight_ptr, atol=1e-8)
    # Should end up with the right number of visis after cutting
    npt.assert_equal(result_obs["n_vis"], expected_obs["n_vis"])
    npt.assert_array_equal(
        result_obs["baseline_info"]["tile_use"],
        expected_obs["baseline_info"]["tile_use"],
    )
    npt.assert_array_equal(
        result_obs["baseline_info"]["freq_use"],
        expected_obs["baseline_info"]["freq_use"],
    )
