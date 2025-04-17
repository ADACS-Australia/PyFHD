import pytest
import numpy as np
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items
from PyFHD.pyfhd_tools.pyfhd_utils import meshgrid


@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return Path(env.get("PYFHD_TEST_PATH"), "meshgrid/")


def test_meshgrid_one(data_dir):
    axis, dimension, elements, integer, expected = get_data_items(
        data_dir,
        "input_axis_1.npy",
        "input_dimension_1.npy",
        "input_elements_1.npy",
        "input_integer_1.npy",
        "output_result_1.npy",
    )
    result = meshgrid(dimension, elements, axis=axis, return_integer=integer)
    assert np.array_equal(result, expected)


def test_meshgrid_two(data_dir):
    dimension, elements, expected = get_data_items(
        data_dir, "input_dimension_2.npy", "input_elements_2.npy", "output_result_2.npy"
    )
    result = meshgrid(dimension, elements)
    assert np.array_equal(result, expected)


def test_meshgrid_three(data_dir):
    axis, dimension, elements, expected = get_data_items(
        data_dir,
        "input_axis_3.npy",
        "input_dimension_3.npy",
        "input_elements_3.npy",
        "output_result_3.npy",
    )
    result = meshgrid(dimension, elements, axis=axis)
    assert np.array_equal(result, expected)
