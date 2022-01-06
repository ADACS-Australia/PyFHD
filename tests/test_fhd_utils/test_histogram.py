import pytest
import numpy as np
from fhd_utils.histogram import histogram
from pathlib import Path
from tests.test_utils import get_data, get_data_items

@pytest.fixture
def data_dir():
    # This assumes you have used the splitter.py and have done a general format of **/FHD/PyFHD/tests/test_fhd_*/data/<function_name_being_tested>/*.npy
    return list(Path.glob(Path.cwd(), '**/histogram/'))[0]

def test_idl_example(data_dir) :
    """
    This test is based on the example from the IDL documentation.
    This ensures we get the same behaviour as an example everybody can see.
    """
    # Setup the test from the histogram data file
    data, expected_hist, expected_indices = get_data(data_dir, 'idl_hist_example.npy', 'idl_example_hist.npy', 'idl_example_inds.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_hundred_nums(data_dir):
    """
    This is a basic test of an array with numbers 0 to 99 in increasing
    order.
    Should produce two bins.
    """
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'hundred_ints.npy', 'hundred_ints_hist_bin50.npy', 'hundred_ints_inds_bin50.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size = 50)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_hundred_ten_bins(data_dir):
     # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'hundred_ints.npy', 'hundred_ints_hist_nbin10.npy', 'hundred_ints_inds_nbin10.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    # This is to show that bin_size is ignored when num_bins is used
    hist, _, indices = histogram(data, num_bins = 10, bin_size=1000)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_min(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'hundred_ints.npy', 'hundred_ints_hist_min10.npy', 'hundred_ints_inds_min10.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size = 10, min = 10)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_max(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'hundred_ints.npy', 'hundred_ints_hist_max50.npy', 'hundred_ints_inds_max50.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size = 10, max = 50)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_min_max(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'hundred_ints.npy', 'hundred_ints_hist_min10_max55.npy', 'hundred_ints_inds_min10_max55.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, bin_size = 10, min = 10, max = 55)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_one_max(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'hundred_ints.npy', 'hundred_ints_hist_binsize1_max55.npy', 'hundred_ints_inds_binsize1_max55.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data, max = 55)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'normals.npy', 'normals_hist.npy', 'normals_inds.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals_binsize(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'normals.npy', 'normals_hist_binsize025.npy', 'normals_inds_binsize025.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, bin_size = 0.25)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals_min_max(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'normals.npy', 'normals_hist_min_max.npy', 'normals_inds_binsize_min_max.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, min = 0, max = 1, bin_size = 0.25)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_normals_times_10(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'normals.npy', 'normals_hist_times10.npy', 'normals_inds_times10.npy')
    data = data * 10
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, bin_size = 2)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_billion(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'one_billion.npy', 'one_billion_hist.npy', 'one_billion_inds.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(int)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_billion_floats(data_dir):
    # Read the histogram file
    data, expected_hist, expected_indices = get_data(data_dir, 'one_billion_floats.npy', 'one_billion_floats_hist.npy', 'one_billion_floats_inds.npy')
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)

def test_full_size(data_dir):
    # Read the histogram file
    data, binsize, min, expected_hist, expected_indices = get_data_items(
        Path(data_dir, 'full_size'),
        'input_1.npy',
        'binsize_1.npy', 
        'min_1.npy', 
        'output_1.npy', 
        'reverse_indices_1.npy'
    )
    # Now that we're using numba it doesn't support every type, set it to more standard NumPy or Python types
    data = data.astype(float)
    hist, _, indices = histogram(data, bin_size = binsize, min = min)
    assert np.array_equal(hist, expected_hist)
    assert np.array_equal(indices, expected_indices)