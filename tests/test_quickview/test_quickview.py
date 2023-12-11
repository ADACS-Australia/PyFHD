from os import environ as env
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from logging import Logger
import astropy
from PyFHD.io.pyfhd_io import convert_sav_to_dict, load, recarray_to_dict, save
from PyFHD.io.pyfhd_quickview import quickview

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "fhd_quickview")

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
    del sav_dict['cal']
    del sav_dict['skymodel']
    del sav_dict['jones']
    del sav_dict['file_path_fhd']
    del sav_dict['ring_radius']
    sav_dict = recarray_to_dict(sav_dict)

    # Set the save everything except the fits we're testing to false
    pyfhd_config = {
    "obs_id": '1088716176' if 'point' in before_file.name else before_file.name.split('_')[0],
        "output_dir": Path(data_dir, f'{tag}_{run}_test_data'),
        "save_obs" : False,
        "save_params": False,
        "save_visibilities": False,
        "save_cal": False,
        "save_calibrated_weights": False,
        "pad_uv_image": sav_dict["pad_uv_image"],
        "image_filter": "filter_uv_uniform"
    }

    sav_dict["pyfhd_config"] = pyfhd_config
    sav_dict["model_uv"] = sav_dict["extra"]["model_uv_holo"]

    save(before_file, sav_dict, "sav_dict")

    return before_file

def test_quickview(before_file, data_dir):
    if (before_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outputting them: {skip_tests}")

    h5_before = load(before_file)
    psf = load(Path(env.get('PYFHD_TEST_PATH'), "beams", "decomp_beam_pointing0.h5"))

    # Make our test directory for saving the FITS files
    h5_before["pyfhd_config"]["output_dir"] = Path(h5_before["pyfhd_config"]["output_dir"])
    h5_before["pyfhd_config"]["output_dir"].mkdir(exist_ok = True)

    # Astropy needs things to be a str Python type, numpy str type no longer works?
    h5_before["obs"]["astr"]["ctype"] = [str(x) for x in h5_before["obs"]["astr"]["ctype"]]

    # Since we're not saving anything other than the fits we can ignore everything
    # else but the things required to make the FITS files. In this case we're lazy loading
    # beam file to save on memory.
    quickview(
        h5_before["obs"], 
        psf, 
        None, 
        None, 
        None, 
        None, 
        h5_before["image_uv_arr"], 
        h5_before["weights_arr"],
        None,
        None,
        h5_before['model_uv'],
        h5_before["pyfhd_config"],
        Logger(1)
    )
    # Close the HDF5 file
    psf.close()
    # Check all the FITS Files
    fits_path = h5_before["pyfhd_config"]["output_dir"] / "fits"
    files = fits_path.glob("*.fits")
    expected_files = Path(data_dir, 'fhd' + 'before_file'.name.split('_')[:-3].join('_'))
    for fits in files:
        fits_diff = astropy.io.fits.FITSDiff(fits, Path(expected_files, fits.name))
        print(fits_diff)
    # Clean up the FITS files directory after the test