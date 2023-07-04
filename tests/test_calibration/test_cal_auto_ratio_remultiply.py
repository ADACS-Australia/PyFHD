import pytest
from os import environ as env
from pathlib import Path
from PyFHD.calibration.calibration_utils import cal_auto_ratio_remultiply
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import deepdish as dd
import numpy.testing as npt
from copy import deepcopy

@pytest.fixture()
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "cal_auto_ratio_remultiply")

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"]]

# For each combination of tag and run, check if the hdf5 file exists, if not, create it and either way return the path
# Tests will fail if the fixture fails, not too worried about exceptions here.

@pytest.fixture()
def before_file(tag, run, data_dir):
    if ([tag, run] in skip_tests):
        return None
    before_file = Path(data_dir, f"{tag}_{run}_before_{data_dir.name}.h5")
    # If the h5 file already exists and has been created, return the path to it
    if before_file.exists():
        return before_file
    
    sav_file = before_file.with_suffix('.sav')

    sav_dict = convert_sav_to_dict(str(sav_file), "faked")

    cal = recarray_to_dict(sav_dict['cal'])
            
    ##super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['cal'] = cal
    h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
    h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']
    h5_save_dict['auto_ratio'] = sav_file_vis_arr_swap_axes(sav_dict['auto_ratio'])

    dd.io.save(before_file, h5_save_dict)

    return before_file

# Same as the before_file fixture, except we're taking the the after files
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

    ##super dictionary to save everything in
    h5_save_dict = {}
    
    h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])
    h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_cal_auto_ratio_remultiply(before_file, after_file):
    """
    Runs the test on `cal_auto_ratio_remultiply` reads in the data in before_file and after_file,
    and then calls `cal_auto_ratio_remultiply`, checking the outputs match expectations
    """
    if (before_file == None or after_file == None):
        pytest.skip(f"""This test has been skipped because the test was listed in the 
                    skipped tests due to FHD not outpoutting them: {skip_tests}""")

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    cal = h5_before['cal']
    auto_tile_i = h5_before['auto_tile_i']
    auto_ratio = h5_before['auto_ratio']

    expected_cal = h5_after['cal']

    result_cal = cal_auto_ratio_remultiply(cal, auto_ratio, auto_tile_i)

    atol = 1e-10

    ##check the gains have been updated
    npt.assert_allclose(expected_cal['gain'], result_cal['gain'], atol=atol)


# def test_pointsource1_vary1(data_dir):
#     """Test using the `pointsource1_vary1` set of inputs"""

#     run_test(data_dir, "pointsource1_vary1")

# def test_pointsource2_vary1(data_dir):
#     """Test using the `pointsource2_vary1` set of inputs"""

#     run_test(data_dir, "pointsource2_vary1")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `cal_auto_ratio_remultiply`
        and converts into an hdf5 format"""

        func_name = 'cal_auto_ratio_remultiply'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        cal = recarray_to_dict(sav_dict['cal'])
            
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['cal'] = cal
        h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']
        h5_save_dict['auto_ratio'] = sav_file_vis_arr_swap_axes(sav_dict['auto_ratio'])

        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `cal_auto_ratio_remultiply`
        and converts into an hdf5 format"""

        func_name = 'cal_auto_ratio_remultiply'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal'] = recarray_to_dict(sav_dict['cal'])
        h5_save_dict['cal']['gain'] = sav_file_vis_arr_swap_axes(h5_save_dict['cal']['gain'])
        

        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `cal_auto_ratio_remultiply`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'cal_auto_ratio_remultiply')

    tag_names = ['pointsource2_vary1', 'pointsource1_vary1']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)