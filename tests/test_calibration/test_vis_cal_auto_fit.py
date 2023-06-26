import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_cal_auto_fit
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
import numpy.testing as npt
import matplotlib.pyplot as plt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_fit")

# @pytest.fixture
# def base_dir():
#     return Path(env.get('PYFHD_TEST_PATH'))

def run_test(data_dir, tag_name):
    """Runs the test on `vis_cal_bandpass` - reads in the data in `data_loc`,
    and then calls `vis_cal_bandpass`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_vis_cal_auto_fit.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_vis_cal_auto_fit.h5"))

    obs = h5_before['obs']
    cal = h5_before['cal']
    vis_auto = h5_before['vis_auto']
    vis_model_auto = h5_before['vis_model_auto']
    auto_tile_i = h5_before['auto_tile_i']

    expected_cal_fit = h5_after['cal_fit']

    return_cal_fit = vis_cal_auto_fit(obs, cal, vis_auto, vis_model_auto, auto_tile_i)
    # auto_scale is nan, nan from FHD and Python, can't compare them due to the nans
    # assert np.array_equal(return_cal_fit['auto_scale'], expected_cal_fit['auto_scale'])
    # cal_fit['auto_params'] came in as an object array
    auto_params = np.empty([2, cal["n_pol"], cal['n_tile']], dtype = np.float64)
    auto_params[0] = expected_cal_fit['auto_params'][0].transpose()
    auto_params[1] = expected_cal_fit['auto_params'][1].transpose()

    ##TODO get this stored somewhere as a test input
    actual_gains = np.load('gains_applied_woden.npz')
    gx = actual_gains['gx'].transpose()
    gy = actual_gains['gy'].transpose()

    # print(gx.shape)

    # print(expected_cal_fit['gain'][0, :, :].shape)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(np.abs(gx[0, :]), 's', mfc='none', linestyle='none', label='Sim gains')
    axs[0].plot(return_cal_fit['gain'][0, 0, :], 'x', mfc='none', linestyle='none', label='Fit PyFHD')

    print(expected_cal_fit['gain'][0, 0, 1])
    print(return_cal_fit['gain'][0, 0, 1])

    axs[1].plot(np.abs(gx[0, :]), 's', mfc='none', linestyle='none', label='Sim gains')
    axs[1].plot(expected_cal_fit['gain'][0, 0, :]*0.5, '^', mfc='none', linestyle='none', label='Fit FHD')

    axs[1].set_xlabel('Tile index')

    axs[0].set_ylabel('Gain value')
    axs[1].set_ylabel('Gain value')

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    fig.savefig('test_vis_cal_auto_fit.png', bbox_inches='tight', dpi=300)
    plt.close()

    rtol = 1e-5
    atol = 1e-3

    # npt.assert_allclose(return_cal_fit['auto_params'], auto_params,
    #                     rtol=rtol, atol=atol)

    # npt.assert_allclose(return_cal_fit['gain'], )

# def test_pointsource1_vary(data_dir):
#     """Test using the `pointsource1_vary1` set of inputs"""

#     run_test(data_dir, "pointsource1_vary1")

def test_pointsource2_vary1(data_dir):
    """Test using the `pointsource2_vary1` set of inputs"""

    run_test(data_dir, "pointsource2_vary1")
    
if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_cal_auto_fit`
        and converts into an hdf5 format"""

        # path = Path(data_dir, f"{tag_name}_before_vis_cal_auto_fit.sav")
        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_vis_cal_auto_fit.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        cal = recarray_to_dict(sav_dict['cal'])

        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal['gain'] = sav_file_vis_arr_swap_axes(cal['gain'])

        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['cal'] = cal
        h5_save_dict['vis_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_auto'])
        h5_save_dict['vis_model_auto'] = sav_file_vis_arr_swap_axes(sav_dict['vis_model_auto'])
        h5_save_dict['auto_tile_i'] = sav_dict['auto_tile_i']
        
        dd.io.save(Path(data_dir, f"{tag_name}_before_vis_cal_auto_fit.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_cal_auto_fit`
        and converts into an hdf5 format"""

        # path = Path(data_dir, f"{data_dir}/{tag_name}_after_vis_cal_auto_fit.sav")
        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_vis_cal_auto_fit.sav", "meh")
        
        cal_fit = recarray_to_dict(sav_dict['cal_fit'])
        
        ##Swap the freq and tile dimensions
        ##this make shape (n_pol, n_freq, n_tile)
        cal_fit['gain'] = sav_file_vis_arr_swap_axes(cal_fit['gain'])
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['cal_fit'] = cal_fit
        dd.io.save(Path(data_dir, f"{tag_name}_after_vis_cal_auto_fit.h5"), h5_save_dict)
        
    def convert_sav(data_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_cal_auto_fit`"""
        convert_before_sav(data_dir, tag_name)
        convert_after_sav(data_dir, tag_name)

    ##Where be all of our data
    data_dir = Path(env.get('PYFHD_TEST_PATH'), "vis_cal_auto_fit")

    print("DATA DIR IS", data_dir)

    ##Each test_set contains a run with a different set of inputs/options
    ##TODO get the tag_names from some kind of glob on the relevant dir
    tag_names = ['pointsource2_vary1']

    for tag_name in tag_names:
        convert_sav(data_dir, tag_name)
        # run_test(data_dir, tag_name)