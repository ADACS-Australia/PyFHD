import pytest
from os import environ as env
from pathlib import Path
from PyFHD.pyfhd_tools.test_utils import get_data_items, get_data_sav
from PyFHD.calibration.calibration_utils import vis_baseline_hist
from PyFHD.use_idl_fhd.use_idl_outputs import convert_sav_to_dict
from PyFHD.pyfhd_tools.test_utils import recarray_to_dict, sav_file_vis_arr_swap_axes
import numpy as np
import deepdish as dd
from logging import RootLogger
import numpy.testing as npt
import matplotlib.pyplot as plt

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), "vis_baseline_hist")

def run_test(data_dir, tag_name):
    """Runs the test on `vis_baseline_hist` - reads in the data in `data_loc`,
    and then calls `vis_baseline_hist`, checking the outputs match expectations"""

    h5_before = dd.io.load(Path(data_dir, f"{tag_name}_before_vis_baseline_hist.h5"))
    h5_after = dd.io.load(Path(data_dir, f"{tag_name}_after_vis_baseline_hist.h5"))

    obs = h5_before['obs']
    params = h5_before['params']
    vis_arr = h5_before['vis_arr']
    vis_model_arr = h5_before['vis_model_arr']

    expec_vis_baseline_hist = h5_after['vis_baseline_hist']

    result_vis_baseline_hist = vis_baseline_hist(obs, params, vis_arr, vis_model_arr)

    num_bins = expec_vis_baseline_hist['baseline_length'].shape[0]
    expec_vis_res_ratio_mean = expec_vis_baseline_hist['vis_res_ratio_mean']
    expec_vis_res_sigma = expec_vis_baseline_hist['vis_res_sigma']

    ##There is an indexing error in the original FHD code, which means only
    ##the first num_bins of the `vis_res_ratio_mean` array are indexed. This means
    ##that only results from the last polarisation and saved, and spread over
    ##the first half of the bin indexes between each pol. We can recover what
    ##the second polaristaion should be at least for testing

    fixed_vis_res_ratio_mean = np.zeros((num_bins, 2))
    range0 = range(0, num_bins, 2)
    fixed_vis_res_ratio_mean[range0, 1] = expec_vis_res_ratio_mean[:int(num_bins/2), 0]
    range1 = range(1, num_bins, 2)
    fixed_vis_res_ratio_mean[range1, 1] = expec_vis_res_ratio_mean[:int(num_bins/2), 1]


    fixed_vis_res_sigma = np.zeros((num_bins, 2))
    range0 = range(0, num_bins, 2)
    fixed_vis_res_sigma[range0, 1] = expec_vis_res_sigma[:int(num_bins/2), 0]
    range1 = range(1, num_bins, 2)
    fixed_vis_res_sigma[range1, 1] = expec_vis_res_sigma[:int(num_bins/2), 1]

    rtol = 1e-5
    atol = 2e-4

    ##Can test that the fixed final polarisation is close to PyFHD result
    ##Out results are ordered by pol, bin so need to do a transpose
    npt.assert_allclose(fixed_vis_res_ratio_mean[:, 1],
                        result_vis_baseline_hist['vis_res_ratio_mean'].transpose()[:, 1],
                        atol=atol, rtol=rtol)
    
    npt.assert_allclose(fixed_vis_res_sigma[:, 1],
                        result_vis_baseline_hist['vis_res_sigma'].transpose()[:, 1],
                        atol=atol, rtol=rtol)
    

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    extent = [-0.5, 1.5, -0.5, num_bins - 0.5]

    im = axs[0].imshow(expec_vis_res_ratio_mean, aspect='auto',
                       extent=extent, origin='lower')
    plt.colorbar(im)
    axs[0].set_xticks([0, 1])

    im = axs[1].imshow(fixed_vis_res_ratio_mean, aspect='auto',
                       extent=extent, origin='lower',
                       vmin=result_vis_baseline_hist['vis_res_ratio_mean'].min(),
                       vmax=result_vis_baseline_hist['vis_res_ratio_mean'].max())
    plt.colorbar(im)
    axs[1].set_xticks([0, 1])

    im = axs[2].imshow(result_vis_baseline_hist['vis_res_ratio_mean'].transpose(), aspect='auto',
                       extent=extent, origin='lower')
    plt.colorbar(im)
    axs[2].set_xticks([0, 1])

    axs[0].set_title('FHD')
    axs[1].set_title('FHD (fixed)')
    axs[2].set_title('PyFHD')

    for ax in axs.flatten():
        ax.set_xlabel('Polarisation')

    axs[0].set_ylabel("Baseline hist bin")

    plt.tight_layout()
    fig.savefig('test_vis_baseline_hist.png', bbox_inches='tight', dpi=300)
    plt.close()


def test_pointsource1_vary1(data_dir):
    """Test using the `pointsource1_vary1` set of inputs"""

    run_test(data_dir, "pointsource1_vary1")

def test_pointsource2_vary1(data_dir):
    """Test using the `pointsource2_vary1` set of inputs"""

    run_test(data_dir, "pointsource2_vary1")

if __name__ == "__main__":

    def convert_before_sav(data_dir, tag_name):
        """Takes the before .sav file out of FHD function `vis_baseline_hist`
        and converts into an hdf5 format"""

        func_name = 'vis_baseline_hist'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_before_{func_name}.sav", "meh")

        obs = recarray_to_dict(sav_dict['obs'])
        params = recarray_to_dict(sav_dict['params'])
        
        vis_arr = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
        vis_model_arr = sav_file_vis_arr_swap_axes(sav_dict['vis_model_arr'])
            
        ##super dictionary to save everything in
        h5_save_dict = {}
        h5_save_dict['obs'] = obs
        h5_save_dict['params'] = params
        h5_save_dict['vis_arr'] = vis_arr
        h5_save_dict['vis_model_arr'] = vis_model_arr

        dd.io.save(Path(data_dir, f"{tag_name}_before_{func_name}.h5"), h5_save_dict)
        
    def convert_after_sav(data_dir, tag_name):
        """Takes the after .sav file out of FHD function `vis_baseline_hist`
        and converts into an hdf5 format"""

        func_name = 'vis_baseline_hist'

        sav_dict = convert_sav_to_dict(f"{data_dir}/{tag_name}_after_{func_name}.sav", "meh")
        
        ##super dictionary to save everything in
        h5_save_dict = {}
        
        h5_save_dict['vis_baseline_hist'] = recarray_to_dict(sav_dict['vis_baseline_hist'])

        dd.io.save(Path(data_dir, f"{tag_name}_after_{func_name}.h5"), h5_save_dict)
        
    def convert_sav(base_dir, tag_name):
        """Load the inputs and outputs needed for testing `vis_baseline_hist`"""
        convert_before_sav(base_dir, tag_name)
        convert_after_sav(base_dir, tag_name)

    ##Where be all of our data
    base_dir = Path(env.get('PYFHD_TEST_PATH'), 'vis_baseline_hist')

    tag_names = ['pointsource1_vary1', 'pointsource2_vary1']

    for tag_name in tag_names:
        convert_sav(base_dir, tag_name)
        # run_test(base_dir, tag_name)