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

@pytest.fixture(scope="function", params=['point_zenith','point_offzenith', '1088716296'])
def tag(request):
    return request.param

@pytest.fixture(scope="function", params=['run1', 'run2', 'run3'])
def run(request):
    return request.param

skip_tests = [['1088716296', "run3"]]

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

    obs = recarray_to_dict(sav_dict['obs'])
    params = recarray_to_dict(sav_dict['params'])
    
    vis_arr = sav_file_vis_arr_swap_axes(sav_dict['vis_arr'])
    vis_model_arr = sav_file_vis_arr_swap_axes(sav_dict['vis_model_arr'])

    #super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['obs'] = obs
    h5_save_dict['params'] = params
    h5_save_dict['vis_arr'] = vis_arr
    h5_save_dict['vis_model_arr'] = vis_model_arr

    dd.io.save(before_file, h5_save_dict)

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

    #super dictionary to save everything in
    h5_save_dict = {}
    h5_save_dict['vis_baseline_hist'] = recarray_to_dict(sav_dict['vis_baseline_hist'])

    dd.io.save(after_file, h5_save_dict)

    return after_file

def test_vis_baseline_hist(before_file: Path, after_file: Path):
    """
    Runs the test on `vis_baseline_hist` - reads in the data in before_file and after_file,
    and then calls `vis_baseline_hist`, checking the outputs match expectations
    """

    if (before_file == None or after_file == None):
        pytest.skip(f"This test has been skipped because the test was listed in the skipped tests due to FHD not outpoutting them: {skip_tests}")

    h5_before = dd.io.load(before_file)
    h5_after = dd.io.load(after_file)

    obs = h5_before['obs']
    params = h5_before['params']
    vis_arr = h5_before['vis_arr']
    vis_model_arr = h5_before['vis_model_arr']

    expec_vis_baseline_hist = h5_after['vis_baseline_hist']

    result_vis_baseline_hist = vis_baseline_hist(obs, params, vis_arr, vis_model_arr)

    num_bins = expec_vis_baseline_hist['baseline_length'].shape[0]
    expec_vis_res_ratio_mean = expec_vis_baseline_hist['vis_res_ratio_mean']
    expec_vis_res_sigma = expec_vis_baseline_hist['vis_res_sigma']

    # There is an indexing error in the original FHD code, which means only
    # the first num_bins of the `vis_res_ratio_mean` array are indexed. This means
    # that only results from the last polarisation and saved, and spread over
    # the first half of the bin indexes between each pol. We can recover what
    # the second polaristaion should be at least for testing
    # The below code was changed to also allow for 4 polarizations
    # Flattening the results from the first num_bins / obs['n_pol'] rows takes out what we need
    fixed_vis_res_ratio_mean = expec_vis_res_ratio_mean[:num_bins // obs['n_pol'], :].flatten()
    fixed_vis_res_sigma = expec_vis_res_sigma[:num_bins // obs['n_pol'], :].flatten()

    rtol = 1e-5
    atol = 4e-4

    #Can test that the fixed final polarisation is close to PyFHD result
    #Out results are ordered by pol, bin so need to do a transpose
    npt.assert_allclose(fixed_vis_res_ratio_mean,
                        result_vis_baseline_hist['vis_res_ratio_mean'].transpose()[:, -1],
                        atol=atol, rtol=rtol)
    
    npt.assert_allclose(fixed_vis_res_sigma,
                        result_vis_baseline_hist['vis_res_sigma'].transpose()[:, -1],
                        atol=atol, rtol=rtol)
    

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    extent = [-0.5, 1.5, -0.5, num_bins - 0.5]

    im = axs[0].imshow(expec_vis_res_ratio_mean, aspect='auto',
                       extent=extent, origin='lower')
    plt.colorbar(im)
    axs[0].set_xticks([0, 1])


    vis_res_ratio_mean_plt = np.zeros([num_bins, 2])
    vis_res_ratio_mean_plt[:,1] = fixed_vis_res_ratio_mean
    im = axs[1].imshow(vis_res_ratio_mean_plt, aspect='auto',
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
    name_split = before_file.name.split('_')
    if (name_split[0] == 'point'):
        tag = f"{name_split[0]}_{name_split[1]}"
        run = f"{name_split[2]}"
    else:
        tag = f"{name_split[0]}"
        run = f"{name_split[1]}"
    fig.savefig(f"test_vis_baseline_hist_{tag}_{run}.png", bbox_inches='tight', dpi=300)
    plt.close()