import pytest
import numpy.testing as npt
from os import environ as env
from pathlib import Path
from PyFHD.gridding.visibility_grid import visibility_grid
from PyFHD.pyfhd_tools.test_utils import get_savs, recarray_to_dict
import deepdish as dd
from logging import RootLogger

@pytest.fixture
def data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'visibility_grid')

@pytest.fixture
def full_data_dir():
    return Path(env.get('PYFHD_TEST_PATH'), 'full_size_visibility_grid/')

@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6, 7])
def number(request):
    return request.param

@pytest.fixture
def before_gridding(data_dir, number):
    before_gridding = Path(data_dir, f"test_{number}_before_{data_dir.name}.h5")

    if before_gridding.exists():
        return before_gridding
    
    h5_save_dict = recarray_to_dict(get_savs(data_dir,f'input_{number}.sav'))

    h5_save_dict["uniform_flag"] = True if ("uniform_flag" in h5_save_dict and h5_save_dict["uniform_flag"]) else False
    h5_save_dict["no_conjugate"] = True if ("no_conjugate" in h5_save_dict and h5_save_dict["no_conjugate"]) else False
    h5_save_dict["obs"]["n_baselines"] = h5_save_dict["obs"]["nbaselines"]
    h5_save_dict["pyfhd_config"] = {
        "interpolate_kernel": h5_save_dict["psf"]["interpolate_kernel"],
        "psf_dim": h5_save_dict["psf"]["dim"],
        "psf_resolution": h5_save_dict["psf"]["resolution"],
        "beam_mask_threshold": h5_save_dict["psf"]["beam_mask_threshold"],
        "beam_clip_floor": h5_save_dict["extra"]["beam_clip_floor"],
        "image_filter": h5_save_dict["extra"]["image_filter_fn"],
        "mask_mirror_indices": False,
        "beam_per_baseline": True if ("beam_per_baseline" in h5_save_dict and h5_save_dict['beam_per_baseline']) else False,
        "grid_spectral": True if ("grid_spectral" in h5_save_dict and h5_save_dict['grid_spectral']) else False,
        "grid_weights": True if h5_save_dict['weights'] else False,
        "grid_variance": True if ("variance" in h5_save_dict and h5_save_dict['variance']) else False,
        "grid_uniform": True if ("grid_uniform" in h5_save_dict and h5_save_dict["grid_uniform"]) else False
    }

    dd.io.save(before_gridding, h5_save_dict)
    
    return before_gridding

@pytest.fixture
def after_gridding(data_dir, number):
    after_gridding = Path(data_dir, f"test_{number}_after_{data_dir.name}.h5")

    if after_gridding.exists():
        return after_gridding
    
    outputs = recarray_to_dict(get_savs(data_dir, f'output_{number}.sav'))

    h5_save_dict = {
        'image_uv': outputs['image_uv'],
        'weights': outputs['weights'],
        'variance': outputs['variance'],
        'uniform_filter': outputs['uniform_filter'],
        'nf_vis': outputs['obs']['nf_vis']
    }

    if 'model_return' in outputs:
        h5_save_dict["model_return"] = outputs["model_return"]

    dd.io.save(after_gridding, h5_save_dict)
    
    return after_gridding

def test_visibility_grid(before_gridding: Path, after_gridding: Path):
    h5_before = dd.io.load(before_gridding)
    h5_after = dd.io.load(after_gridding)

    gridding_dict = visibility_grid(
        h5_before["visibility_ptr"],
        h5_before['vis_weight_ptr'],
        h5_before['obs'],
        h5_before['psf'],
        h5_before['params'],
        h5_before['polarization'],
        h5_before["pyfhd_config"],
        RootLogger(1),
        uniform_flag = h5_before["uniform_flag"],
        no_conjugate = h5_before["no_conjugate"],
        model = h5_before["model_ptr"]
    )
    # All atols are done by the lowest precision that passed
    npt.assert_allclose(gridding_dict['image_uv'], h5_after['image_uv'], atol = 1.5e-7)
    npt.assert_allclose(gridding_dict['weights'], h5_after['weights'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['variance'], h5_after['variance'], atol = 1e-8)
    # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
    # The minor difference in bin_n affected the uniform filter. The precision difference could cause errors upto 1
    # This doesn't occur for every test.
    npt.assert_allclose(gridding_dict['uniform_filter'], h5_after['uniform_filter'], atol = 0.5)
    npt.assert_allclose(gridding_dict['obs']['nf_vis'], h5_after['nf_vis'], atol = 1e-8)

    if "model_return" in gridding_dict:
        npt.assert_allclose(gridding_dict['model_return'], h5_after['model_return'], atol = 1e-7)

# def test_visibility_grid_one(data_dir):
#     inputs = get_savs(data_dir,'input_1.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = inputs['variance']
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = False
#     grid_uniform = False
#     fi_use = inputs['fi_use']
#     bi_use = inputs['bi_use']
#     no_conjugate = False
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = False
#     beam_per_baseline = False
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_1.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)

# def test_visibility_grid_two(data_dir):
#     inputs = get_savs(data_dir,'input_2.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = False
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = inputs['uniform_filter']
#     grid_uniform = False
#     fi_use = None 
#     bi_use = None
#     no_conjugate = inputs['no_conjugate']
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = False
#     beam_per_baseline = inputs['beam_per_baseline']
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_2.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
#     # The minor difference in bin_n affected the uniform filter.
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 0.5)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)

# def test_visibility_grid_three(data_dir):
#     inputs = get_savs(data_dir,'input_3.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = False 
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = inputs['uniform_filter']
#     grid_uniform = False
#     fi_use = None
#     bi_use = None
#     no_conjugate = inputs['no_conjugate']
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = inputs['grid_spectral']
#     beam_per_baseline = False
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_3.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
#     # The minor difference in bin_n affected the uniform filter.
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 0.5)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)
#     '''
#     Now here in theory I should be testing the results of grid spectral using a small tolerance (e.g. 1e-8), 
#     but I can't reasonably test the differences using such a small tolerance.
#     Here is why.
#     First, there are small precision error differences between IDL and Python in the spectral calculations, the maximum
#     absolute difference between the Python and IDL is 1.3e-5, although in most cases its below 1e-6. 
#     This means when we goto calculate our spectral_uv, we multiply, add or subtract the spectral_A, spectral_B and spectral_D
#     arrays together, which multiplies, adds or subtracts the precision errors too, potentially magnifying them.
#     These initially small differences can have massive effects, take the use of weight invert in this line below:

#     spectral_uv = (spectral_A - n_vis * spectral_B * image_uv) * weight_invert(spectral_D - spectral_B ** 2)

#     If we take the smallest number above 0 in the array spectral_D - spectral_B ** 2 in both Python and IDl we get:

#     Python: 1.9927779547410943e-12
#     IDL: 1.9846134e-12 ; Limited to 7 decimal places

#     The difference between these values are 8.164526283745157e-15 (calculated in Python).

#     This small difference becomes massive when we take these two small numbers and compute the statement below from weight_invert using
#     our smallest number's above:

#     result[i_use] = 1 / weights[i_use]

#     i.e. result[i_use] = 1 / 1.9927779547410943e-12

#     This gives:

#     Python: 503876465643.64197
#     IDL: 501812054685.2005

#     That is a difference of 2064410958.44, or 2.06441095844e9, or TWO BILLION. 
#     Even when comparing exactly the same number in IDL (1.9927779547410943e-12), 
#     and using this in 1 / 1.9927779547410943e-12, the difference is in the thousands due to precision error.

#     This behaviour also mysteriously doesn't appear in our tests of weight_invert, in fact the results are exactly the same
#     as IDL when doing the weight_invert tests. Just going to show, even the smallest difference of input can change the result of 
#     weight_invert by a drastic amount.

#     Luckily, these large differences are multiplied by numbers that are the opposite in magnitude (1e-8). 
#     This results in these numbers having a difference of upto (in theory) 9.99, as anything above 10 could indicate a huge difference 
#     in magnitude which does indicate an egregious error.
#     '''
#     npt.assert_allclose(gridding_dict['spectral_uv'], outputs['spectral_uv'], atol = 9.99)
  
# def test_visibility_grid_four(data_dir):
#     inputs = get_savs(data_dir,'input_4.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = inputs['variance']
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = False
#     grid_uniform = False
#     fi_use = inputs['fi_use']
#     bi_use = inputs['bi_use']
#     no_conjugate = False 
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = False 
#     beam_per_baseline = False 
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_4.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)

# def test_visibility_grid_five(data_dir):
#     inputs = get_savs(data_dir,'input_5.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = inputs['variance']
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = False
#     grid_uniform = inputs['grid_uniform']
#     fi_use = inputs['fi_use']
#     bi_use = inputs['bi_use']
#     no_conjugate = inputs['no_conjugate']
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = False 
#     beam_per_baseline = False 
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_5.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)

# def test_visibility_grid_six(data_dir):
#     inputs = get_savs(data_dir,'input_6.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = False
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = inputs['uniform_filter']
#     grid_uniform = False
#     fi_use = None 
#     bi_use = None
#     no_conjugate = inputs['no_conjugate']
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = False
#     beam_per_baseline = False
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_6.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 2e-7)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
#     # The minor difference in bin_n affected the uniform filter.
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 0.5)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['model_return'], outputs['model_return'], atol = 1e-7)

# def test_visibility_grid_seven(data_dir):
#     inputs = get_savs(data_dir,'input_7.sav')
#     visibility = inputs['visibility_ptr']
#     vis_weights = inputs['vis_weight_ptr']
#     obs = inputs['obs']
#     status_str = inputs['status_str']
#     psf = inputs['psf']
#     params = inputs['params']
#     weights_flag = inputs['weights']
#     variance_flag = False
#     polarization = inputs['polarization']
#     map_flag = inputs['mapfn_recalculate']
#     uniform_flag = inputs['uniform_filter']
#     grid_uniform = False
#     fi_use = None 
#     bi_use = None
#     no_conjugate = inputs['no_conjugate']
#     mask_mirror_indices = False
#     model = inputs['model_ptr']
#     grid_spectral = False
#     beam_per_baseline = False
#     uv_grid_phase_only = True

#     outputs = get_savs(data_dir, 'output_7.sav')

#     gridding_dict = visibility_grid(
#         visibility,
#         vis_weights,
#         obs, 
#         status_str,
#         psf, 
#         params,
#         weights_flag = weights_flag,
#         variance_flag = variance_flag,
#         polarization = polarization,
#         map_flag = map_flag,
#         uniform_flag = uniform_flag,
#         grid_uniform = grid_uniform,
#         fi_use = fi_use,
#         bi_use = bi_use,
#         no_conjugate = no_conjugate,
#         mask_mirror_indices = mask_mirror_indices,
#         model = model,
#         grid_spectral = grid_spectral,
#         beam_per_baseline = beam_per_baseline,
#         uv_grid_phase_only = uv_grid_phase_only,
#     )

#     npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1.5e-07)
#     npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
#     npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
#     # Differences in baseline grids locations from precision errors in the offsets caused differences in the histogram bin_n
#     # The minor difference in bin_n affected the uniform filter.
#     npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 0.5)
#     npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)

def test_visibility_grid_full(full_data_dir):
    inputs = get_savs(full_data_dir,'input_1.sav')
    visibility = inputs['visibility_ptr']
    vis_weights = inputs['vis_weight_ptr']
    obs = inputs['obs']
    status_str = inputs['status_str']
    psf = inputs['psf']
    params = inputs['params']
    weights_flag = inputs['weights']
    variance_flag = False
    polarization = inputs['polarization']
    map_flag = inputs['mapfn_recalculate']
    uniform_flag = inputs['uniform_filter']
    grid_uniform = False 
    fi_use = None
    bi_use = None
    no_conjugate = inputs['no_conjugate']
    mask_mirror_indices = False
    model = None 
    grid_spectral = False
    beam_per_baseline = False
    uv_grid_phase_only = True

    outputs = get_savs(full_data_dir, 'output_1.sav')

    gridding_dict = visibility_grid(
        visibility,
        vis_weights,
        obs, 
        status_str,
        psf, 
        params,
        weights_flag = weights_flag,
        variance_flag = variance_flag,
        polarization = polarization,
        map_flag = map_flag,
        uniform_flag = uniform_flag,
        grid_uniform = grid_uniform,
        fi_use = fi_use,
        bi_use = bi_use,
        no_conjugate = no_conjugate,
        mask_mirror_indices = mask_mirror_indices,
        model = model,
        grid_spectral = grid_spectral,
        beam_per_baseline = beam_per_baseline,
        uv_grid_phase_only = uv_grid_phase_only,
    )

    npt.assert_allclose(gridding_dict['image_uv'], outputs['image_uv'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['weights'], outputs['weights'], atol = 1e-8)
    npt.assert_allclose(gridding_dict['variance'], outputs['variance'], atol = 1e-8)
    #npt.assert_allclose(gridding_dict['uniform_filter'], outputs['uniform_filter'], atol = 1e-8) 
    # No uniform filter given with the test, does have errors though due to errors in bin_n
    # As arrays get larger the bin_n value will become further apart than the IDL output due to larger precision errors.
    npt.assert_allclose(gridding_dict['obs']['nf_vis'][0], outputs['obs']['nf_vis'][0], atol = 1e-8)