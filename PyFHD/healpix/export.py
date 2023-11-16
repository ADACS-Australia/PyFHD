import numpy as np
from PyFHD.io.pyfhd_io import save, load
from logging import RootLogger
import h5py
from PyFHD.data_setup.obs import update_obs
from PyFHD.healpix.healpix_utils import healpix_cnv_generate, beam_image_cube
from PyFHD.flagging.flagging import vis_flag_tiles
from PyFHD.pyfhd_tools.pyfhd_utils import vis_weights_update, split_vis_weights, vis_noise_calc

def healpix_snapshot_cube_generate(obs: dict, psf: dict | h5py.File, cal: dict, params: dict, vis_arr: np.ndarray, vis_model_arr: np.ndarray, vis_weights: np.ndarray, pyfhd_config: dict, logger: RootLogger) -> None:
    """
    TODO: _summary_

    Parameters
    ----------
    obs : dict
        _description_
    psf : dict | h5py.File
        _description_
    cal : dict
        _description_
    params : dict
        _description_
    vis_arr : np.ndarray
        _description_
    vis_model_arr : np.ndarray
        _description_
    vis_weights : np.ndarray
        _description_
    pyfhd_config : dict
        _description_
    logger : RootLogger
        _description_
    """
    if pyfhd_config['split_ps_export']:
        cube_name = ['hpx_even', 'hpx_odd']
    else:
        cube_name = ['healpix_cube']

    if pyfhd_config['ps_kbinsize'] is not None:
        kbinsize = pyfhd_config['ps_kbinsize']
    elif pyfhd_config['ps_fov'] is not None:
        # Does this get used in conjunction with the fov_use down
        # below, if so, we're converting from Radians to Degrees twice?
        kbinsize = (180 / np.pi) / pyfhd_config['ps_fov']
    else:
        kbinsize = obs['kpix']

    fov_use = (180/ np.pi) / kbinsize

    if pyfhd_config['ps_kspan'] is not None:
        dimension_use = pyfhd_config['ps_kspan'] / kbinsize
    elif pyfhd_config['ps_dimension'] is not None:
        dimension_use = pyfhd_config['ps_dimension']
    elif pyfhd_config['ps_degpix']:
        dimension_use = fov_use / pyfhd_config['ps_degpix']
    else:
        dimension_use = fov_use / obs['degpix']
    
    if pyfhd_config['ps_nfreq_avg'] is None:
        fbin_i = psf['fbin_i']
        if isinstance(psf, h5py.File):
            fbin_i = fbin_i[:]
        ps_nfreq_avg = np.round(obs['n_freq'] / np.max(fbin_i + 1))
    else:
        ps_nfreq_avg = pyfhd_config['ps_nfreq_avg']

    degpix_use = fov_use / dimension_use
    pix_sky = (4 * np.pi * (180 / np.pi) ** 2) / degpix_use ** 2
    # Below should = 1024 for 0.1119 degrees/pixel
    nside = 2 ** (np.ceil(np.log(np.sqrt(pix_sky / 12)) / np.log(2)))
    
    obs_out = update_obs(obs, dimension_use, obs['kbinsize'], beam_nfreq_avg = ps_nfreq_avg, fov = fov_use)
    # To have a psf that has reacted to the new beam_nfreq_avg you have set that isn't
    # the default, tell PyFHD to re-create the psf here once beam_setup has been translated

    beam_arr, beam_mask = beam_image_cube(obs, psf, logger, square = True, beam_threshold = pyfhd_config['ps_beam_threshold'])

    hpx_radius = fov_use / np.sqrt(2)

    hpx_cnv = healpix_cnv_generate(obs_out, beam_mask, hpx_radius, pyfhd_config, logger, nside = nside)
    hpx_inds = hpx_cnv['inds']

    if len(pyfhd_config['ps_tile_flag_list']) > 0:
        vis_weights = vis_flag_tiles(obs_out, vis_weights, pyfhd_config['ps_tile_flag_list'], logger)
    
    vis_weights, obs_out = vis_weights_update(vis_weights, obs_out, psf, params)

    if pyfhd_config['split_ps_export']:
        n_iter = 2
        vis_weights_use, bi_use = split_vis_weights(obs_out, vis_weights)
        obs_out['vis_noise'] = vis_noise_calc(obs_out, vis_arr, vis_weights, bi_use = bi_use)
        uvf_name = ['even', 'odd']
    else:
        n_iter = 1
        uvf_name = ['']
        obs_out['vis_noise'] = vis_noise_calc(obs_out, vis_arr, vis_weights)
        bi_use = np.zeros(1, dtype = np.int64)

    # Looks like this is set to False by default?
    residual_flag = obs_out['residual']
    # Since the model is imported by default, dirty_flag is usually True
    dirty_flag = not residual_flag and vis_model_arr is not None

    t_hpx = 0
    for iter in range(n_iter):
        # freq_split = 


