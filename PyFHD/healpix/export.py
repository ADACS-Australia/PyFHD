import numpy as np
from logging import RootLogger
import h5py
from pathlib import Path
from PyFHD.io.pyfhd_io import save
from PyFHD.data_setup.obs import update_obs
from PyFHD.healpix.healpix_utils import healpix_cnv_generate, healpix_cnv_apply, beam_image_cube, vis_model_freq_split
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
    n_freq_use = np.floor(obs['n_freq'] / pyfhd_config['n_avg'])
    
    obs_out = update_obs(obs, dimension_use, obs['kbinsize'], beam_nfreq_avg = ps_nfreq_avg, fov = fov_use)
    # To have a psf that has reacted to the new beam_nfreq_avg you have set that isn't
    # the default, tell PyFHD to re-create the psf here once beam_setup has been translated

    beam_arr, beam_mask = beam_image_cube(obs, psf, logger, square = True, beam_threshold = pyfhd_config['ps_beam_threshold'], n_freq_bin = n_freq_use)

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
    # Create the healpix dir Path
    healpix_dir = Path(pyfhd_config['output_dir'], 'healpix')
    healpix_dir.mkdir(exist_ok = True)
    for iter in range(n_iter):
        split = vis_model_freq_split(
            obs_out, 
            psf, 
            params, 
            vis_weights_use, 
            vis_model_arr, 
            vis_arr, 
            pyfhd_config, 
            logger, 
            uvf_name = uvf_name[iter],
            bi_use = bi_use[iter]
        )
        if dirty_flag:
            residual_flag = False
        else:
            residual_flag = True
        nf_vis = split['obs']['nf_vis']
        nf_vis_use = np.zeros(n_freq_use)
        for freq_i in range(n_freq_use):
            nf_vis_use[freq_i] = np.sum(nf_vis[freq_i * pyfhd_config['n_avg'] : (freq_i + 1) * pyfhd_config['n_avg']])
        
        beam_squared_cube = np.zeros([hpx_inds.size , n_freq_use])
        weights_cube = np.zeros([hpx_inds, n_freq_use])
        variance_cube = np.zeros([hpx_inds.size, n_freq_use])
        model_cube = np.zeros([hpx_inds.size, n_freq_use])
        dirty_or_res_cube = np.zeros([hpx_inds.size, n_freq_use])
        
        for pol_i in range(obs['n_pol']):
            for freq_i in range(n_freq_use):
                # TODO: check the indexing, I don't think it will work for Python
                beam_squared_cube[hpx_inds.size * freq_i] = healpix_cnv_apply(beam_arr[pol_i, freq_i] * nf_vis_use[freq_i], hpx_cnv)
                weights_cube[hpx_inds.size * freq_i] = healpix_cnv_apply(split['weights_arr'][pol_i, freq_i], hpx_cnv)
                variance_cube[hpx_inds.size * freq_i] = healpix_cnv_apply(split['variance_arr'][pol_i, freq_i], hpx_cnv)
                model_cube[hpx_inds.size * freq_i] = healpix_cnv_apply(split['model_arr'][pol_i, freq_i], hpx_cnv)
                dirty_or_res_cube[hpx_inds.size * freq_i] = healpix_cnv_apply(split['residual_arr'][pol_i, freq_i], hpx_cnv)
            healpix_pol_dict = {
                "obs": split['obs'],
                "hpx_inds": hpx_inds,
                "n_avg": pyfhd_config["n_avg"],
                "beam_squared_cube" : beam_squared_cube,
                "weights_cube" : weights_cube,
                "variance_cube" : variance_cube,
                "model_cube" : model_cube,
            }
            if residual_flag:
                healpix_pol_dict['res_cube'] = dirty_or_res_cube
            elif dirty_flag:
                healpix_pol_dict['dirty_cube'] = dirty_or_res_cube
            save(healpix_dir / f"healpix_{cube_name[iter]}_{obs['pol_names'][pol_i]}.h5", healpix_pol_dict, f"healpix_{cube_name[iter]}_{obs['pol_names'][pol_i]}", logger = logger)
