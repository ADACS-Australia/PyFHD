import numpy as np
from PyFHD.pyfhd_tools.pyfhd_utils import reshape_and_average_in_time

def calibrate_qu_mixing(vis_arr: np.ndarray, vis_model_arr : np.ndarray, vis_weights: np.ndarray, obs : dict) -> float:
    """_summary_

    Parameters
    ----------
    vis_arr : np.ndarray
        _description_
    vis_model_arr : np.ndarray
        _description_
    vis_weights : np.ndarray
        _description_
    obs : dict
        _description_

    Returns
    -------
    float
        _description_
    """

    n_freq = obs['n_freq']
    # n_tile = obs['n_tile']
    n_time = obs['n_time']
    ##This should be number of baselines for one time step
    n_baselines = obs['nbaselines']

    ## reshape from (n_freq, n_baselines*n_times) to (n_freq, n_times, n_baselines). Turns out due to the row major vs col major difference
    ## between IDL and python, this shape also changes
    new_shape = (n_freq, n_time, n_baselines)

    # Use the xx weightss (yy should be identical at this point)
    weights_use = np.reshape(vis_weights[0, :, :], new_shape)
    ##carried over from FHD code - not sure this is necessary (maybe avoids NaNs?)
    weights_use = np.maximum(weights_use, np.zeros_like(weights_use))
    weights_use = np.minimum(weights_use, np.ones_like(weights_use))

    ##Q = YY - XX for data
    pseudo_q = reshape_and_average_in_time(vis_arr[1, :, :] - vis_arr[0, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)
    
    ##U = YX + XY for data
    pseudo_u = reshape_and_average_in_time(vis_arr[3, :, :] + vis_arr[2, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)
    
    ##Q = YY - XX for model
    pseudo_q_model = reshape_and_average_in_time(vis_model_arr[1, :, :] - vis_model_arr[0, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)
    
    ##U = YX + XY for model
    pseudo_u_model = reshape_and_average_in_time(vis_model_arr[3, :, :] + vis_model_arr[2, :, :],
                                           n_freq, n_time, n_baselines,
                                           weights_use)

    weights_t_avg = np.sum(weights_use, axis=1)

    i_use = np.nonzero(weights_t_avg)
    pseudo_u = np.squeeze(pseudo_u[i_use]).flatten()
    pseudo_q = np.squeeze(pseudo_q[i_use]).flatten()
    pseudo_q_model = np.squeeze(pseudo_q_model[i_use]).flatten()
    pseudo_u_model = np.squeeze(pseudo_u_model[i_use]).flatten()

    # LA_LEAST_SQUARES does not use double precision by default as such you will see differences.
    # Also LA_LEAST_SQUARES uses different method to numpy, generally LA_LEAST_SQUARES assumes the first matrix
    # has full rank, while numpy does not assume this.

    x = np.vstack([pseudo_u, np.ones(pseudo_u.size)]).T
    u_q_mix = np.linalg.lstsq(x, pseudo_q, rcond = None)[0]
    u_q_phase = np.arctan2(u_q_mix[0].imag, u_q_mix[0].real)

    x = np.vstack([pseudo_u_model, np.ones(pseudo_u_model.size)]).T
    u_q_mix_model = np.linalg.lstsq(x, pseudo_q_model, rcond = None)[0]
    u_q_phase_model = np.arctan2(u_q_mix_model[0].imag, u_q_mix_model[0].real)

    return u_q_phase_model - u_q_phase



