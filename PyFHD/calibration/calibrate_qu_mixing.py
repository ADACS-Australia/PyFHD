import numpy as np

def calibrate_qu_mixing(vis_arr: np.ndarray, vis_model_arr : np.ndarray, vis_weights: np.ndarray) -> float:
    """_summary_

    Parameters
    ----------
    vis_arr : np.ndarray
        _description_
    vis_model_arr : np.ndarray
        _description_
    vis_weights : np.ndarray
        _description_

    Returns
    -------
    float
        _description_
    """
    # Use the xx weightss (yy should be identical at this point)
    weights_use = np.maximum(np.zeros_like(vis_weights[0, :, :, :]), vis_weights[0, :, :, :])
    weights_use = np.minimum(weights_use, np.ones_like(weights_use))

    # average the visibilities in time
    pseudo_q = vis_arr[1, :, :, :] - vis_arr[0, :, :, :]
    # TODO: Check axis of pseudo_q
    pseudo_q = np.sum(pseudo_q * weights_use, axis = -1)
    pseudo_u = vis_arr[3, :, :, :] - vis_arr[2, :, :, :]
    # TODO: Check axis of pseudo_u
    pseudo_u = np.sum(pseudo_u * weights_use, axis = -1)
    pseudo_q_model = vis_model_arr[1, :, :, :] - vis_model_arr[0, :, :, :]
    # TODO: Check axis of pseudo_q_model
    pseudo_q_model = np.sum(pseudo_q_model * weights_use, axis = -1)
    pseudo_u_model = vis_model_arr[3, :, :, :] - vis_model_arr[2, :, :, :]
    # TODO: Check axis of pseudo_u_model
    pseudo_u_model = np.sum(pseudo_u_model * weights_use, axis = -1)
    # TODO: Check axis of weight
    weight = np.sum(weights_use, axis = -1)
    i_use = np.nonzero(weight)
    pseudo_u = np.squeeze(pseudo_u[i_use]).flatten
    pseudo_q = np.squeeze(pseudo_q[i_use]).flatten
    pseudo_q_model = np.squeeze(pseudo_q_model[i_use]).flatten
    pseudo_u_model = np.squeeze(pseudo_u_model[i_use]).flatten

    # LA_LEAST_SQUARES does not use double precision by default as such you will see differences.
    # Also LA_LEAST_SQUARES uses different method to numpy, generally LA_LEAST_SQUARES assumes the first matrix
    # has full rank, while numpy does not assume this.
    u_q_mix = np.linalg.lstsq(pseudo_u, pseudo_q, rcond = None)[0]
    # TODO: Check if complex, if not change arctan type
    u_q_phase = np.arctan2(u_q_mix[0].imag, u_q_mix[0].real)
    u_q_mix_model = np.linalg.lstsq(pseudo_u_model, pseudo_q_model, rcond = None)[0]
    # TODO: Check if complex, if not change arctan type
    u_q_phase_model = np.arctan2(u_q_mix_model[0].imag, u_q_mix_model[0].real)

    return u_q_phase_model - u_q_phase



