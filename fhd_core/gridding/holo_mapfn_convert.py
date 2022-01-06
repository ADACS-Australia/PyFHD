import numpy as np
from fhd_utils.modified_astro.meshgrid import meshgrid
from fhd_utils.histogram import histogram

def holo_mapfn_convert(map_fn, psf_dim, dimension, elements = None, norm = 1, threshold = 0):
    """
    TODO: Description

    Parameters
    ----------
    map_fn: ndarray
        TODO: Description
    psf_dim: int, float
        TODO: Description
    dimension: int
        TODO: Description
    elements: None, optional
        TODO: Description
    norm: int
        TODO: Description
    threshold: int, float, optional
        TODO: Description

    Returns
    -------
    
    """
    # Set up all the necessary arrays and numbers
    if elements is None:
        elements = dimension
    psf_dim2 = 2 * psf_dim
    psf_n = psf_dim ** 2
    sub_xv = meshgrid(psf_dim2, 1) - psf_dim
    sub_yv = meshgrid(psf_dim2, 2) - psf_dim
    # Generate an array of shape elements x dimension
    n_arr = np.zeros((elements, dimension))
    # Replace the values of n_arr with the number from each array in mapfn that exceeds the threshold
    for xi in range(dimension - 1):
        for yi in range(elements - 1):
            temp_arr = map_fn[xi, yi]
            n1 = np.size(np.where(abs(temp_arr) > threshold))
            n_arr[xi,yi] = n1
    # Get the ones to use
    i_use = np.where(n_arr)
    # Get the amount we're using
    i_use_size = np.size(i_use)
    # If we aren't using any then return 0
    if i_use_size == 0:
        return 0
    
    # Get the reverse indices
    _, _, ri = histogram(i_use, min = 0)
    # Create zeros of the same size as what we're using
    sa = np.zeros(i_use_size)
    ija = np.zeros(i_use_size)
    # Fill in the sa and ija arrays
    for index in range(i_use_size):
        i = i_use[index]
        xi = i % dimension
        yi = np.floor(i / dimension)
        map_fn_sub = map_fn[xi, yi]
        j_use = np.where(np.abs(map_fn_sub) > threshold)

        xii_arr = sub_xv[j_use] + xi
        yii_arr = sub_yv[j_use] + yi
        sa[index] = map_fn_sub[j_use]
        ija[index] = ri[ri[xii_arr + dimension * yii_arr]]
    # Return as record array
    mapfn = np.recarray(ija, sa, i_use, norm, 1, \
                        dtype=[(('ija', 'IJA'), 'int'), (('sa', 'SA'), 'int'), \
                               (('i_use', 'I_USE'), 'int'), (('norm', 'NORM'), 'float'),\
                               (('indexed', 'INDEXED'), '>i2')])
    return mapfn





            