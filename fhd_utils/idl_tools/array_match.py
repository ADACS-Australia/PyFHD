from fhd_utils.histogram import histogram
import numpy as np

def array_match(array_1, value_match, array_2 = None) :
    """
    TODO: Description for array match

    Parameters
    ----------
    array_1: array
        TODO: Add Description for Array_1
    value_match: array
        TODO: Add Description for Value_Match
    array_2: array, optional
        TODO: Add Description for Array_2, default is None

    Returns
    -------
    indices: array
        TODO: Add Description for return of array_match
    matching indices: array
        TODO: Add Description for return of array_match
    
    Raises
    ------
    ValueError
        Gets raised if value_match is None 
    """
    if value_match is None:
        raise ValueError("Value Match Should be a value not None")
    # If array_2 has been supplied, compare which mins and maxes to use based on two arrays
    if array_2 is not None and np.size(array_2) > 0:
        min_use = np.min([np.min(array_1), np.min(array_2)])
        max_use = np.max([np.max(array_1), np.max(array_2)])
        # Also compute the histogram for array_2
        hist2, _, ri2 = histogram(array_2, min = min_use, max = max_use)
    else:
        # If the second array wasn't supplied
        min_use = np.min(array_1)
        max_use = np.max(array_1)
        # Supply a second hist
        hist2 = np.arange(max_use - min_use + 1)
    # Get the histogram for the first   
    hist1, _ , ri1 = histogram(array_1, min = min_use, max = max_use)
    # Arrays should be the same size, does addition
    hist_combined = hist1 + hist2
    bins = np.where(hist_combined > 0)

    # Select the values to be used
    hist_v1, bins_v1, _ = histogram(bins+min_use)
    omin = bins_v1[0]
    omax = bins_v1[-1]
    hist_v2, _, _ = histogram(value_match, min = omin, max = omax)
    vals = np.nonzero(np.bitwise_and(hist_v1, hist_v2))[0] + omin - min_use
    n_match = vals.size

    if n_match == 0:
        return -1, n_match
    
    ind_arr = np.zeros_like(array_1)
    for vi in range(n_match):
        i = vals[vi]
        if hist1[i] > 0:
            ind_arr[ri1[ri1[i] : ri1[i+1]]] = 1
        if hist2[i] > 0:
            ind_arr[ri2[ri2[i] : ri2[i+1]]] = 1
    
    match_indices = np.nonzero(ind_arr)[0]
    # Return our matching indices
    return match_indices, match_indices.size
