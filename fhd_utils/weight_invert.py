import numpy as np

def weight_invert(weights, threshold = None):
    """
    The weights invert function cleans the weights given by removing
    the values that are 0, NaN or Inf ready for additional calculation.
    If a threshold is set, then we check the values that match the threshold
    instead of checking for zeros.

    Parameters
    ----------
    weights: Complex
        An array of values of some dtype
    threshold: float
        A real number set as the threshold for the array.
        By default its set to None, in this case function checks
        for zeros.
    abs: bool, optional
        Set to False by default. When True, weights are compared against
        threshold or search for zeros using absolute values.

    Returns
    -------
    result: array
        The weights array that has had NaNs and Infinities removed, and zeros OR
        values that don't meet the threshold.
    """
    result = np.zeros(weights.shape, dtype = weights.dtype)
    '''
    Python and IDL use the where function on complex numbers differently.
    On Python, if you apply a real threshold, it applies to only the real numbers,
    and if you apply an imaginary threshold it applies to only imaginary numbers.
    For example Python:
    test = np.array([1j, 2 + 2j, 3j])
    np.where(test >= 2) == array([1])
    np.where(test >= 2j) == array([1,2])
    Meanwhile in IDL:
    test = COMPLEX([0,2,0],[1,2,3]) ;COMPLEX(REAL, IMAGINARY)
    where(test ge 2) == [1, 2]
    where(test ge COMPLEX(0,2)) == [1, 2]

    IDL on the otherhand, uses the ABS function on COMPLEX numbers before using WHERE.
    Hence the behaviour we're seeing above.
    '''
    weights_use = weights
    if np.iscomplexobj(weights):
        weights_use = np.abs(weights)
    # If threshold has been set then...
    if threshold is not None:
        # Get the indexes which meet the threshold
        # As indicated IDL applies abs before using where to complex numbers
        i_use = np.where(weights_use >= threshold)
    else:
        # Otherwise get where they are not zero
        i_use = np.where(weights_use)
    if np.size(i_use) > 0:
            result[i_use] = 1 / weights[i_use]
    # Replace all NaNs with Zeros
    if np.size(np.where(np.isnan(result))) != 0:
        result[np.where(np.isnan(result))] = 0
    # Replace all Infinities with Zeros
    if np.size(np.where(np.isinf(result))) != 0:
        result[np.where(np.isinf(result))] = 0
    # If the result contains only 1 result, then return the result, not an array
    if np.size(result) == 1:
        result = result[0]
    return result
        
