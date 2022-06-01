import numpy as np
from numba import njit
from math import factorial
from typing import Tuple
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.time import Time
from astropy import units as u
from math import pi

@njit(parallel = True)
def get_bins(min, max, bin_size):
    """
    Calculates the bins for the histogram and reverse indices based on a 
    minimum and maximum value, plus a given bin size. It mirrors IDL's way of
    calculating the bin edges. In IDL bins seem to be right hand side of the bin open
    like IDL, even at the end. However, in comparison to NumPy, the last bin is always
    the maximum value. This does utilize numba anbd parallelization to make it faster.

    Parameters
    ----------
    min : int, float
        The minimum chosen for the histogram
    max : int, float
        The maximum chosen for the histogram
    bin_size : int, float
        The bin size chosen for the histogram. This histogram alwways uses bins
        of equal widths.

    Returns
    -------
    bins
        A NumPy array of all the bins ranging from min (bin[0]) to the max (bin[-1]). The step
        is bin_size.

    See Also
    --------
    histogram: Calculates the bins, histogram and reverse indices
    get_hist: Calculates histogram only
    get_ri: Calculates the reverse indices only
    """

    return np.arange(min , max + bin_size, bin_size)

@njit
def get_hist(data, bins, min, max):
    """
    Calculates the histogram based on the given bins and data, taking into account
    the minimum and maximum

    Parameters
    ----------
    data : np.array
        A NumPy array that is of one dtype float, int, or complex. Cannot be object
    bins : np.array
        A NumPy array of bins for the histogram
    min : int, float
        The minimum set for the data
    max : int, float
        The maximum set for the data

    Returns
    -------
    hist
        A histogram of the data using the bins found in the bins array.

    See Also
    --------
    histogram: Calculates the bins, histogram and reverse indices.
    get_bins: Calculates the bins only
    get_ri: Calculates the reverse indices only
    """

    bin_l = bins.size
    # Setup the histogram
    hist = np.zeros(bin_l, dtype = np.int64)
    # Setup the things required for the indexing
    n = bin_l - 1
    bin_min = bins[0]
    bin_max = bins[-1]
    # Now loop through the data
    for idx in range(data.size):
        # Check if its inside the range we set
        if data[idx] < min or data[idx] > max:
            continue
        # Calculate the index for indices and histogram
        bin_i = int(n * (data[idx] - bin_min) / (bin_max - bin_min))
        # Add to the histogram 
        hist[bin_i] += 1
    return hist

@njit
def get_ri(data, bins, hist, min, max):
    """
    Calculates the reverse indices of a data and histogram. 
    The function replicates IDL's REVERSE_INDICES keyword within
    their HISTOGRAM function. The reverse indices array which is returned
    by this function can be hard to understand at first, I will explain it here
    and also link JD Smith's famous article on IDL's HISTOGRAM.

    The reverse indices array is two vectors concatenated together. The first vector contains
    indexes for the second vector, this vector should be the size of bins + 1. 
    The second vector contains indexes from the data itself, and should be the size of data.
    The justification for having such an array is to quickly make adjustments to certain bins
    without having to search the array multiple times, thus avoiding multiple O(data.size) loops.

    The first vector indexes contain the starting positions of each bin in the second vector. For example, 
    between the indexes given by first_vector[0] and first_vector[1] of the second vector should be all the 
    indexes of bins[0] from inside the data. So if I wanted to make adjustments to the entire first bin, 
    and only the first bin I can use the reverse indices array, ri to do this. Let's say I wanted to flag
    all values of bins[0] with -1 for some reason to make them invalid in other calculations with the data, 
    then I could do this:

    `data[ri[ri[0] : ri[1]]] = -1`

    Or more generally

    `data[ri[ri[i] : ri[i + 1]]] = -1`

    Where i is 0 <= i <= bins.size.

    If you wish to gain a better understanding of how this get_ri function works, and the associated
    histogram function I have created here, please use the link given in the Notes section. This
    link will take you JD Smith's article on IDL's HISTOGRAM, it is an article which explains the
    IDL HISTOGRAM function better than IDL's own documentation. If you must gain a deeper understanding,
    read it once, gasp and get your shocks and many cries of why out of your system, then read it again.
    And keep reading it till you understand, as per the editor's note on the article:

    "...If you read it enough, the secrets of the command will be revealed to you. Stranger things have happened"

    Parameters
    ----------
    data : np.array
        A NumPy array of the data
    bins : np.array
        A NumPy array containing the bins for the histogram
    hist : np.array
        A NumPy array containing the histogram
    min : int, float
        The minimum for the dataset
    max : int, float
        The maximum for the dataset

    Returns
    -------
    ri : np.array
        An array containing the reverse indices, which is two vectors, the first vector containing indexes for the second vector.
        The second vector contains indexes for the data.

    See Also
    ---------
    histogram: Calculates the bins, histogram and reverse indices.
    get_bins: Calculates the bins only
    get_hist: Calculates histogram only

    Notes
    ------
    'HISTOGRAM: The Breathless Horror and Disgust' : http://www.idlcoyote.com/tips/histogram_tutorial.html
    """

    bin_l = bins.size
    # Setup the first vector
    data_idx = bin_l + 1
    first_v = [bin_l + 1]
    # Add the bins
    for bin in hist:
        data_idx += bin
        first_v.append(data_idx)
    # Setup the reverse indices 
    ri = np.zeros(bin_l + 1 + data.size, dtype = np.int64)
    ri[0 : bin_l + 1] = first_v
    # Create a tracking array to keep track of where we are in the data indexing
    tracker = np.array(first_v)
    # Setup the things required for the indexing
    n = bin_l - 1
    bin_min = bins[0]
    bin_max = bins[-1]
    # Setup counter to remove elements in case of min or max being set
    counter = 0
    for idx in range(data.size):
        # Check if its inside the range we set
        if data[idx] < min or data[idx] > max:
            counter += 1
            continue
        # Calculate the index for indices and histogram
        bin_i = int(n * (data[idx] - bin_min) / (bin_max - bin_min))
        # Record the current index and update the tracking array
        ri[tracker[bin_i]] = idx
        tracker[bin_i] += 1
    ri = ri[:ri.size - counter]
    return ri

def histogram(data, bin_size = 1, num_bins = None, min = None, max = None):
    """
    The histogram function combines the use of the get_bins, get_hist and get_ri
    functions into one function. For the descriptions and docs of those functions
    look in See Also. This function will return the histogram, bin/bin_edges and
    the reverse indices.

    Parameters
    ----------
    data : np.array
        A NumPy array containing the data we want a histogram of
    bin_size : int, optional
        Sets the bin size for this histogram, by default 1
    num_bins : int, optional
        Set the number of bins this does override bin_size completely, by default None
    min : int, float, optional
        Set a minimum for the dataset, by default None
    max : int, float, optional
        Set a maximum for the dataset, by default None

    Returns
    -------
    hist : np.ndarray
        The histogram
    bins : np.ndarray
        The bins used for the histogram
    ri : np.ndarray
        The reverse indices array

    See Also
    --------
    get_bins: Calculates the bins only
    get_hist: Calculates histogram only
    get_ri: Calculates the reverse indices only
    """

     # If the minimum has not been set, set it
    if min is None:
        min = np.min(data)
    # If the maximum has not been set, set it
    if max is None:
        max = np.max(data)
    # If the number of bins has been set use that
    if num_bins is not None:
        bin_size = (max - min) / num_bins
    # IDL uses the bin_size as equal throughout min to max
    bins = get_bins(min, max, bin_size)
    # However, if we set a max, we must adjust the last bin to max according to IDL specifications
    if bins[-1] > max or num_bins is not None:
        bins = bins[:-1]
    # Flatten the data
    data_flat = data.flatten()
    # Get the histogram
    hist = get_hist(data_flat, bins, min, max)
    # Get the reverse indices
    ri = get_ri(data_flat, bins, hist, min, max)
    # Return
    return hist, bins, ri

def l_m_n(obs, psf, obsdec = None, obsra = None,  declination_arr = None, right_ascension_arr = None) :
    """
    Calculates the l mode, m mode and the n_tracked
    TODO: Add Detailed Description of l_m_n

    Parameters
    ----------
    obs: dict

    psf: dict

    obsdec: array, optional
        By default is set to None, as such by default this value will be set to
        obs['obsdec']
    obsra: array, optional
        By default is set to None, as such by default this value will be set to
        obs['obsra']
    declination_arr: array, optional
        By default is set to None, as such by default this value will be set to
        psf['image_info']['dec_arr']
    right_ascension_arr: array, optional
        By default is set to None, as such by default this value will be set to
        psf['image_info']['ra_arr']

    Returns
    -------
    l_mode: array
        TODO: Add description for l_mode
    m_mode: array
        TODO: Add description for m_mode
    n_tracked: array
        TODO: Add description for n_tracked.
    """
    # If the variables passed through are None them
    if obsdec is None:
        obsdec = obs['obsdec']
    if obsra is None:
        obsra = obs['obsra']
    if  declination_arr is None:
        declination_arr = psf['image_info'][0]['dec_arr'][0]
    if right_ascension_arr  is None:
        right_ascension_arr = psf['image_info'][0]['ra_arr'][0]

    # Convert all the degrees given into radians
    obsdec = np.radians(obsdec)
    obsra = np.radians(obsra)
    declination_arr = np.radians(declination_arr)
    right_ascension_arr = np.radians(right_ascension_arr)

    # Calculate l mode, m mode and the phase-tracked n mode of pixel centers
    cdec0 = np.cos(obsdec)
    sdec0 = np.sin(obsdec)
    cdec = np.cos(declination_arr)
    sdec = np.sin(declination_arr)
    cdra = np.cos(right_ascension_arr - obsra)
    sdra = np.sin(right_ascension_arr - obsra)
    l_mode = sdra * cdec
    m_mode = cdec0 * sdec - cdec * sdec0 * cdra
    # n=1 at phase center, so reference from there for phase tracking
    n_tracked = (sdec * sdec0 + cdec * cdec0 * cdra) - 1

    # find any NaN values
    nan_vals = np.where(np.isnan(n_tracked))
    # If any found, replace them with 0's
    if np.size(nan_vals) > 0:
        n_tracked[nan_vals] = 0
        l_mode[nan_vals] = 0
        m_mode[nan_vals] = 0
    
    # Return the modes
    return l_mode, m_mode, n_tracked

def rebin_columns(a, ax, shape, col_sizer):
    # tile the range of col_sizer
    tiles = np.tile(np.arange(col_sizer), (shape[0], shape[1] // col_sizer-1))
    # Get the differences between the columns
    differences = np.diff(a, axis = ax) / col_sizer
    # Multiply this by the tiles
    inferences_non_pad = np.repeat(differences, col_sizer, axis = ax) * tiles
    # Pad the zeros for the last two rows, and remove the extra zeros to make inferences same shape as desired shape
    inferences = np.pad(inferences_non_pad, (0,col_sizer))[:-col_sizer]
    if np.issubdtype(a.dtype, np.integer):
        inferences = np.floor(inferences).astype("int")
    # Now get our final array by adding the repeat of our rows rebinned to the inferences
    rebinned = inferences + np.repeat(a, col_sizer, axis = ax)
    return rebinned

def rebin_rows(a, ax, shape, old_shape, row_sizer):
    # Tile the range of row_sizer
    tiles = np.tile(np.array_split(np.arange(row_sizer), row_sizer), ((shape[0]- row_sizer) // row_sizer, old_shape[1]))
    # Get the differences between values
    differences = np.diff(a, axis = ax) / row_sizer
    # Multiply differences array by tiles to get desired bins
    inferences_non_pad = np.repeat(differences, row_sizer, axis = ax) * tiles
    # Pad the inferences to get the same shape as above
    inferences = np.pad(inferences_non_pad, (0,row_sizer))[:,:-row_sizer]
    if np.issubdtype(a.dtype, np.integer):
        inferences = np.floor(inferences).astype("int")
    # Add this to the original array that has been repeated to match the size of inference
    row_rebinned = inferences + np.repeat(a, row_sizer, axis = ax)
    return row_rebinned


def rebin(a, shape, sample = False):
    """
    Resizes a 2d array by averaging or repeating elements, new dimensions must be integral factors of original dimensions.

    In the case of expanding an existing array, rebin will interpolate between the original values with a linear function.
    In the case of compressing an existing array, rebin will average

    Parameters
    ----------
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array in (rows, columns)
        Must be a factor or multiple of a.shape
    sample: bool optional
        Use this to get samples using rebin, rather than interpolation, by default False.
        
    Returns
    -------
    rebinned : ndarray
        If the new shape is smaller of the input array, the data are averaged, 
        if the new shape is bigger array elements are repeated and interpolated

    Examples
    --------
    >>> test = np.array([0,10,20,30])
    >>> rebin(test, (1,8)) # Expand Columns
    array([ 0,  5, 10, 15, 20, 25, 30, 30])
    >>> rebin(test, (2,8)) # Expand Rows and Columns
    array([[ 0,  5, 10, 15, 20, 25, 30, 30],
           [ 0,  5, 10, 15, 20, 25, 30, 30]])
    >>> data = np.array([[ -5,   4,   2,  -8,   1], 
                         [  3,   0,   5,  -5,   1], 
                         [  6,  -7,   4,  -4,  -8],
                         [ -1,  -5, -14,   2,   1]])
    >>> rebin(data, (8,10)) # 2D Array example
    array([[ -5,  -1,   4,   3,   2,  -3,  -8,  -4,   1,   1],
           [ -1,   0,   2,   2,   3,  -2,  -7,  -3,   1,   1],
           [  3,   1,   0,   2,   5,   0,  -5,  -2,   1,   1],
           [  4,   0,  -4,   0,   4,  -1,  -5,  -5,  -4,  -4],
           [  6,  -1,  -7,  -2,   4,   0,  -4,  -6,  -8,  -8],
           [  2,  -2,  -6,  -6,  -5,  -3,  -1,  -3,  -4,  -4],
           [ -1,  -3,  -5, -10, -14,  -6,   2,   1,   1,   1],
           [ -1,  -3,  -5, -10, -14,  -6,   2,   1,   1,   1]])
    >>> rebin(data, (2,5)) # Compression
    array([[-1,  2,  3, -6,  1],
           [ 2, -6, -5, -1, -3]])
    >>> to_compress = np.array([[3, 9, 7, 0, 1, 5],
                                [7, 7, 1, 9, 7, 3],
                                [9, 2, 2, 3, 1, 1],
                                [0, 3, 5, 0, 4, 3],
                                [5, 7, 7, 1, 9, 1],
                                [7, 2, 1, 1, 3, 0]])
    >>> rebin(to_compress, (3,3)) # Compression
    array([[6, 4, 4],
           [3, 2, 2],
           [5, 2, 3]])

    References
    ----------
    [1] https://stackoverflow.com/a/8090605
    """
    old_shape  = a.shape
    # Prevent more processing than needed if we want the same shape
    if old_shape == shape:
        return a
    if len(old_shape) == 1:
        old_shape = (1,old_shape[0])
    # Sample the original array using rebin
    if sample:
        # If its a 1D array then...
        if old_shape[0] == 1:
            if shape[1] > old_shape[1]:
                rebinned = np.repeat(a, shape[1] // old_shape[1], axis = 0)
            else:
                rebinned = a[::old_shape[1] // shape[1]]
        # Assume its a 2D array
        else:
            # Do the Rows first
            if shape[0] >= old_shape[0]:
                # Expand Rows
                rebinned = np.repeat(a, shape[0] // old_shape[0], axis = 0)
            else:
                # Compress Rows
                rebinned = a[::old_shape[0] // shape[0], :]
            # Then do the columns
            if shape[1] >= old_shape[1]:
                # Expand Columns
                rebinned = np.repeat(rebinned, shape[1] // old_shape[1], axis = 1)
            else:
                # Compress columns
                rebinned = rebinned[:, ::old_shape[1] // shape[1]]
        # Return the rebinned without adjusting dtype as none of the functions above change it
        return rebinned

    # If we are downsizing
    elif shape[0] < old_shape[0] or shape[1] < old_shape[1]:
        if (max(old_shape[0],shape[0]) % min(old_shape[0],shape[0]) != 0) or \
           (max(old_shape[1],shape[1]) % min(old_shape[1],shape[1]) != 0):
            raise ValueError("Your new shape should be a factor of the original shape")
        # If we are increasing the rows or columns and reducing the other, increase them now and change the old shape
        if shape[0] > old_shape[0]:
            a = np.tile(a, (shape[0], 1))
            old_shape = a.shape
        elif shape[1] > old_shape[1]:
            a = np.tile(a, (1, shape[1]))
            old_shape = a.shape
        # Create the shape we need (rows, rows that can fit in old_shape, cols, cols that can fit into old_shape)    
        sh = shape[0], old_shape[0] // shape[0], shape[1], old_shape[1] // shape[1]
        # Create the 4D array
        rebinned = np.reshape(a, sh)
        # Get the average of the columns first
        rebinned = rebinned.mean(-1)
        # To ensure that we get the result same as IDL
        # it seems to fix the values after every calculation if integer
        if np.issubdtype(a.dtype, np.integer):
            rebinned = np.fix(rebinned).astype("int")
        # Now get it for the rows
        rebinned = rebinned.mean(1)
        # If we are expecting 1D array ensure it gets returned as a 1D array
        if (shape[0] == 1):
            rebinned = rebinned[0]
        # To ensure that we get the result same as IDL 
        # it seems to fix the values after every calculation if integer
        if np.issubdtype(a.dtype, np.integer):
            rebinned = np.fix(rebinned).astype("int")

    # Otherwise we are expanding
    else:
        if shape[0] % old_shape[0] != 0 or shape[1] % old_shape[1] != 0:
            raise ValueError("Your new shape should be a multiple of the original shape")
        # Get the size changes of the row and column separately
        row_sizer = shape[0] // old_shape[0]
        col_sizer = shape[1] // old_shape[1]
        ax = 0
        # If 1D array then do along the columns
        if old_shape[0] == 1:
            rebinned = rebin_columns(a, ax, shape, col_sizer)
            if shape[0] == 1:
                rebinned = rebinned[0]
        # Else its a 2D array
        else:
            row_rebinned = rebin_rows(a, ax, shape, old_shape, row_sizer)
            # If it matches the new shape, then return it
            if row_rebinned.shape == shape:
                return row_rebinned
            else:
                ax = 1
                rebinned = rebin_columns(row_rebinned, ax, shape, col_sizer)
    return rebinned

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

    Returns
    -------
    result: array
        The weights array that has had NaNs and Infinities removed, and zeros OR
        values that don't meet the threshold.
    """

    result = np.zeros_like(weights)
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

def meshgrid(dimension, elements, axis = None, return_integer = False):
    """
    Generates a 2D array of X or Y values. Could be replaced by a another function 

    Parameters
    ----------
    dimension: int
        Sets the column size of the array to return
    elements: int
        Sets the row size of the array to return
    axis = int, optional
        Set axis = 1 for X values, set axis = 2 for Y values, default is None
    return_float: bool, optional
        The default is False, dtype is implied by dimension and/or elements
        by default. If True, sets return array to float.
    
    Returns
    -------
    result: ndarray
        A numpy array of shape (elements, dimension) that is 
        a modified np.arange(elements * dimension).
    """
    if axis is None:
        axis = elements
        elements = dimension
    # If elements is set as 2
    if axis == 2:
        # Replicate LINDGEN by doing arange and resizing, result is floored
        result = np.reshape(np.floor(np.arange(elements * dimension) / dimension), (elements, dimension))
    else:
        # Replicate LINDGEN by doing arange and resizing 
        result = np.reshape(np.arange(elements * dimension) % dimension, (elements, dimension))
    # If we need to return a float, then set the result and return
    if return_integer:
        return result.astype("int")
    # Otherwise return as is
    else:
        return result

def deriv_coefficients(n, divide_factorial = False):
    """
    Computes an array of coefficients resulting in taking the 
    n-th derivative of a function of the form x^a (a must not 
    be a positive integer less than n)

    Parameters
    ----------
    n: int
        Decides the length of coefficients
    divide_factorial: bool
        Determine if we need to divide by the factorial

    Returns
    -------
    coeff: ndarray
        An array of coefficients
    """
    if n <= 0:
        return 0
    # Set up the array
    coeff = np.zeros(n)
    # Set the first coefficient to 1
    coeff[0] = 1
    # For every coefficient
    for m in range(1, n):
        # Had to do m + 1 for the range as IDL coeff[1:1] == coeff[1], but Python coeff[1:1] == array([])
        coeff[1 : m + 1] += -m * coeff[0 : m]
    # If we are to divide by the factorial do that to each coefficient
    if divide_factorial:
        for m in range(n):
            coeff[m] /= factorial(m + 1)
    
    # Return coefficients
    return coeff

def idl_argunique(arr : np.ndarray) -> np.ndarray:
    """
    In IDL the UNIQ function returns the indexes of the unique values within
    an array (that is assumed to be sorted beforehand). In NumPy they use the first
    unique index when using return index, where as in IDL they use the last index.

    Parameters
    ----------
    arr : np.ndarray
        A sorted numpy array of any type.

    Returns
    -------
    np.ndarray
        An array containing indexes of the last occurence of the unique value in arr

    Examples
    --------
    >>> test = np.array([10, 30, 5, 100, 200, 75, 200, 100, 30])
    >>> idl_argunique(test)
    array([0, 1, 3, 4, 6, 8])
    """
    return np.searchsorted(arr, np.unique(arr), side = 'right') - 1

def angle_difference(ra1 : float, dec1 : float, ra2 : float, dec2 : float, degree = False, nearest = False) -> float:
    """
    Calculates the angle difference between two given celestial coordinates.

    Parameters
    ----------
    ra1 : float
        Right Ascension of a coordinate 1 in radians or degrees
    dec1 : float
        Declination of a coordinate 1 in radians or degrees
    ra2 : float
        Right Ascension of a coordinate 2 in radians or degrees
    dec2 : float
        Declination of a coordinate 2 in radians or degrees
    degree : bool, optional
        If True, then we treate the coordinates given as degrees, by default False
    nearest : bool, optional
        Calculate implied angle instead, by default False

    Returns
    -------
    relative_angle : float
        The angle difference in degrees
    """
 
    if degree:
        unit = u.deg
    else:
        unit = u.rad
    coord1 = SkyCoord(ra = ra1*unit, dec = dec1*unit)
    coord2 = SkyCoord(ra = ra2*unit, dec = dec2*unit)
    # Use built in astropy separtion instead of calculating it
    relative_angle = coord1.separation(coord2)
    if nearest:
        return max(relative_angle, 2 * pi - relative_angle)
    else:
        return relative_angle.value

def parallactic_angle(latitude : float, hour_angle : float, dec : float) -> float:
    """
    Calculates the parallactic angle given latitude (usually a declination), hour_angle and another declination

    Parameters
    ----------
    latitude : float
        An angle in degrees, usually a declination in this package
    hour_angle : float
        The hour angle in degrees
    dec : float
        A declination in degrees

    Returns
    -------
    parallactic_angle : float
        The angle between the great circle through a celestial object and the zenith, and the hour circle of the object
    """

    y_term = np.sin(np.radians(hour_angle))
    x_term = np.cos(np.radians(dec)) * np.tan(np.radians(latitude)) - np.sin(np.radians(dec)) * np.cos(np.radians(hour_angle))
    return np.degrees(np.arctan(y_term/ x_term))
