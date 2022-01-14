import numpy as np
from numba import njit, prange

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
    >>> data[ri[ri[0] : ri[1]]] = -1

    Or more generally

    >>> data[ri[ri[i] : ri[i + 1]]] = -1

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
    ri
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