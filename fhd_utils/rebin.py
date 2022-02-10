import numpy as np

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