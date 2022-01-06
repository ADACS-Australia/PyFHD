import numpy as np

def fft_shift(image):
    """
    This function shifts the pixels of an entire image.

    Parameters
    ----------
    image: ndarray
        A 2D array that represents an image
    
    Returns
    -------
    shifted: ndarray
        A 2D array that is the image shifted
    """
    # elements is rows, dimension is columns
    elements, dimension = image.shape
    # To get the same shift behaviour with np.roll as IDL's shift we do columns first
    shifted = np.roll(image, dimension // 2, axis = 1)
    # Then we do the rows
    shifted = np.roll(shifted, elements // 2, axis = 0)
    # Return the shifted image
    return shifted