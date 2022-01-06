import numpy as np

def conjugate_mirror(image):
    """
    This takes a 2D array and mirrors it, shifts it and
    its an array of complex numbers its get the conjugates
    of the 2D array

    Parameters
    ----------
    image: array
        A 2D array of real or complex numbers
    
    Returns
    -------
    conj_mirror_image: array
        The mirrored and shifted image array
    """
    # Flip image left to right (i.e. flips columns) & Flip image up to down (i.e. flips rows)
    conj_mirror_image = np.flip(image)
    # Shifts columns then rows by 1
    conj_mirror_image = np.roll(conj_mirror_image ,  1, axis = (1,0))
    # If any of the array is complex, or its a complex array, get the conjugates
    if np.iscomplexobj(image):   
        conj_mirror_image = np.conjugate(conj_mirror_image)
    return conj_mirror_image