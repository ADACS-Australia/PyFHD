import numpy as np

def interpolate_kernel(kernel_arr, x_offset, y_offset, dx0dy0, dx1dy0, dx0dy1, dx1dy1):
    """
    TODO: Description

    Parameters
    ----------
    kernel_arr: array
        The array we are applying the kernel too
    x_offset: array
        x_offset array, which will now technically do the y.
        Will likely change the name of this
    y_offset: array
        y_offset array, which will now technically do the x
        Will likely change the name of this
    dx0dy0: array
        TODO: description
    dx1dy0: array
        TODO: Description
    dx0dy1: array
        TODO: Description
    dx1dy1: array
        TODO: Description

    Returns
    -------
    kernel: array
        TODO: Description
    """
    # x_offset and y_offset needed to be swapped around as IDL is column-major, while Python is row-major
    kernel = kernel_arr[y_offset, x_offset] * dx0dy0
    kernel += kernel_arr[y_offset, x_offset + 1] * dx1dy0
    kernel += kernel_arr[y_offset + 1, x_offset] * dx0dy1
    kernel += kernel_arr[y_offset + 1, x_offset + 1] * dx1dy1

    return kernel