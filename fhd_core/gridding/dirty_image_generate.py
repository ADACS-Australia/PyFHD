import numpy as np
from fhd_utils.modified_astro.meshgrid import meshgrid
from fhd_output.fft_filters.filter_uv_uniform import filter_uv_uniform
from fhd_utils.rebin import rebin
from scipy.signal import convolve
from astropy.convolution import Box2DKernel

def dirty_image_generate(dirty_image_uv, mask = None, baseline_threshold = 0, normalization = None,
                         resize = None, width_smooth = None, degpix = None, not_real = False,
                         image_filter_fn = 'filter_uv_uniform', pad_uv_image = None, filter = None,
                         vis_count = None, weights = None, beam_ptr = None, obs = None, psf = None, params = None, 
                         fi_use = None, bi_use = None):
    """[summary]

    Parameters
    ----------
    dirty_image_uv : [type]
        [description]
    mask : [type], optional
        [description], by default None
    baseline_threshold : int, optional
        [description], by default 0
    normalization : [type], optional
        [description], by default None
    resize : [type], optional
        [description], by default None
    width_smooth : [type], optional
        [description], by default None
    degpix : [type], optional
        [description], by default None
    real : bool, optional
        [description], by default False
    image_filter_fn : str, optional
        [description], by default 'filter_uv_uniform'
    pad_uv_image : [type], optional
        [description], by default None
    filter : [type], optional
        [description], by default None
    vis_count : [type], optional
        [description], by default None
    weights : [type], optional
        [description], by default None
    beam_ptr : [type], optional
        [description], by default None
    obs : [type], optional
        [description], by default None
    psf : [type], optional
        [description], by default None
    params : [type], optional
        [description], by default None
    fi_use : [type], optional
        [description], by default None
    bi_use : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """

    # dimension is columns, elements is rows
    elements, dimension = dirty_image_uv.shape
    di_uv_use = dirty_image_uv
    # If the baseline threshold has been set
    if baseline_threshold is not None:
        # If width smooth hasn't been set, set it
        if width_smooth is None:
            width_smooth = np.floor(np.sqrt(dimension * elements) / 100)    
        rarray = np.sqrt((meshgrid(dimension, 1) - dimension / 2) ** 2 + (meshgrid(elements, 2) - elements / 2) ** 2)
        # Get all the values that meet the threshold
        if baseline_threshold >= 0:
            cut_i = np.where(rarray.flatten() < baseline_threshold)
        else:
            cut_i = np.where(rarray.flatten() > np.abs(baseline_threshold))
        # Create the mask array of ones
        mask_bt = np.ones((elements, dimension))
        # If there are values from cut, then use all those here and replace with 0
        if np.size(cut_i) > 0:
            mask_bt_flatiter = mask_bt.flat
            mask_bt_flatiter[cut_i] = 0
        if width_smooth is not None:
            # Get the kernel width
            kernel_width = np.max([width_smooth,1])
            # In IDL if the kernel width is even one is added to make it odd
            if kernel_width % 2 == 0:
                kernel_width += 1
            # Use a box width averaging filter over the mask, use valid so we can insert it in later
            box_averages = convolve(mask_bt, Box2DKernel(kernel_width), mode = 'valid')
            # Since IDL SMOOTH edges by default are the edges of the array used, ignore edges (its the reason why we used a valid convolve)
            start = int(kernel_width // 2)
            end = int(mask_bt.shape[1] - (kernel_width // 2))
            mask_bt[start : end, start : end] = box_averages  
        # Apply boxed mask to the dirty image
        di_uv_use *=  mask_bt
    
    # If a mask was supplied use that too
    if mask is not None:
        di_uv_use *= mask
    
    # If a filter was supplied as a numpy array (we can adjust this to support different formats)
    if filter is not None:
        if isinstance(filter, np.ndarray):
            # If the filter is already the right size, use it
            if np.size(filter) == np.size(di_uv_use):
                di_uv_use *= filter
            # Otherwise use a filter function
            else:
                di_uv_use, _ = eval("{}(di_uv_use, vis_count = vis_count, obs = obs, psf = psf, params = params, weights = weights, fi_use = fi_use, bi_use = bi_use, mask_mirror_indices = mask_mirror_indices)".format(image_filter_fn))
    
    # Resize the dirty image by the factor resize    
    if resize is not None:
        dimension *= resize
        elements *= resize
        di_uv_real = di_uv_use.real
        di_uv_img = di_uv_use.imag
        # Use rebin to resize, apply to real and complex separately
        di_uv_real = rebin(di_uv_real, (elements, dimension))
        di_uv_img = rebin(di_uv_img, (elements, dimension))
        # Combine real and complex back together
        di_uv_use = di_uv_real + di_uv_img * 1j
    
    # Apply padding if it was supplied
    if pad_uv_image is not None:
        # dimension_new = int(np.max([np.max([dimension, elements]) * pad_uv_image, np.max([dimension, elements])]))
        # di_uv1 = np.zeros((dimension_new, dimension_new), dtype = "complex")
        # di_uv1[dimension_new // 2 - elements // 2 : dimension_new // 2 + elements // 2,
        #        dimension_new // 2 - dimension // 2 : dimension_new // 2 + dimension // 2] = di_uv_use
        di_uv1 = np.pad(di_uv_use, np.max([dimension, elements]) // 2)
        di_uv_use = di_uv1 * (pad_uv_image ** 2)
    
    # FFT normalization
    if degpix is not None:
        di_uv_use /= np.radians(degpix) ** 2

    # Multivariate Fast Fourier Transform
    dirty_image = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(di_uv_use), norm = "forward"))
    if not not_real:
        dirty_image = dirty_image.real
    
    # filter_uv_optimal produces images that are weighted by one factor of the beam
    # Weight by an additional factor of the beam to align with FHD's convention
    if image_filter_fn == 'filter_uv_optimal' and beam_ptr is not None:
        dirty_image *= beam_ptr
    
    # If we are returning complex, make sure its complex
    if not_real:
        dirty_image = dirty_image.astype("complex")
    else:
        dirty_image = dirty_image.real
    # Normalize by the matrix given, if it was given
    if normalization is not None:
        dirty_image *= normalization
        return dirty_image, normalization
    #Return
    return dirty_image  