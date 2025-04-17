import numpy as np
import matplotlib.pyplot as plt
import os


def quick_image(
    image,
    xvals=None,
    yvals=None,
    data_range=None,
    data_min_abs=None,
    xrange=None,
    yrange=None,
    data_aspect=None,
    log=False,
    color_profile="log_cut",
    xtitle=None,
    ytitle=None,
    title=None,
    cb_title=None,
    note=None,
    charsize=None,
    xlog=False,
    ylog=False,
    window_num=1,
    multi_pos=None,
    start_multi_params=None,
    alpha=None,
    missing_value=None,
    savefile=None,
    png=False,
    eps=False,
    pdf=False,
):
    """
    Python translation of the IDL quick_image procedure with additional features.
    

    Parameters
    ----------
    image : _type_
        _description_
    xvals : _type_, optional
        _description_, by default None
    yvals : _type_, optional
        _description_, by default None
    data_range : _type_, optional
        _description_, by default None
    data_min_abs : _type_, optional
        _description_, by default None
    xrange : _type_, optional
        _description_, by default None
    yrange : _type_, optional
        _description_, by default None
    data_aspect : _type_, optional
        _description_, by default None
    log : bool, optional
        _description_, by default False
    color_profile : str, optional
        _description_, by default "log_cut"
    xtitle : _type_, optional
        _description_, by default None
    ytitle : _type_, optional
        _description_, by default None
    title : _type_, optional
        _description_, by default None
    cb_title : _type_, optional
        _description_, by default None
    note : _type_, optional
        _description_, by default None
    charsize : _type_, optional
        _description_, by default None
    xlog : bool, optional
        _description_, by default False
    ylog : bool, optional
        _description_, by default False
    window_num : int, optional
        _description_, by default 1
    multi_pos : _type_, optional
        _description_, by default None
    start_multi_params : _type_, optional
        _description_, by default None
    alpha : _type_, optional
        _description_, by default None
    missing_value : _type_, optional
        _description_, by default None
    savefile : _type_, optional
        _description_, by default None
    png : bool, optional
        _description_, by default False
    eps : bool, optional
        _description_, by default False
    pdf : bool, optional
        _description_, by default False

    Raises
    ------
    ValueError
        _description_
    """      
    # Determine if the output is to be saved to disk
    pub = bool(savefile or png or eps or pdf)

    # Handle file extension and output format
    if pub:
        if not (png or eps or pdf):
            if savefile:
                _, extension = os.path.splitext(savefile)
                extension = extension.lower()
                if extension == ".eps":
                    eps = True
                elif extension == ".png":
                    png = True
                elif extension == ".pdf":
                    pdf = True
                else:
                    print("Unrecognized extension, using PNG")
                    png = True

        # Set default savefile if not provided
        if not savefile:
            savefile = "idl_quick_image"
            print(
                f"No filename specified for quick_image output. Using {os.getcwd()}/{savefile}"
            )

        # Ensure only one output format is set
        formats_set = sum([png, eps, pdf])
        if formats_set > 1:
            print("Only one of eps, png, pdf can be set. Defaulting to PNG.")
            eps = pdf = False
            png = True

        # Append the appropriate file extension
        if png:
            savefile += ".png"
        elif pdf:
            savefile += ".pdf"
        elif eps:
            savefile += ".eps"

    # Validate the image input
    if image is None or not isinstance(image, np.ndarray):
        print("Image is undefined or not a valid numpy array.")
        return

    # Ensure the image is 2D
    if image.ndim != 2:
        print("Image must be 2-dimensional.")
        return

    # Handle complex images. Default is to show the real part.
    if np.iscomplexobj(image):
        print("Image is complex, showing real part.")
        image = np.real(image)

    # Handle missing values by setting them to NaN
    if missing_value is not None:
        wh_missing = np.where(image == missing_value)
        count_missing = len(wh_missing[0])
        if count_missing > 0:
            image[wh_missing] = np.nan
            missing_color = 0
    else:
        count_missing = 0
        wh_missing = None
        missing_color = None

    # Apply logarithmic scaling if set. This modifies the image input directly
    # to be logarithmically scaled in the color bar range.
    if log:
        image, cb_ticks, cb_ticknames = log_color_calc(
            data=image,
            data_range=data_range,
            color_profile=color_profile,
            log_cut_val=None,
            min_abs=data_min_abs,
            count_missing=count_missing,
            wh_missing=wh_missing,
            missing_color=missing_color,
            invert_colorbar=False,
        )
    else:
        # Apply linear scaling by default. This modifies the image input directly
        # to be linearly scaled in the color bar range.
        if data_range is None:
            data_range = [np.nanmin(image), np.nanmax(image)]

        data_color_range, data_n_colors = color_range(
            image, count_missing=count_missing
        )

        # Scale image data to be in the color range
        image = (image - data_range[0]) * (data_n_colors - 1) / (
            data_range[1] - data_range[0]
        ) + data_color_range[0]
        print(data_range, data_color_range, data_n_colors)

        # Handle out-of-bounds values
        wh_low = np.where(image < data_range[0])
        if len(wh_low[0]) > 0:
            image[wh_low] = data_color_range[0]
        wh_high = np.where(image > data_range[1])
        if len(wh_high[0]) > 0:
            image[wh_high] = data_color_range[1]

        # Handle missing values
        if missing_value is not None and count_missing > 0:
            image[wh_missing] = missing_color

        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{tick * (data_range[1] - data_range[0]) / (data_n_colors - 1) + data_range[0]:.2g}"
            for tick in cb_ticks
        ]
        print(cb_ticks, cb_ticknames)

    # Set up the plot
    fig, ax = plt.subplots(num=window_num)  # Use window_num to manage figure windows
    cmap = plt.get_cmap("viridis")

    # Set up the x and y ranges
    extent = None
    if xvals is not None and yvals is not None:
        extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]
    elif xrange is not None and yrange is not None:
        extent = [xrange[0], xrange[1], yrange[0], yrange[1]]

    im = ax.imshow(
        image,
        extent=extent,
        aspect=data_aspect or "auto",
        cmap=cmap,
        vmin=0,
        vmax=255,
        alpha=alpha,
    )

    # Add titles and labels
    if title:
        ax.set_title(title, fontsize=charsize or 12)
    if xtitle:
        ax.set_xlabel(xtitle, fontsize=charsize or 10)
    if ytitle:
        ax.set_ylabel(ytitle, fontsize=charsize or 10)

    # Handle logarithmic axes
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    # if log:
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels(cb_ticknames)
    if cb_title:
        cbar.set_label(cb_title, fontsize=charsize or 10)

    # Add note if provided
    if note:
        plt.figtext(
            0.99, 0.02, note, horizontalalignment="right", fontsize=charsize or 8
        )

    # Multi-panel plotting
    if multi_pos is not None:
        if len(multi_pos) != 4:
            raise ValueError(
                "multi_pos must be a 4-element list defining the plot position."
            )
        ax.set_position(multi_pos)

    # Handle start_multi_params for multi-panel layout
    if start_multi_params is not None:
        nrows = start_multi_params.get("nrow", 1)
        ncols = start_multi_params.get("ncol", 1)
        index = start_multi_params.get("index", 1) - 1  # Convert to 0-based index
        ax.set_position(
            [
                (index % ncols) / ncols,
                1 - (index // ncols + 1) / nrows,
                1 / ncols,
                1 / nrows,
            ]
        )

    # Save or show the plot
    if pub:
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
        print(f"Image saved to {savefile}")
    else:
        plt.show()

    plt.close(fig)


def log_color_calc(
    data,
    data_range=None,
    color_profile="log_cut",
    log_cut_val=None,
    min_abs=None,
    count_missing=None,
    wh_missing=None,
    missing_color=None,
    invert_colorbar=False,
):
    """
    Translated version of log_color_calc from IDL to Python.

    Parameters
    ----------
    data : _type_
        _description_
    data_range : _type_, optional
        _description_, by default None
    color_profile : str, optional
        _description_, by default "log_cut"
    log_cut_val : _type_, optional
        _description_, by default None
    min_abs : _type_, optional
        _description_, by default None
    count_missing : _type_, optional
        _description_, by default None
    wh_missing : _type_, optional
        _description_, by default None
    missing_color : _type_, optional
        _description_, by default None
    invert_colorbar : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
   
    """
    # Define valid color profiles
    color_profile_enum = ["log_cut", "sym_log", "abs"]
    if color_profile not in color_profile_enum:
        raise ValueError(
            f"Color profile must be one of: {', '.join(color_profile_enum)}"
        )

    # Handle data_range
    if data_range is None:
        no_input_data_range = True
        data_range = [np.min(data), np.max(data)]
    else:
        if len(data_range) != 2:
            raise ValueError("data_range must be a 2-element vector")
        no_input_data_range = False

    if data_range[1] < data_range[0]:
        raise ValueError("data_range[0] must be less than data_range[1]")

    # Handle sym_log profile constraints
    if color_profile == "sym_log" and data_range[0] > 0:
        print(
            "sym_log profile cannot be selected with an entirely positive data range. Switching to log_cut"
        )
        color_profile = "log_cut"

    data_color_range, data_n_colors = color_range(data, count_missing=count_missing)

    # Handle positive values
    wh_pos = np.where(data > 0)
    count_pos = len(wh_pos[0])
    if count_pos > 0:
        min_pos = np.min(data[wh_pos])
    elif data_range[0] > 0:
        min_pos = data_range[0]
    elif data_range[1] > 0:
        min_pos = data_range[1] / 10
    else:
        min_pos = 0.01

    # Handle negative values
    wh_neg = np.where(data < 0)
    count_neg = len(wh_neg[0])
    if count_neg > 0:
        max_neg = np.max(data[wh_neg])
    elif data_range[1] < 0:
        max_neg = data_range[1]
    else:
        max_neg = data_range[0] / 10

    # Handle zero values
    wh_zero = np.where(data == 0)
    count_zero = len(wh_zero[0])

    # Handle log_cut color profile
    if color_profile == "log_cut":
        if data_range[1] < 0:
            raise ValueError(
                "log_cut color profile will not work for entirely negative arrays."
            )

        if log_cut_val is None:
            if data_range[0] > 0:
                log_cut_val = np.log10(data_range[0])
            else:
                log_cut_val = np.log10(min_pos)

        log_data_range = [log_cut_val, np.log10(data_range[1])]

        # Handle zero values
        if count_zero > 0:
            min_pos_color = 2
            zero_color = 1
            zero_val = log_data_range[0]
        else:
            min_pos_color = 1

        neg_color = 0
        neg_val = log_data_range[0]

        data_log = np.log10(data)
        wh_under = np.where(data < 10**log_cut_val)
        if len(wh_under[0]) > 0:
            data_log[wh_under] = log_data_range[0]

        wh_over = np.where(data_log > log_data_range[1])
        if len(wh_over[0]) > 0:
            data_log[wh_over] = log_data_range[1]

        # Normalize data
        data_log_norm = (
            (data_log - log_data_range[0])
            * (data_n_colors - min_pos_color - 1)
            / (log_data_range[1] - log_data_range[0])
            + data_color_range[0]
            + min_pos_color
        )

        if count_neg > 0:
            data_log_norm[wh_neg] = neg_color
        if count_zero > 0:
            data_log_norm[wh_zero] = zero_color

    elif color_profile == "sym_log":
        if data_range[1] < 0:
            raise ValueError(
                "sym_log color profile will not work for entirely negative arrays."
            )

        # Calculate the minimum absolute value
        if min_abs is None:
            if count_pos > 0 and count_neg > 0:
                min_abs = min(min_pos, abs(max_neg))
            elif count_pos > 0:
                min_abs = min_pos
            elif count_neg > 0:
                min_abs = abs(max_neg)
            else:
                min_abs = 1.0

        log_data_range = [np.log10(min_abs), np.log10(data_range[1])]

        # Normalize data
        data_log_norm = np.zeros_like(data, dtype=float)
        wh_pos = np.where(data > 0)
        wh_neg = np.where(data < 0)
        wh_zero = np.where(data == 0)

        if len(wh_pos[0]) > 0:
            data_log_norm[wh_pos] = (np.log10(data[wh_pos]) - log_data_range[0]) * (
                data_n_colors - 1
            ) / (log_data_range[1] - log_data_range[0]) + data_color_range[0]

        if len(wh_neg[0]) > 0:
            data_log_norm[wh_neg] = (
                np.log10(abs(data[wh_neg])) - log_data_range[0]
            ) * (data_n_colors - 1) / (
                log_data_range[1] - log_data_range[0]
            ) + data_color_range[
                0
            ]

        if len(wh_zero[0]) > 0:
            data_log_norm[wh_zero] = data_color_range[0]

        # Handle out-of-bounds values
        wh_under = np.where(data_log_norm < data_color_range[0])
        if len(wh_under[0]) > 0:
            data_log_norm[wh_under] = data_color_range[0]

        wh_over = np.where(data_log_norm > data_color_range[1])
        if len(wh_over[0]) > 0:
            data_log_norm[wh_over] = data_color_range[1]

    # Handle abs color profile
    elif color_profile == "abs":
        data_abs = np.abs(data)
        data_log_norm = (data_abs - data_range[0]) * (data_n_colors - 1) / (
            data_range[1] - data_range[0]
        ) + data_color_range[0]

        # Handle out-of-bounds values
        wh_under = np.where(data_log_norm < data_color_range[0])
        if len(wh_under[0]) > 0:
            data_log_norm[wh_under] = data_color_range[0]

        wh_over = np.where(data_log_norm > data_color_range[1])
        if len(wh_over[0]) > 0:
            data_log_norm[wh_over] = data_color_range[1]

    # Handle missing values
    if count_missing > 0:
        data_log_norm[wh_missing] = missing_color

    # Handle invert_colorbar option
    if invert_colorbar:
        data_log_norm = data_color_range[1] - (data_log_norm - data_color_range[0])

    # Generate colorbar ticks and tick names
    if color_profile == "log_cut" or color_profile == "sym_log":
        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{10**(tick * (log_data_range[1] - log_data_range[0]) / (data_n_colors - 1) + log_data_range[0]):.2g}"
            for tick in cb_ticks
        ]
    elif color_profile == "abs":
        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{tick * (data_range[1] - data_range[0]) / (data_n_colors - 1) + data_range[0]:.2g}"
            for tick in cb_ticks
        ]

    return data_log_norm, cb_ticks, cb_ticknames


def color_range(data, count_missing=None):
    """
    _summary_

    Parameters
    ----------
    data : _type_
        _description_
    count_missing : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """    

    # Initialize color range
    color_range = [0, 255]
    if count_missing > 0:
        data_color_range = [1, 255]
    else:
        data_color_range = color_range

    n_colors = color_range[1] - color_range[0] + 1
    data_n_colors = data_color_range[1] - data_color_range[0] + 1

    return data_color_range, data_n_colors
