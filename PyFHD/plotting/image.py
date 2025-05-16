import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
    General function to display and/or save a 2D data array as an image with an appropriately 
    scaled color bar.


    Parameters
    ----------
    image : NDArray[np.int_ | np.float_ | np.complex_]
        A 2D array of data to be displayed as an image. 
        The data can be of type int, float, or complex.
    xvals : NDArray[np.int_ | np.float_], optional
        An array of x-axis values, by default None
    yvals : NDArray[np.int_ | np.float_], optional
        An array of y-axis values, by default None
    data_range : NDArray[np.int_ | np.float_], optional
        Min/max color bar range, by default [np.nanmin(image), np.nanmax(image)]
    data_min_abs : _type_, optional
        The minimum absolute value for the color bar, by default None
    xrange : NDArray[np.int_ | np.float_], optional
        The indices (or xvals, if provided) to zoom the image, by default None
    yrange : NDArray[np.int_ | np.float_], optional
        The indices (or yvals, if provided) to zoom the image, by default None
    data_aspect : int | float, optional
        The aspect ratio of y to x, by default None
    log : bool, optional
        Color bar on logarithmic scale, by default False
    color_profile : str, optional
        Color bar profiles for logarithmic scaling. 
        "log_cut", "sym_log", "abs", by default "log_cut"
    xtitle : str, optional
        The title of the x-axis, by default None
    ytitle : str, optional
        The title of the x-axis, by default None
    title : str, optional
        The title of the image, by default None
    cb_title : str, optional
        The title of the color bar, by default None
    note : str, optional
        A small note to place on the bottom right of the image, by default None
    charsize : int, optional
        The size of the font, by default None
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
        Transparancy for the image, by default None
    missing_value : _type_, optional
        Exclude value from the color bar, by default None
    savefile : str, optional
        The save file name, by default None
    png : bool, optional
        Create a png of the image, by default False
    eps : bool, optional
        Create an eps of the image, by default False
    pdf : bool, optional
        Create a pdf of the image, by default False

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
                # Convert savefile to a Path object if it's a string
                savefile = Path(savefile) if isinstance(savefile, str) else savefile
                extension = savefile.suffix.lower()
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
        if isinstance(savefile, Path):
            if png:
                savefile = savefile.with_suffix(".png")
            elif pdf:
                savefile = savefile.with_suffix(".pdf")
            elif eps:
                savefile = savefile.with_suffix(".eps")
        elif isinstance(savefile, str):
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

    # Validate that 2-value inputs are only 2 values
    if data_range is not None:
        if not isinstance(data_range, np.ndarray) or len(data_range) != 2:
            raise ValueError("data_range must be an array with exactly two values.")
    if xrange is not None:
        if not isinstance(xrange, np.ndarray) or len(xrange) != 2:
            raise ValueError("xrange must be an array with exactly two values.")
    if yrange is not None:
        if not isinstance(yrange, np.ndarray) or len(yrange) != 2:
            raise ValueError("yrange must be an array with exactly two values.")

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
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")

    # Set up the x and y ranges
    extent = None
    if xvals is not None and yvals is not None:
        # Default extent based on full xvals and yvals
        extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]
        # Apply xrange to crop the image and adjust extent
        if xrange is not None:
            x_indices = np.logical_and(xvals >= xrange[0], xvals <= xrange[1])
            image = image[:, x_indices]
            xvals = xvals[x_indices]  # Update xvals to match cropped image
            extent[0], extent[1] = xrange[0], xrange[1]
        # Apply yrange to crop the image and adjust extent
        if yrange is not None:
            y_indices = np.logical_and(yvals >= yrange[0], yvals <= yrange[1])
            image = image[y_indices, :]
            yvals = yvals[y_indices]  # Update yvals to match cropped image
            extent[2], extent[3] = yrange[0], yrange[1]
    elif xrange is not None and yrange is not None:
        # If xvals and yvals are not provided, use xrange and yrange directly
        extent = [xrange[0], xrange[1], yrange[0], yrange[1]]
        image = image[np.ix_(yrange, xrange)]

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
    data : NDArray[np.int\_ | np.float\_ | np.complex\_]
        A 2D array of data to be displayed as an image. 
        The data can be of type int, float, or complex.
    data_range : NDArray[np.int\_ | np.float\_], optional
        Min/max color bar range, by default [np.nanmin(image), np.nanmax(image)]
    color_profile : str, optional
        Color bar profiles for logarithmic scaling. 
        "log_cut", "sym_log", "abs", by default "log_cut"
    log_cut_val : int | float, optional
        Minimum log value to cut at, by default None
    data_min_abs : int | float, optional
        The minimum absolute value for the color bar, by default None
    count_missing : int, optional
        The number of missing values, by default None
    wh_missing : int, optional
        The location of the missing values, by default None
    missing_color : int, optional
        The index of the color bar for missing values, by default None
    invert_colorbar : bool, optional
        Invert the color bar, by default False

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
        data_range = [np.nanmin(data), np.nanmax(data)]
    else:
        if len(data_range) != 2:
            raise ValueError("data_range must be a 2-element vector")
    
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
        min_pos = np.nanmin(data[wh_pos])
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
        max_neg = np.nanmax(data[wh_neg])
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
        if data_range[0] >= 0 or data_range[1] <= 0:
            raise ValueError("sym_log color profile requires both negative and positive values in data_range.")

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

        midpoint = (data_color_range[1] - data_color_range[0]) // 2

        if len(wh_pos[0]) > 0:
            data_log_norm[wh_pos] = (np.log10(data[wh_pos]) - log_data_range[0]) * (
                midpoint
            ) / (log_data_range[1] - log_data_range[0]) + data_color_range[0] + midpoint

        if len(wh_neg[0]) > 0:
            # Reverse the mapping for negative values
            data_log_norm[wh_neg] = data_color_range[0] + midpoint - (
                (np.log10(abs(data[wh_neg])) - log_data_range[0])
                * midpoint
                / (log_data_range[1] - log_data_range[0])
            )
    
        if len(wh_zero[0]) > 0:
            data_log_norm[wh_zero] = data_color_range[0] + midpoint

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
    if color_profile == "log_cut":
        cb_ticks = np.linspace(data_color_range[0], data_color_range[1], num=5)
        cb_ticknames = [
            f"{10**(tick * (log_data_range[1] - log_data_range[0]) / (data_n_colors - 1) + log_data_range[0]):.2g}"
            for tick in cb_ticks
        ]
    elif color_profile == "sym_log":
        pos_ticks = np.linspace(midpoint, data_color_range[1], num=5)
        neg_ticks = np.linspace(data_color_range[0], midpoint, num=5)
        cb_ticks = np.concatenate([neg_ticks, [midpoint], pos_ticks])
        cb_ticknames = (
            [f"-{10**(log_data_range[1] - (tick - data_color_range[0]) * (log_data_range[1] - log_data_range[0]) / midpoint):.2g}" for tick in neg_ticks]
            + ["0"]
            + [f"{10**((tick - midpoint) * (log_data_range[1] - log_data_range[0]) / midpoint + log_data_range[0]):.2g}" for tick in pos_ticks]
        )
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
