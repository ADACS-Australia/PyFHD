import numpy as np
import pytest
import re
from scipy.signal import convolve2d
from pathlib import Path

from pyfhd.plotting.image import quick_image


@pytest.fixture
def pyramid():
    top_hat_arr = np.zeros((12, 12), dtype=float)
    # set middle to 1, leave zero edges
    top_hat_arr[1:-1, 1:-1] = np.ones((10, 10), dtype=float)

    pyramid = convolve2d(top_hat_arr, top_hat_arr)
    yield pyramid

@pytest.fixture
def quick_image_defaults(tmp_path):
    """Default keyword arguments for quick_image."""
    return dict(
        xvals=None,
        yvals=None,
        xrange=None,
        yrange=None,
        cmap="idl",
        log=False,
        missing_value=None,
        color_profile="log_cut",
        data_range=None,
        title="pyramid",
        xtitle="East (m)",
        ytitle="North (m)",
        cb_title="Height (m)",
        note=None,
        sigma_clip_level=None,
        percentile_clip_level=None,
        savefile=tmp_path / "output.png",
    )


@pytest.mark.github_actions
@pytest.mark.parametrize("file_type", ["png", "eps", "pdf"])
@pytest.mark.parametrize("file_is_path", [True, False])
def test_quick_image_pyramid(tmp_path, pyramid, file_type, file_is_path):
    """This is just a smoke test to make sure the code runs."""

    savefile = tmp_path / f"pyramid.{file_type}"
    cmap = "idl"
    missing_value = None
    log = False
    color_profile = "log_cut"
    data_range = None
    xvals = None
    yvals = None
    xrange = None
    yrange = None
    title = "pyramid"
    xtitle = "East (m)"
    ytitle = "North (m)"
    cb_title = "Height (m)"
    note = None
    sigma_clip_level = None
    percentile_clip_level = None

    if not file_is_path:
        savefile = str(savefile)
        # set parameters differently to access different parts of the code
        cmap = "magma"
        log = True
        pyramid_shape = pyramid.shape
        xvals = np.arange(pyramid_shape[0])
        yvals = np.arange(pyramid_shape[1])

    # set parameters differently to access different parts of the code
    if file_type == "pdf":
        title = None
        xtitle = None
        ytitle = None
        cb_title = None
        note = "foo"
        xrange = np.array([1, 21])
        yrange = np.array([1, 21])
        missing_value = 0
        sigma_clip_level = 3
        percentile_clip_level = 1
        color_profile = "abs"
    elif file_type == "eps":
        pyramid_max = pyramid.max()
        nonzero_min = np.min(pyramid[pyramid > 0])
        cmap = None
        color_profile = "sym_log"
        if log:
            data_range = np.array([-1 * pyramid_max, pyramid_max])
        else:
            data_range = np.array([nonzero_min, pyramid_max - nonzero_min])

    quick_image(
        pyramid,
        xvals=xvals,
        yvals=yvals,
        xrange=xrange,
        yrange=yrange,
        cmap=cmap,
        log=log,
        missing_value=missing_value,
        color_profile=color_profile,
        data_range=data_range,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        cb_title=cb_title,
        note=note,
        sigma_clip_level=sigma_clip_level,
        percentile_clip_level=percentile_clip_level,
        savefile=savefile,
    )

    assert Path(savefile).is_file()

@pytest.mark.github_actions
# @pytest.mark.skip(reason="TODO")
class TestValueErrors:
    class TestQuickImageValueErrors:
        # Raise ValueError if image is undefined or not a numpy array.
        # Run test with image as None and as a list.
        @pytest.mark.parametrize("image_input", [ None, [1, 1] ])
        def test_image_type(self, quick_image_defaults, image_input):
            args = { **quick_image_defaults }
            with pytest.raises(
                ValueError,
                match="Image is undefined or not a valid numpy array."
            ):
                quick_image(image_input, **args)
                
        # Raise ValueError if image is not 2-dimensional.
        # Run test on 1D, 3D, and 0D images.
        @pytest.mark.parametrize("dimensions", [(1), (1, 1, 1), ()])
        def test_invalid_image_dimensions(self, quick_image_defaults, dimensions):
            image = np.zeros(dimensions)
            args = { **quick_image_defaults } 
            with pytest.raises(ValueError, match="Image must be 2-dimensional."):
                quick_image(image, **args)

        # Raise ValueError if data_range, xrange, or yrange are not a numpy array
        # or list or if they are, they do not have exactly two values.
        @pytest.mark.parametrize("param", ["data_range", "xrange", "yrange"])
        @pytest.mark.parametrize("bad_range", [
            np.array([1]),          # right type, too short
            np.array([1, 2, 3]),    # right type, too long
            np.array([]),           # right type, empty
            [1],                    # right type, too short
            (1, 2),                 # wrong type (tuple)
            "12",                   # wrong type (string)
            12                      # wrong type (int)
        ])
        def test_ranges(
            self, pyramid, quick_image_defaults, param, bad_range
        ):
            args = { **quick_image_defaults, param: bad_range }
            with pytest.raises(
                ValueError,
                match=f"{param} must be an array with exactly two values."
            ):
                quick_image(pyramid, **args)

        # Raise ValueError if multi_pos does not have exactly 4 elements.
        @pytest.mark.parametrize("bad_list", [
            [ 1, 2, 3 ],        # too short
            [ 1, 2, 3, 4, 5 ],  # too long
            []                  # empty
        ])
        def test_multi_pos_values(self, pyramid, quick_image_defaults, bad_list):
            args = { **quick_image_defaults, "multi_pos": bad_list }
            with pytest.raises(
                ValueError,
                match="multi_pos must be a 4-element list defining the plot position."
            ):
                quick_image(pyramid, **args)

    class TestLogColorCalcValueErrors:
        # Raise ValueError if color_profile is not "log_cut", "sym_log", or "abs"
        def test_invalid_color_profile(self, pyramid, quick_image_defaults):
            args = {
                **quick_image_defaults, "color_profile": "invalid" , "log": True
            }
            color_profile_enum = ["log_cut", "sym_log", "abs"]
            with pytest.raises(
                ValueError,
                match=f"Color profile must be one of: {', '.join(color_profile_enum)}"
            ):
                quick_image(pyramid, **args)

        # Raise ValueError if data_range[0] is greater than data_range[1].
        def test_data_range_less_than(self, quick_image_defaults, pyramid):
            args = { **quick_image_defaults, "data_range": [2, 1], "log": True }
            with pytest.raises(
                ValueError,
                match=re.escape("data_range[0] must be less than data_range[1]")
            ):
                quick_image(pyramid, **args)

        # Raise ValueError if color_profile is log_cut and data_range is
        # entirely negative.
        def test_log_cut(self, pyramid, quick_image_defaults):
            args = {
                **quick_image_defaults, "log": True, "data_range": [-2, -1],
                "color_profile": "log_cut"
            }
            with pytest.raises(
                ValueError,
                match="log_cut color profile will not work for entirely negative arrays."
            ):
                quick_image(pyramid, **args)

        def test_sym_log(self, pyramid):
            # color_profile is sym_log and data_range[0] is positive or
            # data_range[1] is negative
            # color_profile is sym_log and data_range[0] is positive and
            # data_range[1] is negative
            pass