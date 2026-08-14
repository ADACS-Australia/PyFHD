import numpy as np
import pytest
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
@pytest.mark.skip(reason="TODO")
class TestValueErrors:
    class TestQuickImageValueErrors:
        def test_invalid_image_type(self, pyramid): 
            # image is none
            # image is not valid numpy array
            pass

        def test_invalid_image_dimensions(self, pyramid):
            # image has < 2 dimensions
            # image has > 2 dimensions
            pass

        def test_data_range_values(self, pyramid):
            # data range is not an array
            # data range has < 2 values
            # data range has > 2 values
            pass

        def test_xrange_values(self, pyramid):
            # xrange is not an array
            # xrange has < 2 values
            # xrange has > 2 values
            pass

        def test_yrange_values(self, pyramid):
            # yrange is not an array
            # yrange has < 2 values
            # yrange has > 2 values
            pass

        def test_multi_pos_values(self, pyramid):
            # multi_pos has < 4 elements
            # multi_pos has > 4 elements
            pass

        def test_invalid_color_profile(self, pyramid):
            # color_profile is not "log_cut", "sym_log", or "abs"
            pass

    class TestLogColorCalcValueErrors:
        def test_data_range_less_than(self, pyramid):
            # data_range[0] > data_range[1]
            pass

        def test_log_cut(self, pyramid):
            # color_profile is log_cut and data_range[1] is negative
            pass

        def test_sym_log(self, pyramid):
            # color_profile is sym_log and data_range[0] is positive or
            # data_range[1] is negative
            # color_profile is sym_log and data_range[0] is positive and
            # data_range[1] is negative
            pass