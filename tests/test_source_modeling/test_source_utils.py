import logging

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import Longitude, Latitude
from pyradiosky import SkyModel

from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.source_modeling.source_utils import (
    generate_source_cal_skymodel,
    stokes_cnv,
    source_dft,
    vis_delay_filter,
)
from pyfhd.source_modeling.vis_source_model import vis_source_model
from pyfhd.pyfhd_tools.test_utils import get_savs
from pyfhd.pyfhd_tools.unit_conv import altaz_to_radec, radec_to_pixel


@pytest.fixture(scope="session")
def gleam_cut_2013_sky(mwa_aee_beam_zenith_2013):

    _, psf, obs, _ = mwa_aee_beam_zenith_2013
    catalog_path = fetch_data("gleam_rlb2019_cut")

    sky = generate_source_cal_skymodel(
        obs=obs, psf=psf, logger=None, catalog_path=catalog_path
    )

    yield sky


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
@pytest.mark.parametrize("refraction", [None, "astropy", "idl"])
def test_generate_source_cal_skymodel(zenith_obs_2013, zenith_psf_2013_cut, refraction):
    psf = zenith_psf_2013_cut
    obs = zenith_obs_2013

    catalog_path = fetch_data("gleam_rlb2019_cut")

    fhd_cal_sources_file = fetch_data("2013_zenith_gleam_sources")

    expected_sky = SkyModel.from_file(
        fhd_cal_sources_file, extra_columns={"x": "image_x", "y": "image_y"}
    )

    # take this out once pyradiosky fix is in:
    if expected_sky.extended_model_group is not None:
        extended_comps = np.nonzero(expected_sky.extended_model_group != "")[0]
        if extended_comps.size == 0:
            expected_sky.extended_model_group = None

    expected_sky.at_frequencies(np.atleast_1d(expected_sky.reference_frequency[0]))

    # obs dimension should be an int
    obs["dimension"] = int(obs["dimension"])

    # set obs freq_array to match psf
    freqs = psf["freq"]
    n_freq = freqs.size
    obs["n_freq"] = n_freq
    obs["nf_vis"] = obs["nf_vis"][0:n_freq]
    obs["freq_center"] = np.mean(freqs)
    obs["baseline_info"]["freq"] = freqs
    obs["baseline_info"]["freq_use"] = np.ones((n_freq,), dtype=int)
    obs["baseline_info"]["fbin_i"] = np.arange(n_freq, dtype=int)

    sky = generate_source_cal_skymodel(
        obs=obs, psf=psf, catalog_path=catalog_path, logger=None, refraction=refraction
    )

    # sorting is slightly different because of different beam values
    # from round vs truncate pixel and refraction (if refraction != "idl")
    sky._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(sky.stokes[0, 0, :]))}
    )
    expected_sky._select_along_param_axis(
        {"Ncomponents": np.flip(np.argsort(expected_sky.stokes[0, 0, :]))}
    )

    if refraction is None:
        image_xy_atol = 0.09
    elif refraction == "astropy":
        image_xy_atol = 0.04
    else:
        image_xy_atol = 0.0011

    for col_name in ["image_x", "image_y"]:
        np.testing.assert_allclose(
            sky.extra_columns[col_name],
            expected_sky.extra_columns[col_name],
            rtol=0,
            atol=image_xy_atol,
        )

    # remove extra cols to test the rest of the objects
    sky.extra_columns = None
    expected_sky.extra_columns = None

    # FHD renumbers the components, which mucks up the name attribute
    # set them equal to enable comparison
    sky.name = expected_sky.name

    # make histories match for comparison purposes
    sky.history = expected_sky.history

    assert sky == expected_sky


@pytest.mark.github_actions
def test_stokes_cnv_zenith_src(mwa_aee_beam_zenith_2013):
    antenna, _, obs, _ = mwa_aee_beam_zenith_2013

    n_comp = 3
    zenith_sky = SkyModel(
        name=[f"zen_src_{ind}" for ind in range(n_comp)],
        ra=Longitude([obs["zenra"] * units.deg] * n_comp),
        dec=Latitude([obs["zendec"] * units.deg] * n_comp),
        spectral_type="spectral_index",
        reference_frequency=np.array([obs["freq_center"]] * n_comp) * units.Hz,
        spectral_index=[-0.8] * n_comp,
        stokes=np.repeat(np.reshape([1, 0, 0, 0], (4, 1, 1)), n_comp, axis=2)
        * units.Jy,
        frame="fk5",
        extra_column_dict={
            "image_x": np.array([obs["zenx"]] * n_comp),
            "image_y": np.array([obs["zeny"]] * n_comp),
        },
    )
    new_sky = stokes_cnv(sky=zenith_sky, antenna=antenna, obs=obs, inverse=True)

    # I expected a slightly better tolerance than this...
    for pol_i in range(2):
        np.testing.assert_allclose(
            new_sky.extra_columns[f"flux_pol_{pol_i}"],
            np.full((n_comp,), 0.5, dtype=np.complex128),
            atol=2e-5,
            rtol=0,
        )


@pytest.mark.github_actions
def test_stokes_cnv_off_zenith(mwa_aee_beam_zenith_2013):
    antenna, _, obs, _ = mwa_aee_beam_zenith_2013

    za_grid = np.linspace(5, 15, 5)
    az_grid = np.linspace(0, 360, 8, endpoint=False)

    za_array, az_array = np.meshgrid(za_grid, az_grid)
    za_array = za_array.flatten()
    az_array = az_array.flatten()

    ra_array, dec_array = altaz_to_radec(
        alt=90 - za_array,
        az=az_array,
        lat=obs["lat"],
        lon=obs["lon"],
        height=obs["alt"],
        time=obs["jd0"],
    )

    image_x, image_y = radec_to_pixel(ra_array, dec_array, obs["astr"])

    n_comp = za_array.size
    sky = SkyModel(
        name=[f"grid_src_{ind}" for ind in range(n_comp)],
        ra=Longitude(ra_array * units.deg),
        dec=Latitude(dec_array * units.deg),
        spectral_type="spectral_index",
        reference_frequency=np.array([obs["freq_center"]] * n_comp) * units.Hz,
        spectral_index=[-0.8] * n_comp,
        stokes=np.repeat(np.reshape([1, 0, 0, 0], (4, 1, 1)), n_comp, axis=2)
        * units.Jy,
        frame="fk5",
        extra_column_dict={"image_x": image_x, "image_y": image_y},
    )
    new_sky = stokes_cnv(sky=sky, antenna=antenna, obs=obs, inverse=True)

    # I expected a slightly better tolerance than this...
    for pol_i in range(2):
        np.testing.assert_allclose(
            new_sky.extra_columns[f"flux_pol_{pol_i}"],
            np.full((n_comp,), 0.5, dtype=np.complex128),
            atol=0.02,
            rtol=0,
        )


@pytest.mark.github_actions
def test_source_dft_center():
    # check that a centered source returns all ones in uv plane

    model_uv = source_dft(
        x_loc=np.atleast_1d([2048.0 / 2]),
        y_loc=np.atleast_1d([2048.0 / 2]),
        dimension=2048,
        elements=2048,
        flux=np.reshape(np.array([0.5, 0.5, 0, 0]), (4, 1)),
        logger=logging.getLogger(),
    )

    np.testing.assert_allclose(model_uv[0:2], 0.5, atol=1e-15, rtol=0)
    np.testing.assert_allclose(model_uv[2:], 0, atol=1e-15, rtol=0)


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_source_dft_model(model_uv_zenith_2013):
    model_uv_full = model_uv_zenith_2013

    fhd_model_uv_file = fetch_data("2013_zenith_gleam_model_uv_small")
    fhd_model_uv_sav_dict = get_savs(fhd_model_uv_file, "")
    fhd_model_uv_sav_dict = recarray_to_dict(fhd_model_uv_sav_dict)
    fhd_model_uv_cut = fhd_model_uv_sav_dict["model_uv_cut"].T
    xrange = fhd_model_uv_sav_dict["xrange"]
    yrange = fhd_model_uv_sav_dict["yrange"]

    np.testing.assert_allclose(
        model_uv_full[xrange[0] : xrange[1] + 1, yrange[0] : yrange[1] + 1],
        fhd_model_uv_cut,
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.github_actions
def test_vis_delay_filter(zenith_obs_2013, zenith_params_2013):
    obs = zenith_obs_2013
    params = zenith_params_2013

    cut_vis_model_file = fetch_data("2013_zenith_gleam_model_vis_75bl")
    cut_vis_model_dict = get_savs(cut_vis_model_file, "")
    cut_vis_model_dict = recarray_to_dict(cut_vis_model_dict)
    cut_vis_model = cut_vis_model_dict["vis_model_ptr"].transpose(0, 2, 1)

    nblt = cut_vis_model.shape[2]
    # make params lengths match cut down vis length
    for key in params:
        params[key] = params[key][:nblt]

    filtered_vis_arr = vis_delay_filter(cut_vis_model, params=params, obs=obs)

    fhd_filtered_vis_model_file = fetch_data(
        "2013_zenith_gleam_model_vis_75bl_filtered"
    )
    fhd_filtered_vis_model_dict = get_savs(fhd_filtered_vis_model_file, "")
    fhd_filtered_vis_model_dict = recarray_to_dict(fhd_filtered_vis_model_dict)
    fhd_filtered_vis_model = fhd_filtered_vis_model_dict["vis_model_ptr"].transpose(
        0, 2, 1
    )

    np.testing.assert_allclose(
        filtered_vis_arr, fhd_filtered_vis_model, atol=1e-10, rtol=0
    )


@pytest.mark.github_actions
@pytest.mark.filterwarnings("ignore:Some Stokes values are NaNs.")
@pytest.mark.filterwarnings("ignore:Some Stokes I values are negative.")
def test_vis_source_model_smoke(
    mwa_aee_beam_zenith_2013, zenith_params_2013, gleam_cut_2013_sky
):
    # This just checks that calling vis_source_model doesn't error
    antenna, psf, obs, pyfhd_config = mwa_aee_beam_zenith_2013
    sky = gleam_cut_2013_sky
    params = zenith_params_2013

    vis_source_model(
        pyfhd_config=pyfhd_config,
        obs=obs,
        psf=psf,
        params=params,
        antenna=antenna,
        skymodel=sky,
        logger=logging.getLogger(),
        vis_weights=None,
        fill_model_visibilities=True,
    )
