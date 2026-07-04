import numpy as np
import pytest

from pyfhd.beam_setup.beam_utils import beam_power
from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.test_utils import get_savs


@pytest.mark.github_actions
def test_beam_power(mwa_aee_beam_zenith_2013):
    antenna, psf, obs, pyfhd_config = mwa_aee_beam_zenith_2013

    psf_superres_fhd_file = fetch_data("2013_zenith_psf_superres")

    psf_superres_fhd_dict = get_savs(psf_superres_fhd_file, "")
    psf_superres_fhd_dict = recarray_to_dict(psf_superres_fhd_dict)
    psf_superres_fhd = psf_superres_fhd_dict["psf_base_superres"].T

    zen_int_x = (obs["zenx"] - obs["obsx"]) / psf["scale"] + psf["image_dim"] / 2
    zen_int_y = (obs["zeny"] - obs["obsy"]) / psf["scale"] + psf["image_dim"] / 2

    res_super = 1 / (psf["resolution"] / psf["intermediate_res"])

    xvals_uv_superres, yvals_uv_superres = np.meshgrid(
        np.arange(psf["superres_dim"]), np.arange(psf["superres_dim"])
    )
    xvals_uv_superres = (
        xvals_uv_superres * res_super
        - np.floor(psf["dim"] / 2) * psf["intermediate_res"]
        + np.floor(psf["image_dim"] / 2)
    )
    yvals_uv_superres = (
        yvals_uv_superres * res_super
        - np.floor(psf["dim"] / 2) * psf["intermediate_res"]
        + np.floor(psf["image_dim"] / 2)
    )

    for ant_pol1 in np.arange(obs["n_pol"]):
        for ant_pol2 in np.arange(obs["n_pol"]):
            psf_base_superres = beam_power(
                antenna=antenna,
                ant_pol_1=ant_pol1,
                ant_pol_2=ant_pol2,
                freq_i=1,
                psf=psf,
                zen_int_x=zen_int_x,
                zen_int_y=zen_int_y,
                xvals_uv_superres=xvals_uv_superres,
                yvals_uv_superres=yvals_uv_superres,
                pyfhd_config=pyfhd_config,
            )

            # tolerance is not super tight because the codes used to produce
            # these are different. Among other things, the pyfhd code implements
            # the updated decomposition while the FHD code still uses the old one.
            # The frequencies are also slightly different.
            np.testing.assert_allclose(
                psf_base_superres,
                psf_superres_fhd[ant_pol1, ant_pol2],
                rtol=0,
                atol=6e-4,
            )
