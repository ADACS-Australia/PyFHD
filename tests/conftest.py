import logging

import numpy as np
import pytest
from pyuvdata.datasets import fetch_data as uvdata_fetch

from pyfhd.beam_setup.beam import create_psf
from pyfhd.data.datasets import fetch_data
from pyfhd.io.pyfhd_io import recarray_to_dict
from pyfhd.pyfhd_tools.test_utils import get_savs, sav_file_rearrange_psf


@pytest.fixture(scope="session")
def zenith_obs_2013():

    # obs_file = fetch_data("2013_zenith_obs")
    obs_file = "/Users/bryna/Projects/Physics/pyfhd-datasets/fhd_extracts/MWA/std_1061316296/cut_down_obs.sav"

    obs_sav_dict = get_savs(obs_file, "")
    obs_sav_dict = recarray_to_dict(obs_sav_dict)
    obs = obs_sav_dict["obs"]
    obs["n_baselines"] = obs["nbaselines"]
    obs["delays"] = obs["delays"].astype("int").repeat(2).reshape((2, 16))
    obs["dimension"] = int(obs["dimension"])
    obs["elements"] = int(obs["elements"])

    yield obs


@pytest.fixture(scope="session")
def zenith_psf_2013_cut():
    psf_cut_file = fetch_data("2013_zenith_psf_small")

    psf_sav_dict = get_savs(psf_cut_file, "")
    # fix the psf to be properly arranged
    psf = sav_file_rearrange_psf(psf_sav_dict["psf"])

    yield psf


@pytest.fixture(scope="session")
def mwa_aee_beam_zenith_2013(zenith_obs_2013, tmp_path_factory):
    obs = zenith_obs_2013

    # check to make sure we have the right version of pyuvdata installed
    from pyuvdata import UVBeam

    assert hasattr(UVBeam, "decompose_feed_aligned_terms")

    mwa_aee_jfile = uvdata_fetch("mwa_jmatrix")
    mwa_aee_zfile = uvdata_fetch("mwa_zmatrix")

    # set obs freq_array to match uvbeam freq_array
    freqs = np.array([1.6512e08, 1.8048e08])
    n_freq = freqs.size
    obs["n_freq"] = n_freq
    obs["nf_vis"] = obs["nf_vis"][0:n_freq, :]
    obs["freq_center"] = np.mean(freqs)
    obs["baseline_info"]["freq"] = freqs
    obs["baseline_info"]["freq_use"] = np.ones((n_freq,), dtype=int)
    obs["baseline_info"]["fbin_i"] = np.arange(n_freq, dtype=int)

    pyfhd_config = {
        "instrument": "mwa",
        "psf_dim": 14,
        "psf_resolution": 10,
        "beam_mask_threshold": 1e2,
        "uvbeam_file_path": mwa_aee_jfile,
        "uvbeam_zfile_path": mwa_aee_zfile,
        "interpolate_kernel": False,
        "beam_offset_time": None,
        "analytic_beam_yaml": None,
        "uvbeam_freq_buffer": 2e6,
        "beam_clip_floor": False,
        "uvbeam_mwa_include_cross_feed_coupling": False,
        "output_dir": tmp_path_factory.mktemp("zenith_2013"),
        "obs_id": 1061316296,
    }

    psf, antenna = create_psf(obs, pyfhd_config, logger=logging.getLogger())

    yield antenna, psf, obs, pyfhd_config
