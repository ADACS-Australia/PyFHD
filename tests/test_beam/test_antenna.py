import numpy as np
import pytest
from pyuvdata.datasets import fetch_data as uvdata_fetch

from pyfhd.beam_setup.antenna import jones_to_mueller


@pytest.mark.github_actions
def test_jones_to_mueller():

    # check to make sure we have the right version of pyuvdata installed
    from pyuvdata import UVBeam

    assert hasattr(UVBeam, "decompose_feed_aligned_terms")

    mwa_aee_jfile = uvdata_fetch("mwa_jmatrix")
    mwa_aee_zfile = uvdata_fetch("mwa_zmatrix")

    mwa_aee_beam = UVBeam.from_file(mwa_aee_jfile, mwa_zfile=mwa_aee_zfile)
    _, k_beam = mwa_aee_beam.decompose_feed_aligned_terms()
    k_arr = k_beam.data_array

    l_arr = jones_to_mueller(k_arr)

    j_inds = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    for sky_pi in range(4):
        sky_j = j_inds[sky_pi]
        for pol_i in range(4):
            inst_j = j_inds[pol_i]
            np.testing.assert_allclose(
                l_arr[sky_pi, pol_i],
                k_arr[sky_j[0], inst_j[0]] * np.conjugate(k_arr[sky_j[1], inst_j[1]]),
            )
