import numpy as np
from typing import Tuple
import logging

def beam_setup(pyfhd_config : dict, obs : dict) -> Tuple[dict, dict]:

    psf = {}
    antenna = {}
    return psf, antenna