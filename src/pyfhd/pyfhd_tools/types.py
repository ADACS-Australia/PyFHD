"""Define some commonly used numpy array type hint aliases."""

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]
StrArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]
