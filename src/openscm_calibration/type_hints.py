"""
Helpful type hints that can be used throughout

Implementation is an open question, see #23
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np
import numpy.typing as nptype

NPAnyFloat = np.floating[Any]
"""Numpy float of any type"""

NPAnyInt = np.integer[Any]
"""Numpy integer of any type"""

NPArrayFloatOrInt = nptype.NDArray[Union[NPAnyFloat, NPAnyInt]]
"""Numpy array of float or int type"""
