from typing import Any

import numpy as np
import numpy.typing as nptype

class Backend:
    def get_chain(self, **kwargs: Any) -> nptype.NDArray[np.floating[Any]]: ...
