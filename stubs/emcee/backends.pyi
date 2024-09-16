from typing import Any

import numpy as np
import numpy.typing as nptype

class Backend:
    iteration: int
    def get_autocorr_time(
        self, discard: int = 0, thin: int = 1, **kwargs: Any
    ) -> nptype.NDArray[np.floating[Any]]: ...
    def get_chain(
        self, **kwargs: Any
    ) -> nptype.NDArray[np.floating[Any] | np.integer[Any]]: ...
    def get_last_sample(self, **kwargs: Any) -> nptype.NDArray[np.floating[Any]]: ...
    def get_log_prob(self, **kwargs: Any) -> nptype.NDArray[np.floating[Any]]: ...
