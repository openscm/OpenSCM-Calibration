from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as nptype

from .state import State

class EnsembleSampler:
    iteration: int
    ndim: int
    # Dimensionality of type hint would be helpful, anyway...
    def get_autocorr_time(
        self, discard: int = 0, thin: int = 1, **kwargs: Any
    ) -> nptype.NDArray[np.floating[Any]]: ...
    def get_chain(
        self, **kwargs: Any
    ) -> nptype.NDArray[np.floating[Any] | np.integer[Any]]: ...
    def get_last_sample(self, **kwargs: Any) -> nptype.NDArray[np.floating[Any]]: ...
    def get_log_prob(self, **kwargs: Any) -> nptype.NDArray[np.floating[Any]]: ...
    def sample(
        self,
        initial_state: nptype.NDArray[np.floating[Any]],
        iterations: int = 1,
        tune: bool = False,
        skip_initial_state_check: bool = False,
        thin_by: int = 1,
        store: bool = True,
        progress: bool | str = False,
        progress_kwargs: dict[str, Any] | None = None,
    ) -> Iterator[State]: ...
