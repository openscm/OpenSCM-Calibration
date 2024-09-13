"""
Minimisation helpers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from openscm_calibration.store import OptResStore
from openscm_calibration.typing import (
    DataContainer,
    DataContainer_co,
    DataContainer_contra,
)

if TYPE_CHECKING:
    from typing import Any


class SupportsCostCalculation(Protocol[DataContainer_contra]):
    """
    Class that supports cost calculations
    """

    def calculate_cost(self, model_results: DataContainer_contra) -> float:
        """
        Calculate cost function

        Parameters
        ----------
        model_results
            Model results for which to calculate the cost

        Returns
        -------
        :
            Cost function value
        """


class SupportsModelRun(Protocol[DataContainer_co]):
    """
    Class that supports model runs
    """

    def run_model(
        self,
        x: np.typing.NDArray[np.number[Any]],
    ) -> DataContainer_co:
        """
        Calculate cost function

        Parameters
        ----------
        x
            Parameter values

        Returns
        -------
        :
            Results of model run
        """


def to_minimize_full(
    x: np.typing.NDArray[np.number[Any]],
    cost_calculator: SupportsCostCalculation[DataContainer],
    model_runner: SupportsModelRun[DataContainer],
    store: OptResStore[DataContainer] | None = None,
    known_error: type[ValueError] | None = None,
) -> float:
    """
    Calculate cost for given set of parameters

    This is a function that can be minimised with scipy

    Before passing to scipy, all the arguments except ``x`` should be filled
    using e.g. :func:`functools.partial`

    Parameters
    ----------
    x
        Parameter array

    cost_calculator
        Calculator of the cost function for each set of model results

    model_runner
        Model runner

    store
        Store of results at each step in the optimisation (useful for plotting)

    known_error
        Known error that can occur when solving. If any other error is
        encountered, it is raised rather than being allowed to pass.

    Returns
    -------
        Cost function for array ``x``
    """
    if known_error:
        try:
            model_results = model_runner.run_model(x)
        except known_error:
            cost = np.inf
            if store:
                store.note_failed_run(cost, x)
            return cost
    else:
        model_results = model_runner.run_model(x)

    cost = cost_calculator.calculate_cost(model_results)
    if store:
        store.append_result_cost_x(model_results, cost, x)

    return cost
