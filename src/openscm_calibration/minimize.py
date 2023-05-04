"""
Minimisation helpers
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import scmdata.run

from openscm_calibration.store import OptResStore

if TYPE_CHECKING:
    from openscm_calibration.type_hints import NPArrayFloatOrInt


class SupportsCostCalculation(Protocol):
    """
    Class that supports cost calculations
    """

    def calculate_cost(self, model_results: scmdata.run.BaseScmRun) -> float:
        """
        Calculate cost function

        Parameters
        ----------
        model_results
            Model results for which to calculate the cost

        Returns
        -------
            Cost function value
        """


class SupportsModelRun(Protocol):
    """
    Class that supports model runs
    """

    def run_model(
        self,
        x: NPArrayFloatOrInt,
    ) -> scmdata.run.BaseScmRun:
        """
        Calculate cost function

        Parameters
        ----------
        x
            Parameter values

        Returns
        -------
            results of model run
        """


def to_minimize_full(
    x: NPArrayFloatOrInt,
    cost_calculator: SupportsCostCalculation,
    model_runner: SupportsModelRun,
    store: OptResStore | None = None,
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
