"""
Types used throughout
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as nptype

DataContainer = TypeVar("DataContainer")
DataContainer_co = TypeVar("DataContainer_co", covariant=True)
DataContainer_contra = TypeVar("DataContainer_contra", contravariant=True)


class SupportsModelRun(Protocol[DataContainer_co]):
    """
    Class that supports model runs
    """

    def run_model(
        self,
        x: nptype.NDArray[np.number[Any]],
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


class SupportsNegativeLogLikelihoodCalculation(Protocol[DataContainer_contra]):
    """
    Class that supports negative log likelihood calculations
    """

    def calculate_negative_log_likelihood(
        self, model_results: DataContainer_contra
    ) -> float:
        """
        Calculate negative log likelihood

        Parameters
        ----------
        model_results
            Model results for which to calculate the cost

        Returns
        -------
        :
            Negative log likelihood
        """
