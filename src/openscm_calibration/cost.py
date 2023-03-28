"""
Calculate cost of model results
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    import attr
    import pandas as pd
    import scmdata.run


def _works_with_self_target(
    instance: OptCostCalculatorSSE,
    attribute: attr.Attribute[scmdata.run.BaseScmRun],
    value: scmdata.run.BaseScmRun,
) -> None:
    def _get_msg() -> str:
        target_ts = instance.target.timeseries()
        value_ts = value.timeseries()

        msg = (
            "target and normalisation are somehow misaligned "
            "(passing self.target to self.calculate_cost results "
            "in nan), please check.\n"
            f"target timeseries:\n{target_ts}\n"
            f"{attribute.name} timeseries:\n{value_ts}"
        )

        return msg

    try:
        instance.calculate_cost(instance.target)
    except KeyError as exc:
        raise ValueError(_get_msg()) from exc


def _is_meta_in_target(
    instance: OptCostCalculatorSSE,
    attribute: attr.Attribute[str],
    value: str,
) -> None:
    available_metadata = instance.target.meta_attributes
    if value not in available_metadata:
        msg = (
            f"value of ``{attribute.name}``, '{value}' is not in the metadata "
            f"of target. Available metadata: {available_metadata}"
        )

        raise KeyError(msg)


@define
class OptCostCalculatorSSE:  # pylint: disable=too-few-public-methods
    """
    Cost calculator based on sum of squared errors

    This is a convenience class. We may want to refactor it in future to
    provide greater flexibility for other cost calculations.
    """

    target: scmdata.run.BaseScmRun
    """Target timeseries"""

    model_col: str = field(validator=[_is_meta_in_target])
    """
    Column which contains the name of the model.

    This is used when subtracting the model results from the target
    """

    normalisation: scmdata.run.BaseScmRun = field(validator=[_works_with_self_target])
    """
    Normalisation values

    Should have same timeseries as target. See the class methods for helpers.
    """

    @classmethod
    def from_unit_normalisation(
        cls, target: scmdata.run.BaseScmRun, model_col: str
    ) -> OptCostCalculatorSSE:
        """
        Initialise assuming unit normalisation for each timeseries.

        This is a convenience method, but is not recommended for any serious
        work as unit normalisation is unlikely to be a good choice for most
        problems.

        Parameters
        ----------
        target
            Target timeseries

        model_col
            Column which contains of the model in ``target``

        Returns
        -------
            :obj:`OptCostCalculatorSSE` such that the normalisation is 1 for
            all timepoints (with the units defined by whatever the units of
            each timeseries are in ``target``)
        """
        norm = target.timeseries()
        norm.loc[:, :] = 1
        norm = type(target)(norm)

        return cls(target=target, normalisation=norm, model_col=model_col)

    @classmethod
    def from_series_normalisation(
        cls,
        target: scmdata.run.BaseScmRun,
        model_col: str,
        normalisation_series: pd.Series,
    ) -> OptCostCalculatorSSE:
        """
        Initialise starting from a series that defines normalisation for each timeseries.

        The series is broadcast to match the timeseries in target, using the
        same value for all timepoints in each timeseries.

        Parameters
        ----------
        target
            Target timeseries

        model_col
            Column which contains of the model in ``target``

        normalisation_series
            Series to broadcast to create the desired normalisation

        Returns
        -------
            Initialised :obj:`OptCostCalculatorSSE`
        """
        required_columns = {"variable", "unit"}
        missing_cols = required_columns - set(normalisation_series.index.names)
        if missing_cols:
            msg = (
                "normalisation is missing required column(s): "
                f"``{sorted(missing_cols)}``"
            )
            raise KeyError(msg)

        target_ts_no_unit = target.timeseries().reset_index("unit", drop=True)

        # This is basically what pandas does internally when doing ops:
        # align and then broadcast
        norm_series_aligned, _ = normalisation_series.align(target_ts_no_unit)

        if norm_series_aligned.isnull().any().any():
            msg = (
                "Even after aligning, there are still nan values.\n"
                f"norm_series_aligned:\n{norm_series_aligned}\n"
                f"target_ts_no_unit:\n{target_ts_no_unit}"
            )
            raise ValueError(msg)

        if norm_series_aligned.shape[0] != target_ts_no_unit.shape[0]:
            msg = (
                "After aligning, there are more rows in the normalisation "
                "than in the target.\n"
                f"norm_series_aligned:\n{norm_series_aligned}\n"
                f"target_ts_no_unit:\n{target_ts_no_unit}"
            )
            raise ValueError(msg)

        norm_series_aligned = type(target_ts_no_unit)(
            np.broadcast_to(norm_series_aligned.values, target_ts_no_unit.T.shape).T,
            index=norm_series_aligned.index,
            columns=target_ts_no_unit.columns,
        )

        normalisation = type(target)(norm_series_aligned)

        return cls(target=target, normalisation=normalisation, model_col=model_col)

    def calculate_cost(self, model_results: scmdata.run.BaseScmRun) -> float:
        """
        Calculate cost function based on model results

        Parameters
        ----------
        model_results
            Model results of which to calculate the cost

        Returns
        -------
            Cost
        """
        diff = model_results.subtract(
            self.target, op_cols={self.model_col: "res - target"}
        ).divide(
            self.normalisation,
            op_cols={self.model_col: "(res - target) / normalisation"},
        )

        cost = float((diff.convert_unit("1") ** 2).values.sum().sum())

        return cost
