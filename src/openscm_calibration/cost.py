"""
Calculate cost of model results
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define, field

from openscm_calibration.exceptions import AlignmentError, MissingValueError

if TYPE_CHECKING:
    import attr
    import pandas as pd
    import scmdata.run


def _works_with_self_target(
    instance: OptCostCalculatorSSE,
    attribute: attr.Attribute[scmdata.run.BaseScmRun],
    value: scmdata.run.BaseScmRun,
) -> None:
    try:
        instance.calculate_cost(instance.target)
    except KeyError as exc:
        value_aligned, target_ts_aligned = value.timeseries().align(
            instance.target.timeseries(), axis="rows"
        )
        raise AlignmentError(
            name_left="self.target",
            val_left=target_ts_aligned,
            name_right=attribute.name,
            val_right=value_aligned,
            extra_context=(
                "Note we have aligned the timeseries "
                "so that nan values appear where there are alignment issues"
            ),
        ) from exc


def _is_meta_in_target(
    instance: OptCostCalculatorSSE,
    attribute: attr.Attribute[str],
    value: str,
) -> None:
    available_metadata = instance.target.meta_attributes
    if value not in available_metadata:
        raise MissingValueError(
            "instance.target.meta_attributes",
            vals=available_metadata,
            missing_vals=value,
        )


@define
class OptCostCalculatorSSE:
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
        available_cols = set(normalisation_series.index.names)
        missing_cols = required_columns - available_cols
        if missing_cols:
            raise MissingValueError(
                "normalisation_series.index.names",
                vals=sorted(available_cols),
                missing_vals=sorted(missing_cols),
            )

        target_ts_no_unit = target.timeseries().reset_index("unit", drop=True)

        # This is basically what pandas does internally when doing ops:
        # align and then broadcast
        norm_series_aligned, _ = normalisation_series.align(target_ts_no_unit)

        if norm_series_aligned.isna().any().any():
            raise AlignmentError(
                name_left="target_ts_no_unit",
                val_left=target_ts_no_unit,
                name_right="norm_series_aligned",
                val_right=norm_series_aligned,
                extra_context="Even after aligning, there are still nan values",
            )

        if norm_series_aligned.shape[0] != target_ts_no_unit.shape[0]:
            raise AlignmentError(
                name_left="target_ts_no_unit",
                val_left=target_ts_no_unit,
                name_right="norm_series_aligned",
                val_right=norm_series_aligned,
                extra_context=(
                    "After aligning, there are more rows in the normalisation "
                    "than in the target"
                ),
            )

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

        cost = float((diff.convert_unit("1") ** 2).values.sum().sum())  # noqa: PD011

        return cost
