"""
Classes and functions used in our calibration demo notebook

See `docs/how-to-guides/how-to-run-a-calibration.py`.

These are in their own module to help the parallelisation
(local functions and classes don't pickle well),
but also just for clarity in the documentation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable

import numpy as np
import pint
from attrs import define

from openscm_calibration.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    import matplotlib.axes
    import pandas as pd


@define
class CostCalculator:
    """Calculate the cost function for a given set of model results"""

    target: ExperimentResultCollection
    """The target to which we are calibrating"""

    normalisation: pint.registry.UnitRegistry.Quantity
    """
    The normalisation to apply to the difference between model results and the target
    """

    def calculate_cost(
        self,
        model_results: ExperimentResultCollection,
    ) -> float:
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
        model_result_dict = model_results.to_dict()
        target_dict = self.target.to_dict()

        sses = 0.0
        for experiment_id in model_result_dict:
            model_result_exp = model_result_dict[experiment_id]
            target_exp = target_dict[experiment_id]

            if not np.isclose(
                model_result_exp.time,
                target_exp.time,
            ).all():
                msg = "Time axes differ"
                raise NotImplementedError(msg)

            diff_normalised = (
                model_result_exp.values - target_exp.values
            ) / self.normalisation
            diff_squared = diff_normalised**2

            sses += diff_squared.sum()

        return sses

    def calculate_negative_log_likelihood(
        self,
        model_results: ExperimentResultCollection,
    ) -> float:
        """
        Calculate the negative log likelihood of a given set of results

        Parameters
        ----------
        model_results
            Model results for which to calculate the negative log likelihood

        Returns
        -------
        :
            Negative log likelihood (up to an additive constant)
        """
        sses = self.calculate_cost(model_results)
        # TODO: find the proof of this
        negative_log_likelihood = -sses / 2

        return negative_log_likelihood


@define
class Timeseries:
    """Timeseries container"""

    time: pint.registry.UnitRegistry.Quantity
    """The time axis"""

    values: pint.registry.UnitRegistry.Quantity
    """The timeseries value at each point in time"""


@define
class ExperimentResult:
    """Results of an experiment"""

    experiment_id: str
    """ID of the experiment"""

    result: Timeseries
    """The position of the mass over time in the experiment"""


@define
class ExperimentResultCollection:
    """Collection of results from one or more experiments"""

    results: tuple[ExperimentResult, ...]
    """Results of the experiments"""

    iteration: int | None = None
    """Optimisation iteration in which these results were generated"""

    def to_dict(self) -> dict[str, Timeseries]:
        """Convert to a dictionary"""
        return {r.experiment_id: r.result for r in self.results}

    def to_timeseries(
        self,
        value_unit: str | None = None,
        time_unit: str = "yr",
    ) -> pd.DataFrame:
        """
        Convert self to timeseries

        In OpenSCM Calibration, timeseries means that the time axis is the index

        Parameters
        ----------
        value_unit
            Unit to use for the values.

            If not supplied, no conversion is performed.

        time_unit
            Unit to use for the time axis

        Returns
        -------
        :
            'Timeseries' view of the data in `self`
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "to_timeseries", requirement="pandas"
            ) from exc

        ts_l = []
        for result in self.results:
            values = result.result.values
            if value_unit is not None:
                # pint weirdness causing type hinting issues
                values = values.to(value_unit)  # type: ignore

            df_r = pd.DataFrame(
                values.m,
                columns=pd.MultiIndex.from_tuples(
                    tuples=[(result.experiment_id, values.u)],
                    names=["experiment_id", "unit"],
                ),
                index=result.result.time.to(time_unit).m,
            )
            ts_l.append(df_r)

        ts = pd.concat(ts_l, axis="columns")

        return ts

    def lineplot(
        self,
        ax: matplotlib.axes.Axes,
        x_units: str = "yr",
        y_units: str = "m",
    ) -> matplotlib.axes.Axes:
        """
        Make a line plot

        Parameters
        ----------
        ax
            Axes on which to plot

        x_units
            Units to use for the x-axis

        y_units
            Units to use for the y-axis

        Returns
        -------
        :
            Axes on which the plot was made
        """
        for result in self.results:
            ax.plot(
                result.result.time.to(x_units).m,
                result.result.values.to(y_units).m,
                label=result.experiment_id,
            )

        ax.set_ylabel(f"[{y_units}]")
        ax.set_xlabel(f"[{x_units}]")

        return ax


def add_iteration_info(
    res: ExperimentResultCollection, iteration: int
) -> ExperimentResultCollection:
    """Add iteration info to a result"""
    res.iteration = iteration

    return res


def convert_results_to_plot_dict(
    res: ExperimentResultCollection,
) -> dict[str, ExperimentResultCollection]:
    """Convert results into a dictionary, grouped for plotting"""
    return {
        k: ExperimentResultCollection(
            results=(
                ExperimentResult(
                    experiment_id=k,
                    result=v,
                ),
            ),
        )
        for k, v in res.to_dict().items()
    }


def get_timeseries(
    res: ExperimentResultCollection,
    value_unit: str | None = None,
) -> pd.DataFrame:
    """
    Get timeseries from results
    """
    return res.to_timeseries()


def plot_timeseries(  # noqa: PLR0913
    best_run: ExperimentResultCollection,
    others_to_plot: tuple[ExperimentResultCollection, ...],
    target: ExperimentResultCollection,
    convert_results_to_plot_dict: Callable[
        [ExperimentResultCollection], dict[str, ExperimentResultCollection]
    ],
    timeseries_keys: Iterable[str],
    axes: dict[str, matplotlib.axes.Axes],
    get_timeseries: Callable[[ExperimentResultCollection], pd.DataFrame],
) -> None:
    """
    Plot timeseries

    Parameters
    ----------
    best_run
        Best run from iterations

    others_to_plot
        Other results to plot from iterations

    target
        Target to which we are calibrating

    convert_results_to_plot_dict
        Callable which converts the data into a dictionary
        in which the keys are a subset of the values in `axes`

    timeseries_keys
        Keys of the timeseries to plot

    axes
        Axes on which to plot

    get_timeseries
        Function which converts the data into a
        [`pandas.DataFrame`][] for plotting
    """
    others_to_plot_dict = defaultdict(list)
    for res in others_to_plot:
        res_dict = convert_results_to_plot_dict(res)
        for k, v in res_dict.items():
            others_to_plot_dict[k].append(v)

    target_dict = convert_results_to_plot_dict(target)
    best_dict = convert_results_to_plot_dict(best_run)

    for k in timeseries_keys:
        ax = axes[k]

        others_to_plot_k = others_to_plot_dict[k]
        for other_to_plot in others_to_plot_k:
            get_timeseries(other_to_plot).plot.line(
                ax=ax,
                legend=False,
                linewidth=0.5,
                alpha=0.3,
                color="tab:gray",
                zorder=1.5,
            )

        target_k = target_dict[k]
        get_timeseries(target_k).plot.line(
            ax=ax,
            legend=False,
            linewidth=2,
            alpha=0.8,
            color="tab:blue",
            zorder=2.0,
        )

        best_k = best_dict[k]
        get_timeseries(best_k).plot.line(
            ax=ax,
            legend=False,
            linewidth=2,
            alpha=0.8,
            color="tab:orange",
            zorder=2.5,
        )

        ax.set_ylabel(k)
