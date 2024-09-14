"""
Support for plotting during scipy optimisation with [`scmdata`][]
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

from openscm_calibration.exceptions import (
    MissingOptionalDependencyError,
)
from openscm_calibration.scipy_plotting.base import ResultsToDictConverter

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.axes
    import pandas as pd
    import scmdata.run

DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS: dict[str, Any] = {
    "legend": False,
    "linewidth": 0.5,
    "alpha": 0.3,
    "color": "tab:gray",
    "zorder": 1.5,
}
"""
Default value for `background_ts_kwargs` used by `plot_timeseries_scmrun`

Provided to give the user an easy to modify these defaults if they wish
or a starting point.
"""


DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS: dict[str, Any] = {
    "legend": False,
    "linewidth": 2,
    "alpha": 0.8,
    "color": "tab:blue",
    "zorder": 2,
}
"""
Default value for `target_ts_kwargs` used by `plot_timeseries_scmrun`

Provided to give the user an easy to modify these defaults if they wish
or a starting point.
"""


DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS: dict[str, Any] = {
    "legend": False,
    "linewidth": 2,
    "alpha": 0.8,
    "color": "tab:orange",
    "zorder": 2.5,
}
"""
Default value for `best_ts_kwargs` used by `plot_timeseries_scmrun`

Provided to give the user an easy to modify these defaults if they wish
or a starting point.
"""


def get_timeseries_scmrun(
    inp: scmdata.run.BaseScmRun,
    time_axis: str = "year-month",
) -> pd.DataFrame:
    """
    Get timeseries for plotting from an [`scmdata.run.BaseScmRun`][]

    Parameters
    ----------
    inp
        Object to convert

    time_axis
        Passed to [`inp.timeseries`] when doing the conversion

    Returns
    -------
    :
        Data with the time axis as rows,
        ready for simplified plotting using panda's plotting methods.
    """
    return inp.timeseries(time_axis=time_axis).T


def convert_target_to_model_output_units_scmrun(
    *,
    target: scmdata.run.BaseScmRun,
    model_output: scmdata.run.BaseScmRun,
    convert_results_to_plot_dict: ResultsToDictConverter[scmdata.run.BaseScmRun],
) -> scmdata.run.BaseScmRun:
    """
    Convert the target data to the model output's units

    This is a helper function
    that allows the data to be lined up before setting up plotting etc.

    Parameters
    ----------
    target
        Target data to convert

    model_output
        A sample of the model's output

    convert_results_to_plot_dict
        The function that will be used to convert
        [`scmdata.run.BaseScmRun`][] to a dictionary when doing the plotting.

    Returns
    -------
    :
        Target data with units that match the model output
    """
    try:
        import scmdata
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "convert_target_to_model_output_units_scmrun", requirement="scmdata"
        ) from exc

    target_d = convert_results_to_plot_dict(target)
    model_output_d = convert_results_to_plot_dict(model_output)

    tmp = []
    for group, run in target_d.items():
        model_unit = str(model_output_d[group].get_unique_meta("unit", True))
        run_converted = run.convert_unit(model_unit)

        tmp.append(run_converted)

    out = scmdata.run_append(tmp)

    return out


def plot_timeseries_scmrun(  # noqa: PLR0913,too-many-locals
    best_run: scmdata.run.BaseScmRun,
    others_to_plot: tuple[scmdata.run.BaseScmRun, ...],
    target: scmdata.run.BaseScmRun,
    convert_results_to_plot_dict: ResultsToDictConverter[scmdata.run.BaseScmRun],
    timeseries_keys: Iterable[str],
    axes: dict[str, matplotlib.axes.Axes],
    get_timeseries: Callable[[scmdata.run.BaseScmRun], pd.DataFrame],
    background_ts_kwargs: dict[str, Any] | None = None,
    target_ts_kwargs: dict[str, Any] | None = None,
    best_ts_kwargs: dict[str, Any] | None = None,
    ylabel_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Plot timeseries

    This plots the target, the best result so far and other results

    Parameters
    ----------
    best_run
        Best run from iterations

    others_to_plot
        Other results to plot from iterations

    target
        Target timeseries

    convert_results_to_plot_dict
        Callable which converts [`scmdata.run.BaseScmRun`][]
        into a dictionary in which the keys are a subset of the values in `axes`.

    timeseries_keys
        Keys of the timeseries to plot

    axes
        Axes on which to plot

    get_timeseries
        Function which converts [`scmdata.run.BaseScmRun`][] into a
        [`pd.DataFrame`][pandas.DataFrame] for plotting

    background_ts_kwargs
        Passed to [`pd.DataFrame.plot.line`][pandas.DataFrame.plot.line]
        when plotting the background timeseries.
        If not supplied, we use
        [`DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS`][openscm_calibration.scipy_plotting.scmdata.DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS].

    target_ts_kwargs
        Passed to [`pd.DataFrame.plot.line`][pandas.DataFrame.plot.line]
        when plotting the target timeseries.
        If not supplied, we use
        [`DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS`][openscm_calibration.scipy_plotting.scmdata.DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS].

    best_ts_kwargs
        Passed to [`pd.DataFrame.plot.line`][pandas.DataFrame.plot.line]
        when plotting the best timeseries.
        If not supplied, we use
        [`DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS`][openscm_calibration.scipy_plotting.scmdata.DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS].

    ylabel_kwargs
        Passed to [`ax.set_ylabel`][matplotlib.axes.Axes.set_ylabel]
        when setting the y-labels of each panel.
    """
    try:
        import scmdata
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plot_timeseries_scmrun", requirement="scmdata"
        ) from exc

    if not background_ts_kwargs:
        background_ts_kwargs = DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS

    if not target_ts_kwargs:
        target_ts_kwargs = DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS

    if not best_ts_kwargs:
        best_ts_kwargs = DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS

    if not ylabel_kwargs:
        ylabel_kwargs = {}

    best_run_d = convert_results_to_plot_dict(best_run)
    others_to_plot_d = convert_results_to_plot_dict(scmdata.run_append(others_to_plot))
    target_runs = convert_results_to_plot_dict(target)

    for k in timeseries_keys:
        ax = axes[k]
        best_k = best_run_d[k]
        background_runs = others_to_plot_d[k]
        model_unit = str(background_runs.get_unique_meta("unit", True))

        target_k = target_runs[k]
        target_k_unit = str(target_k.get_unique_meta("unit", True))
        if target_k_unit != model_unit:
            # Avoidable user side, hence warn (see
            # https://docs.python.org/3/howto/logging.html#when-to-use-logging)
            warn_msg = (
                f"Converting target units ({target_k_unit!r}) "
                f"to model output units ({model_unit!r}), "
                "this will happen every time you plot and is slow. "
                "Please convert the target units to the model's units "
                "before doing the optimisation for increased performance "
                "(the function `convert_target_to_model_output_units_scmrun` "
                "may be helpful)."
            )
            warnings.warn(warn_msg)
            target_k = target_k.convert_unit(model_unit)

        if not background_runs.empty:
            get_timeseries(background_runs).plot.line(
                ax=ax,
                **background_ts_kwargs,
            )

        get_timeseries(target_k).plot.line(
            ax=ax,
            **target_ts_kwargs,
        )

        get_timeseries(best_k).plot.line(
            ax=ax,
            **best_ts_kwargs,
        )

        ax.set_ylabel(k, **ylabel_kwargs)
