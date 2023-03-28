"""
Support for plotting during scipy optimisation
"""
from __future__ import annotations

import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
import scmdata
from attrs import define, field

if TYPE_CHECKING:
    import attr
    import matplotlib
    import numpy.typing as nptype
    import pandas as pd
    import scmdata.run
    import tqdm

    import openscm_calibration.store


logger: logging.Logger = logging.getLogger(__name__)
"""Logger for this module"""


class SupportsScipyOptCallback(Protocol):
    """
    Class that supports being used as a callback with Scipy's optimisers
    """

    def callback_minimize(
        self,
        xk: nptype.NDArray[Union[np.float_, np.int_]],  # pylint:disable=invalid-name
    ) -> None:
        """
        Get cost of parameter vector

        This callback is intended to be used with `scipy.optimize.minimize`

        Parameters
        ----------
        xk
            Last used parameter vector
        """

    def callback_differential_evolution(
        self,
        xk: nptype.NDArray[Union[np.float_, np.int_]],  # pylint:disable=invalid-name
        convergence: Optional[float] = None,
    ) -> None:
        """
        Get cost of parameter vector

        This callback is intended to be used with
        `scipy.optimize.differential_evolution`

        Parameters
        ----------
        xk
            Parameter vector with best solution found so far

        convergence
            Received from :func:`scipy.optimize.differential_evolution`
            on callback. We are not sure what this does is or is used for.
        """


class SupportsFigUpdate(Protocol):  # pylint:disable=too-few-public-methods
    """
    Class that supports updating figures

    For example, :class:`IPython.core.display_functions.DisplayHandle`
    """

    def update(
        self,
        fig: matplotlib.figure.Figure,
    ) -> None:
        """
        Update the figure

        Parameters
        ----------
        fig
            Figure to update
        """


ScmRunToDictConverter = Callable[
    [scmdata.run.BaseScmRun], Dict[str, scmdata.run.BaseScmRun]
]
"""
Type hint helper for functions used as ``convert_scmrun_to_plot_dict``

Such functions convert an :obj:`scmdata.run.BaseScmRun` to a dictionary of
:obj:`scmdata.run.BaseScmRun`
"""


def _all_in_axes(
    instance: OptPlotter,  # pylint: disable=unused-argument
    attribute: attr.Attribute[Tuple[str]],
    value: Tuple[str],
) -> None:
    """
    Check all values are present in ``self.axes``

    Parameters
    ----------
    self
        Object instance

    attribute
        Attribute to check

    value
        Value to check

    Raises
    ------
    ValueError
        Not all elements in ``value`` are keys in ``self.axes``
    """
    missing_from_axes = [k for k in value if k not in instance.axes]
    if missing_from_axes:
        error_msg = (
            f"Error setting '{attribute.name}'. "
            "The following keys are missing from ``self.axes``: "
            f"``{missing_from_axes}``. "
            f"``self.axes.keys()`` is ``{instance.axes.keys()}``."
        )
        raise KeyError(error_msg)


def _compatible_with_convert_and_target(
    instance: OptPlotter,  # pylint: disable=unused-argument
    attribute: attr.Attribute[Tuple[str]],
    value: Tuple[str],
) -> None:
    """
    Check that the values are compatible with the target and the ScmRun conversion

    Specifically, compatible with ``self.target`` and
    ``self.convert_scmrun_to_plot_dict``

    Parameters
    ----------
    self
        Object instance

    attribute
        Attribute to check

    value
        Value to check

    Raises
    ------
    ValueError
        If the values in ``value`` are not all keys in the dictionary that is
        returned when ``self.convert_scmrun_to_plot_dict`` is applied to
        ``self.target``.
    """
    target_converted = instance.convert_scmrun_to_plot_dict(instance.target)
    missing_from_target_converted = [k for k in value if k not in target_converted]
    if missing_from_target_converted:
        error_msg = (
            f"Error setting '{attribute.name}'. "
            "The following keys are missing when "
            "``self.convert_scmrun_to_plot_dict`` is applied to "
            f"``self.target``: ``{missing_from_target_converted}``. "
            f"``target_converted.keys()`` is ``{target_converted.keys()}``."
        )
        raise ValueError(error_msg)


@define
class OptPlotter:
    """
    Optimisation progress plotting helper

    This class is an adapter between interfaces required by Scipy's callback
    arguments and updating the plots. The class holds all the information
    required to make useful plots. It is intended to be used in interactive
    Python i.e. to make updating plots.
    """

    holder: SupportsFigUpdate
    """Figure updater, typically :obj:`IPython.core.display_functions.DisplayHandle`"""

    fig: matplotlib.figure.Figure
    """Figure on which to plot"""

    axes: Dict[str, matplotlib.axes.Axes]
    """
    Dictionary storing axes on which to plot

    The plot of the cost function over time will be plotted on the axes with
    key given by ``cost_key``.

    Each parameter will be plotted on the axes with the same key as the
    parameter (as defined in ``parameters``)

    The timeseries will be plotted on the axes specified by
    ``timeseries_axes``. See docstring of ``timeseries_axes`` for rules about
    its values.
    """

    cost_key: str
    """Key for the axes on which the cost function should be plotted"""

    parameters: Tuple[str] = field(validator=[_all_in_axes])
    """
    Parameters to be optimised

    This must match the order in which the parameters are handled by the
    optimiser i.e. it is used to translate the unlabeled array of parameter
    values onto the desired axes.
    """

    timeseries_axes: Tuple[str] = field(
        validator=[_all_in_axes, _compatible_with_convert_and_target]
    )
    """
    Axes on which to plot timeseries

    The timeseries in ``target`` and ``store.res`` are
    converted into dictionaries using ``convert_scmrun_to_plot_dict``. The
    keys of the result of ``convert_scmrun_to_plot_dict`` must match the
    values in ``timeseries_axes``.
    """

    convert_scmrun_to_plot_dict: ScmRunToDictConverter
    """
    Callable which converts :obj:`scmdata.run.BaseScmRun` into a dictionary in
    which the keys are a subset of the values in ``timeseries_axes``
    """

    target: scmdata.run.BaseScmRun
    """Target to which we are optimising"""

    store: openscm_calibration.store.OptResStore
    """Optimisation result store"""

    thin_ts_to_plot: int = 20
    """
    Thinning to apply to the timeseries to plot

    In other words, only plot every ``thin_ts_to_plot`` runs on the timeseries
    plots. Plotting all runs can be very expensive.
    """

    plot_cost_kwargs: Optional[Dict[str, Any]] = None
    """Kwargs to pass to :func:`plot_costs`."""

    plot_parameters_kwargs: Optional[Dict[str, Any]] = None
    """Kwargs to pass to :func:`plot_parameters`."""

    plot_timeseries_kwargs: Optional[Dict[str, Any]] = None
    """Kwargs to pass to :func:`plot_timeseries`."""

    get_timeseries: Optional[Callable[[scmdata.run.BaseScmRun], pd.DataFrame]] = None
    """
    Function which converts :obj:`scmdata.run.BaseScmRun` into a :obj:`pd.DataFrame` for plotting

    If not provided, :func:`get_timeseries_default` is used
    """

    def callback_minimize(
        self,
        xk: nptype.NDArray[  # pylint:disable=invalid-name, unused-argument
            Union[np.float_, np.int_]
        ],
    ) -> None:
        """
        Update the plots

        Intended to be used as the `callback` argument to
        `scipy.optimize.minimize`

        Parameters
        ----------
        xk
            Last used parameter vector
        """
        self.update_plots()

    def callback_differential_evolution(
        self,
        xk: nptype.NDArray[  # pylint:disable=invalid-name, unused-argument
            Union[np.float_, np.int_]
        ],
        convergence: Optional[float] = None,  # pylint:disable=unused-argument
    ) -> None:
        """
        Update the plots

        Intended to be used as the `callback` argument to
        `scipy.optimize.differential_evolution`

        Parameters
        ----------
        xk
            Parameter vector with best solution found so far

        convergence
            Received from :func:`scipy.optimize.differential_evolution`
            on callback. We are not sure what this does is or what it is used
            for.
        """
        self.update_plots()

    def update_plots(self) -> None:
        """
        Update all the plots
        """
        costs, para_vals, res = self.store.get_costs_labelled_xsamples_res()

        # check if anything to plot
        if np.all(~np.isfinite(costs)):
            logger.info("No runs succeeded, nothing to plot")
            return

        # plot cost function
        ax_cost = self.axes[self.cost_key]
        ax_cost.clear()

        plot_cost_kwargs = (
            self.plot_cost_kwargs if self.plot_cost_kwargs is not None else {}
        )
        plot_costs(ax=ax_cost, ylabel=self.cost_key, costs=costs, **plot_cost_kwargs)

        # plot parameters
        for parameter in self.parameters:
            self.axes[parameter].clear()

        plot_parameters_kwargs = (
            self.plot_parameters_kwargs
            if self.plot_parameters_kwargs is not None
            else {}
        )
        plot_parameters(
            axes=self.axes,
            para_vals=para_vals,
            **plot_parameters_kwargs,
        )

        # plot timeseries
        best_run, others_to_plot = get_runs_to_plot(costs, res, self.thin_ts_to_plot)

        for k in self.timeseries_axes:
            self.axes[k].clear()

        plot_timeseries_kwargs = (
            self.plot_timeseries_kwargs
            if self.plot_timeseries_kwargs is not None
            else {}
        )

        if self.get_timeseries is not None:
            get_timeseries = self.get_timeseries
        else:
            get_timeseries = get_timeseries_default

        plot_timeseries(
            best_run=best_run,
            others_to_plot=others_to_plot,
            target=self.target,
            convert_scmrun_to_plot_dict=self.convert_scmrun_to_plot_dict,
            timeseries_keys=self.timeseries_axes,
            axes=self.axes,
            get_timeseries=get_timeseries,
            **plot_timeseries_kwargs,
        )

        # update and return
        self.fig.tight_layout()
        self.holder.update(self.fig)


def get_timeseries_default(
    inp: scmdata.run.BaseScmRun,
    time_axis: str = "year-month",
) -> pd.DataFrame:
    """
    Get timeseries for plotting from an :obj:`scmdata.run.BaseScmRun`

    This is the default function for doing this conversion in this module

    Parameters
    ----------
    inp
        :obj:`scmdata.run.BaseScmRun` to convert

    time_axis
        Passed to :meth:`inp.timeseries` when doing the
        conversion

    Returns
    -------
        Data with the time axis as rows, ready for simplified plotting using
        panda's plotting methods.
    """
    return inp.timeseries(time_axis=time_axis).T


def plot_costs(  # pylint:disable=too-many-arguments
    ax: matplotlib.axes.Axes,  # pylint:disable=invalid-name
    ylabel: str,
    costs: Tuple[float, ...],
    ymin: float = 0.0,
    get_ymax: Optional[Callable[[Tuple[float, ...]], float]] = None,
    alpha: float = 0.7,
    **kwargs: Any,
) -> None:
    r"""
    Plot cost function

    Parameters
    ----------
    ax
        Axes on which to plot

    ylabel
        y-axis label

    costs
        Costs to plot

    ymin
        Minimum y-axis value

    get_ymax
        Function which gets the y-max based on the costs. If not provided,
        :func:`get_ymax_default` is used

    alpha
        Alpha to apply to plotted points

    **kwargs
        Passed to :meth:`ax.scatter`
    """
    if get_ymax is None:
        get_ymax = get_ymax_default

    ax.scatter(range(len(costs)), costs, alpha=alpha, **kwargs)
    ax.set_ylabel(ylabel)

    ymax = get_ymax(costs)
    if not np.isfinite(ymax):
        ymax = 10**3

    ax.set_ylim(
        ymin=ymin,
        ymax=ymax,
    )


def get_ymax_default(
    costs: Tuple[float, ...],
    min_scale_factor: float = 10.0,
    min_v_median_scale_factor: float = 2.0,
) -> float:
    r"""
    Get y-max based on costs

    This is the default function used by :func:`plot_costs`. The algorithm is

    .. math::

        \text{ymax} = min(
            \text{min_scale_factor} \times min(costs),
            max(
                median(costs),
                \text{min_v_median_scale_factor} \times min(costs)
            )
        )

    Parameters
    ----------
    costs
        Cost function values

    min_scale_factor
        Value by which the minimum value is scaled when determining the plot
        limits

    min_v_median_scale_factor
        Value by which the minimum value is scaled when comparing to the
        median as part of determining the plot limits

    Returns
    -------
        Maximum value to use on the y-axis
    """
    min_cost = np.min(costs)
    ymax = np.min(
        [
            min_scale_factor * min_cost,
            np.max([np.median(costs), min_v_median_scale_factor * min_cost]),
        ]
    )

    return float(ymax)


def plot_parameters(
    axes: Dict[str, matplotlib.axes.Axes],
    para_vals: Dict[str, nptype.NDArray[Union[np.float_, np.int_]]],
    alpha: float = 0.7,
    **kwargs: Any,
) -> None:
    """
    Plot parameters

    Parameters
    ----------
    axes
        Axes on which to plot. The keys should match the keys in ``para_vals``

    para_vals
        Parameter values. Each key should be the name of a parameter

    alpha
        Alpha to use when calling :meth:`matplotlib.axes.Axes.scatter`

    **kwargs
        Passed to each call to :meth:`matplotlib.axes.Axes.scatter`
    """
    for parameter, values in para_vals.items():
        axes[parameter].scatter(range(len(values)), values, alpha=alpha, **kwargs)
        axes[parameter].set_ylabel(parameter)


DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS: Dict[str, Any] = {
    "legend": False,
    "linewidth": 0.5,
    "alpha": 0.3,
    "color": "tab:gray",
    "zorder": 1.5,
}
"""
Default value for ``background_ts_kwargs`` used by ``plot_timeseries``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS: Dict[str, Any] = {
    "legend": False,
    "linewidth": 2,
    "alpha": 0.8,
    "color": "tab:blue",
    "zorder": 2,
}
"""
Default value for ``target_ts_kwargs`` used by ``plot_timeseries``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS: Dict[str, Any] = {
    "legend": False,
    "linewidth": 2,
    "alpha": 0.8,
    "color": "tab:orange",
    "zorder": 2.5,
}
"""
Default value for ``best_ts_kwargs`` used by ``plot_timeseries``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


def plot_timeseries(  # pylint:disable=too-many-arguments,too-many-locals
    best_run: scmdata.run.BaseScmRun,
    others_to_plot: scmdata.run.BaseScmRun,
    target: scmdata.run.BaseScmRun,
    convert_scmrun_to_plot_dict: Callable[
        [scmdata.run.BaseScmRun], Dict[str, scmdata.run.BaseScmRun]
    ],
    timeseries_keys: Iterable[str],
    axes: Dict[str, matplotlib.axes.Axes],
    get_timeseries: Callable[[scmdata.run.BaseScmRun], pd.DataFrame],
    background_ts_kwargs: Optional[Dict[str, Any]] = None,
    target_ts_kwargs: Optional[Dict[str, Any]] = None,
    best_ts_kwargs: Optional[Dict[str, Any]] = None,
    ylabel_kwargs: Optional[Dict[str, Any]] = None,
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

    convert_scmrun_to_plot_dict
        Callable which converts :obj:`scmdata.run.BaseScmRun` into a
        dictionary in which the keys are a subset of the values in
        ``axes``

    timeseries_keys
        Keys of the timeseries to plot

    axes
        Axes on which to plot

    get_timeseries
        Function which converts :obj:`scmdata.run.BaseScmRun` into a
        :obj:`pd.DataFrame` for plotting

    background_ts_kwargs
        Passed to :meth:`pd.DataFrame.plot.line` when plotting the background
        timeseries. If not supplied, we use
        ``DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS``.

    target_ts_kwargs
        Passed to :meth:`pd.DataFrame.plot.line` when plotting the target
        timeseries. If not supplied, we use
        ``DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS``.

    best_ts_kwargs
        Passed to :meth:`pd.DataFrame.plot.line` when plotting the best
        timeseries. If not supplied, we use
        ``DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS``.

    ylabel_kwargs
        Passed to :meth:`ax.set_ylabel` when setting the y-labels of each panel
    """
    if not background_ts_kwargs:
        background_ts_kwargs = DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS

    if not target_ts_kwargs:
        target_ts_kwargs = DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS

    if not best_ts_kwargs:
        best_ts_kwargs = DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS

    if not ylabel_kwargs:
        ylabel_kwargs = {}

    best_run_d = convert_scmrun_to_plot_dict(best_run)
    others_to_plot_d = convert_scmrun_to_plot_dict(others_to_plot)
    target_runs = convert_scmrun_to_plot_dict(target)

    for k in timeseries_keys:
        ax = axes[k]  # pylint: disable=invalid-name
        best_k = best_run_d[k]
        background_runs = others_to_plot_d[k]
        model_unit = background_runs.get_unique_meta("unit", True)

        target_k = target_runs[k]
        target_k_unit = target_k.get_unique_meta("unit", True)
        if target_k_unit != model_unit:
            # Avoidable user side, hence warn (see
            # https://docs.python.org/3/howto/logging.html#when-to-use-logging)
            warn_msg = (
                f"Converting target units ('{target_k_unit}') to model output "
                f"units ('{model_unit}'), this will happen every time you "
                "plot and is slow. Please convert the target units to the "
                "model's units before doing the optimisation for increased "
                "performance (the function "
                "``convert_target_to_model_output_units`` may be helpful)."
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


def get_runs_to_plot(
    costs: Tuple[float, ...],
    res: Tuple[scmdata.run.BaseScmRun, ...],
    thin_ts_to_plot: int,
) -> Tuple[scmdata.run.BaseScmRun, scmdata.run.BaseScmRun]:
    """
    Get runs to plot

    This retrieves the run which best matches the target (has lowest cost) and
    then a series of others to plot

    Parameters
    ----------
    costs
        Cost function value for each run (used to determine the best result)

    res
        Results of each run. It is assumed that the elements in ``res`` and
        ``costs`` line up i.e. the nth element of ``costs`` is the cost
        function for the nth element in ``res``

    thin_ts_to_plot
        Thinning to apply to the timeseries to plot

        In other words, only plot every ``thin_ts_to_plot`` runs on the
        timeseries plots. Plotting all runs can be very expensive.

    Returns
    -------
        Best iteration and then other runs to plot

    Raises
    ------
    ValueError
        No successful runs are included in ``res``
    """
    # Convert to dict for quicker lookup later
    res_d_success = {i: v for i, v in enumerate(res) if v is not None}
    if not res_d_success:
        raise ValueError("No successful runs, please check")

    best_it = int(np.argmin(costs))
    out_best = res_d_success[best_it]

    success_keys = list(res_d_success.keys())
    to_plot_not_best = success_keys[
        len(success_keys) - 1 :: -thin_ts_to_plot  # noqa: E203 (prefer black)
    ]
    if best_it in to_plot_not_best:
        to_plot_not_best.remove(best_it)

    res_not_best = [res_d_success[i] for i in to_plot_not_best]
    out_not_best = scmdata.run_append(res_not_best)

    return out_best, out_not_best


@define
class CallbackProxy:
    """
    Callback helper

    This class acts as a proxy and decides whether the real callback should
    actually be called. If provided, it also keeps track of the number of
    model calls via a progress bar.
    """

    real_callback: SupportsScipyOptCallback
    """Callback to be called if a sufficient number of runs have been done"""

    store: openscm_calibration.store.OptResStore
    """Optimisation result store"""

    last_callback_val: int = 0
    """Number of model calls at last callback"""

    update_every: int = 50
    """Update the plots every X calls to the model"""

    progress_bar: Optional[tqdm.std.tqdm] = None
    """Progress bar to track iterations"""

    def callback_minimize(
        self,
        xk: nptype.NDArray[Union[np.float_, np.int_]],  # pylint:disable=invalid-name
    ) -> None:
        """
        Update the plots

        Intended to be used as the `callback` argument to
        `scipy.optimize.minimize`

        Parameters
        ----------
        xk
            Last used parameter vector
        """
        if self.time_to_call_real_callback():
            self.real_callback.callback_minimize(xk)

    def callback_differential_evolution(
        self,
        xk: nptype.NDArray[Union[np.float_, np.int_]],  # pylint:disable=invalid-name
        convergence: Optional[float] = None,
    ) -> None:
        """
        Update the plots

        Intended to be used as the `callback` argument to
        `scipy.optimize.differential_evolution`

        Parameters
        ----------
        xk
            Parameter vector with best solution found so far

        convergence
            Received from :func:`scipy.optimize.differential_evolution`
            on callback. Not sure what this does is or is used for.
        """
        if self.time_to_call_real_callback():
            self.real_callback.callback_differential_evolution(xk, convergence)

    def time_to_call_real_callback(self) -> bool:
        """
        Check whether it is time to call the real callback

        Returns
        -------
            ``True`` if the real callback should be called
        """
        n_model_calls = sum(x is not None for x in self.store.x_samples)
        if self.progress_bar:
            self.update_progress_bar(n_model_calls)

        if n_model_calls < self.last_callback_val + self.update_every:
            return False

        # Note that we ran the full callback
        self.last_callback_val = n_model_calls

        return True

    def update_progress_bar(self, n_model_calls: int) -> None:
        """
        Update the progress bar

        Parameters
        ----------
        n_model_calls
            Number of model calls in total

        Raises
        ------
        ValueError
            ``self.progress_bar`` is not set
        """
        if not self.progress_bar:
            raise ValueError("``self.progress_bar`` is not set")

        self.progress_bar.update(n_model_calls - self.progress_bar.last_print_n)


def convert_target_to_model_output_units(
    *,
    target: scmdata.run.BaseScmRun,
    model_output: scmdata.run.BaseScmRun,
    convert_scmrun_to_plot_dict: ScmRunToDictConverter,
) -> scmdata.run.BaseScmRun:
    """
    Convert the target data to the model output's units

    This is a helper function that allows the data to be lined up before
    setting up plotting etc.

    Parameters
    ----------
    target
        Target data to convert

    model_output
        A sample of the model's output

    convert_scmrun_to_plot_dict
        The function that will be used to convert
        :obj:`scmdata.run.BaseScmRun` to a dictionary when doing the plotting

    Returns
    -------
        Target data with units that match the model output
    """
    target_d = convert_scmrun_to_plot_dict(target)
    model_output_d = convert_scmrun_to_plot_dict(model_output)

    tmp = []
    for group, run in target_d.items():
        model_unit = model_output_d[group].get_unique_meta("unit", True)
        run_converted = run.convert_unit(model_unit)

        tmp.append(run_converted)

    out = scmdata.run_append(tmp)

    return out
