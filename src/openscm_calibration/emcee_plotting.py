"""
Support for plotting emcee during run and afterwards
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)

import numpy as np
import numpy.typing as nptype

from openscm_calibration.emcee_utils import (
    ChainProgressInfo,
    get_acceptance_fractions,
    get_autocorrelation_info,
    get_labelled_chain_data,
    get_start,
)
from openscm_calibration.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    from collections.abc import Iterator

    import emcee
    import IPython
    import matplotlib
    import matplotlib.axes
    import matplotlib.figure

DEFAULT_PROGRESS_KWARGS = {"leave": True}
"""Default arguments to use for displaying progress bars"""


def plot_chains(  # noqa: PLR0913
    inp: emcee.backends.Backend,
    burnin: int,
    parameter_order: list[str],
    neg_log_likelihood_name: str,
    axes_d: dict[str, matplotlib.axes.Axes],
    get_neg_log_likelihood_ylim: Callable[
        [np.typing.NDArray[np.floating[Any] | np.integer[Any]]], tuple[float, float]
    ]
    | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot chains in MCMC run

    The y-limits of the log likelihood axis are set dynamically

    Parameters
    ----------
    inp
        Object from which to plot the state

    burnin
        Number of iterations to treat as burn in

    parameter_order
        Order of model parameters. This must match the order used by  ``inp``.

    neg_log_likelihood_name
        Label when plotting negative log likelihood. Must match the expected
        name in ``axes_d``

    axes_d
        Axes on which to plot the chains. Must have a key for each name in
        ``parameter_order`` plus a key equal to the value of
        ``neg_log_likelihood_name``

    get_neg_log_likelihood_ylim
        Function which gets the y-limits for the negative log likelihood plot
        based on the costs. If not provided,
        :func:`get_neg_log_likelihood_ylim_default` is used

    **kwargs
        Passed to :func:`plot_parameter_chains`
    """
    labelled_data = get_labelled_chain_data(
        inp,
        parameter_order,
        neg_log_likelihood_name,
        burnin=0,
        thin=1,
    )

    if get_neg_log_likelihood_ylim is None:
        get_neg_log_likelihood_ylim = get_neg_log_likelihood_ylim_default

    for label, to_plot in labelled_data.items():
        ax = axes_d[label]
        plot_parameter_chains(
            ax,
            to_plot,
            burnin=burnin,
            **kwargs,
        )
        ax.set_ylabel(label)

        if label == neg_log_likelihood_name:
            ax.set_ylim(*get_neg_log_likelihood_ylim(to_plot))


def plot_parameter_chains(  # noqa: PLR0913
    ax: matplotlib.axes.Axes,
    chain_values: np.typing.NDArray[np.number[Any]],
    burnin: int,
    alpha_chain: float = 0.3,
    linewidth: float = 0.5,
    color: str = "0.2",
    alpha_vspan: float = 0.3,
    kwargs_chain: dict[str, Any] | None = None,
    kwargs_vspan: dict[str, Any] | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot chains for a single parameter in an MCMC run

    Parameters
    ----------
    ax
        Axes on which to plot

    chain_values
        Chain values to plot (should be 2D)

    burnin
        Number of iterations to treat as burn in

    alpha_chain
        Alpha to use for the chains

    linewidth
        Linewidth to use for the chains

    color
        Colour to use for the chains

    alpha_vspan
        Alpha to use for the vertical span (which shows the burnin period)

    kwargs_chain
        Pass to :meth:`ax.plot` when plotting the chains

    kwargs_vspan
        Pass to :meth:`ax.axvspan` when plotting the vertical span
    """
    if not kwargs_chain:
        kwargs_chain = {}

    if not kwargs_vspan:
        kwargs_vspan = {}

    ax.plot(
        chain_values,
        lw=linewidth,
        alpha=alpha_chain,
        color=color,
        **kwargs_chain,
    )
    # If burnin is 2, then we burn the first two steps
    # i.e. 0 and 1 so the span should cover 0 and 1, but
    # not 2 i.e. the span covers burnin -1, but not burnin.
    # To make this visually easier to see, we do a span that
    # ends halfway between the last step in burnin and the first
    # in the chain i.e. we end at burnin - 0.5
    ax.axvspan(0, burnin - 0.5, alpha=alpha_vspan, **kwargs_vspan)
    ax.set_xlim(0, chain_values.shape[0])

    return ax


def get_neg_log_likelihood_ylim_default(
    neg_ll_values: np.typing.NDArray[np.floating[Any] | np.integer[Any]],
    median_scaling: float = 1.5,
    max_scaling: float = 2.0,
) -> tuple[float, float]:
    r"""
    Get the y-limits for the negative log likelihood axes

    This is the default algorithm

    Parameters
    ----------
    neg_ll_values
        Negative log likelihood values being plotted

    median_scaling
        Scaling to apply to the median value

    max_scaling
        Scaling to apply to the maximum value

    Returns
    -------
        y-limits

    Notes
    -----
    The algorithm is

    .. math::

        \text{median_scaled} = \text{median_scaling} \times \text{med}(\text{neg_ll_values}) \\
        \text{max_scaled} = \text{max_scaling} \times \max(\text{neg_ll_values}) \\
        \text{ymin} = \min(0, \text{median_scaled}, \text{max_scaled}) \\
        \text{ymax} = \max(0, \text{median_scaled}, \text{max_scaled})
    """  # noqa: E501
    median_scaled = float(median_scaling * np.median(neg_ll_values))
    if not np.isfinite(median_scaled):
        median_scaled = 0.0

    max_scaled = float(max_scaling * np.max(neg_ll_values))
    if not np.isfinite(max_scaled):
        max_scaled = 0.0

    ymin = float(
        min(
            0.0,
            median_scaled,
            max_scaled,
        )
    )
    ymax = float(
        max(
            0.0,
            median_scaled,
            max_scaled,
        )
    )

    return (ymin, ymax)


def plot_dist(  # noqa: PLR0913
    inp: emcee.backends.Backend,
    burnin: int,
    thin: int,
    parameter_order: list[str],
    axes_d: dict[str, matplotlib.axes.Axes],
    common_norm: bool = False,
    fill: bool = True,
    legend: bool = False,
    **kwargs: Any,
) -> None:
    """
    Plot distributions from MCMC run

    This is a thin wrapper around :func:`sns.kdeplot` that sets helpful
    defaults.

    Parameters
    ----------
    inp
        Object from which to plot the state

    burnin
        Number of iterations to treat as burn in

    thin
        Thinning to use when sampling the chains

    parameter_order
        Order of model parameters

    axes_d
        Axes on which to plot the distributions

    common_norm
        Should all the distributions use the same normalisation? We generally
        set this to ``False`` as we want to see the chains independently. See
        docstring of :func:`sns.kdeplot` for full details.

    fill
        Should the KDE plots be filled? See docstring of :func:`sns.kdeplot`
        for full details.

    legend
        Should a legend be added to the plots? The legend is pretty
        meaningless (it is just the chain numbers) so we generally set this to
        ``False``. See docstring of :func:`sns.kdeplot` for full details.

    **kwargs
        Passed to :func:`sns.kdeplot`.
    """
    try:
        import seaborn as sns
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plot_dist", requirement="seaborn"
        ) from exc

    burnt_in_samples_labelled = get_labelled_chain_data(
        inp,
        parameter_order,
        burnin=burnin,
        thin=thin,
    )

    for label, to_plot in burnt_in_samples_labelled.items():
        ax = axes_d[label]

        sns.kdeplot(
            data=to_plot,
            ax=ax,
            common_norm=common_norm,
            fill=fill,
            legend=legend,
            **kwargs,
        )
        ax.set_xlabel(label)


DEFAULT_PLOT_CORNER_TITLE_KWARGS: dict[str, Any] = {"fontsize": 12}
"""
Default value for ``title_kwargs`` used by ``plot_corner``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


DEFAULT_PLOT_CORNER_LABEL_KWARGS: dict[str, Any] = {"fontsize": "x-small"}
"""
Default value for ``label_kwargs`` used by ``plot_corner``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


def plot_corner(  # noqa: PLR0913,too-many-locals
    inp: emcee.backends.Backend,
    burnin: int,
    thin: int,
    parameter_order: list[str],
    fig: matplotlib.figure.Figure,
    bins: int = 30,
    plot_contours: bool = True,
    smooth: bool = True,
    quantiles: Sequence[float] = (0.05, 0.17, 0.5, 0.83, 0.95),
    show_titles: bool = True,
    title_quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    title_kwargs: dict[str, Any] | None = None,
    title_fmt: str = ".3f",
    label_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot corner plot of distribution from MCMC run

    This is a thin wrapper around :func:`corner.corner` which uses some
    sensible defaults.

    Parameters
    ----------
    inp
        Object from which to plot the state

    burnin
        Number of iterations to treat as burn in

    thin
        Thinning to use when sampling the chains

    parameter_order
        Order of model parameters

    fig
        Figure to use for plotting, should be empty i.e. have been cleared
        before being passed here

    bins
        Number of bins to use in histograms. See docstring of
        :func:`corner.corner` for full details.

    plot_contours
        Whether to plot contours on the 2D distribution plots. See docstring
        of :func:`corner.corner` for full details.

    smooth
        Whether to smooth the contours on the 2D distribution plots. See
        docstring of :func:`corner.corner` for full details.

    quantiles
        Quantiles at which to draw vertical lines in the histogram plots. See
        docstring of :func:`corner.corner` for full details.

    show_titles
        Whether to show titles on the histogram plots. See docstring of
        :func:`corner.corner` for full details.

    title_quantiles
        Quantiles to put in the titles of the histogram plots. See docstring
        of :func:`corner.corner` for full details.

    title_kwargs
        Keyword arguments to use when making the titles on the histogram
        plots. If not supplied, we use
        ``DEFAULT_PLOT_CORNER_TITLE_KWARGS``.

    title_fmt
        Format string to use when creating the titles on the histogram plots.
        If not supplied, our own internal defaults are used (see source code
        for values). See docstring of :func:`corner.corner` for full details.

    label_kwargs
        Keyword arguments to use when creating the labels for the plot. If not
        supplied, we use ``DEFAULT_PLOT_CORNER_LABEL_KWARGS``.

    **kwargs
        Passed to :func:`corner.corner`
    """
    try:
        import corner
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            "plot_corner", requirement="corner"
        ) from exc

    if title_kwargs is None:
        title_kwargs = DEFAULT_PLOT_CORNER_TITLE_KWARGS

    if label_kwargs is None:
        label_kwargs = DEFAULT_PLOT_CORNER_LABEL_KWARGS

    burnt_in_samples_flat = inp.get_chain(discard=burnin, thin=thin, flat=True)

    corner.corner(
        burnt_in_samples_flat,
        labels=parameter_order,
        fig=fig,
        bins=bins,
        plot_contours=plot_contours,
        smooth=smooth,
        quantiles=quantiles,
        show_titles=show_titles,
        title_quantiles=title_quantiles,
        title_kwargs=title_kwargs,
        title_fmt=title_fmt,
        label_kwargs=label_kwargs,
        **kwargs,
    )


def plot_tau(  # noqa: PLR0913
    ax: matplotlib.axes.Axes,
    autocorr: nptype.NDArray[np.float64],
    steps: nptype.NDArray[np.int64],
    parameter_order: tuple[str, ...],
    convergence_ratio: float,
    convergence_ratio_line_kwargs: dict[str, Any] | None = None,
    legend_loc: str | None = "lower right",
    xlabel: str = "Number steps (post burnin)",
    ylabel: str = "Autocorrelation time, tau",
) -> None:
    """
    Plot the autocorrelation time, tau

    Parameters
    ----------
    ax
        Axes on which to plot

    autocorr
        Autocorrelation information.

        This should be multi-dimensional,
        with each column being the autocorrelation time for a different parameter.

    steps
        The number of steps taken before each row in `autocorr` was calculated.

        In other words, the x-axis for the plot.

    parameter_order
        The order of the parameters in `autocorr`.

    convergence_ratio
        Convergence ratio (used to show the convergence line on the plot)

    convergence_ratio_line_kwargs
        Keyword arguments to pass to [`axline`][matplotlib.axes.Axes.axline]
        when plotting the convergence ratio line.

        If not supplied, we use `{"color": "k", "linestyle": "--"}`.

    legend_loc
        Location of the legend.

        If `None`, we don't explicitly set the legend location.

    xlabel
        x-label to apply to the plot.

    ylabel
        y-label to apply to the plot.
    """
    if convergence_ratio_line_kwargs is None:
        convergence_ratio_line_kwargs = {"color": "k", "linestyle": "--"}

    ax.plot(
        steps,
        autocorr,
        label=parameter_order,
    )
    ax.axline((0, 0), slope=1 / convergence_ratio, **convergence_ratio_line_kwargs)

    if legend_loc is not None:
        ax.legend(loc=legend_loc)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# TODO: simplify this
def plot_emcee_progress(  # noqa: PLR0913
    sampler: emcee.ensemble.EnsembleSampler,
    iterations: int,
    burnin: int,
    thin: int,
    plot_every: int,
    parameter_order: tuple[str, ...],
    neg_log_likelihood_name: str,
    holder_chain: IPython.core.display_functions.DisplayHandle,
    figure_chain: matplotlib.figure.Figure,
    axes_chain: dict[str, matplotlib.axes.Axes],
    holder_dist: IPython.core.display_functions.DisplayHandle,
    figure_dist: matplotlib.figure.Figure,
    axes_dist: dict[str, matplotlib.axes.Axes],
    holder_corner: IPython.core.display_functions.DisplayHandle,
    figure_corner: matplotlib.figure.Figure,
    holder_tau: IPython.core.display_functions.DisplayHandle,
    figure_tau: matplotlib.figure.Figure,
    ax_tau: matplotlib.axes.Axes,
    start: nptype.NDArray[np.float64] | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    progress: str | bool = "notebook",
    progress_kwargs: dict[str, Any] | None = None,
    min_samples_before_plot: int = 2,
    corner_kwargs: dict[str, Any] | None = None,
    convergence_ratio: float = 50,
) -> Iterator[ChainProgressInfo]:
    """
    Plot MCMC progress

    Parameters
    ----------
    sampler
        Sampler used for sampling the posterior distribution

    iterations
        Number of iterations to perform

    burnin
        Burn-in to apply to the chains

    thin
        Thinning to apply to the chains

    plot_every
        How many steps to wait before updating the plots

    parameter_order
        Order of the parameters.

        This is used for ensuring that the plots are labelled correctly.

    neg_log_likelihood_name
        Name to use for the negative log likelihood in plots

    holder_chain
        Holder of the figure that displays the chains

    figure_chain
        Figure that displays the chains

    axes_chain
        Axes on which to plot the chains.

        Each parameter in `parameter_order` should be a key in `axes_chain`.

    holder_dist
        Holder of the figure that displays the parameter distributions

    figure_dist
        Figure that displays the parameter distributions

    axes_dist
        Axes on which to plot the parameter distributions.

        Each parameter in `parameter_order` should be a key in `axes_dist`.

    holder_corner
        Holder of the figure that displays the corner plot

    figure_corner
        Figure that displays the corner plot

    holder_tau
        Holder of the figure that displays the autocorrelation

    figure_tau
        Figure that displays the autocorrelation

    ax_tau
        Axes on which to plot the autocorrelation

    start
        Starting point for the sampling.

        Only required if the sampler has not performed iterations already.

    sample_kwargs
        Arguments to pass to `sampler.sample`.

    progress
        Whether to show a progress bar or not.

        If this is a string, it must match the values supported by `sampler.sample`.

    progress_kwargs
        Arguments to pass to the progress bar, if used.

    min_samples_before_plot
        Minimum number of samples that must be taken before any plot can be made.

    corner_kwargs
        Passed to [`plot_corner`][openscm_calibration.emcee_plotting.plot_corner].

    convergence_ratio
        Ratio to use to check whether the chains have converged or not.

        Passed to
        [`get_autocorrelation_info`][openscm_calibration.emcee_utils.get_autocorrelation_info].

    Yields
    ------
    :
        Information about the chain's progress.

        The yield occurs each time the plots are updated,
        so can also be used as a way to perform other actions.
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    if progress_kwargs is None:
        progress_kwargs = DEFAULT_PROGRESS_KWARGS

    if corner_kwargs is None:
        corner_kwargs = {}

    start_use = get_start(
        sampler=sampler,
        start=start,
    )

    # Stores for autocorrelation values
    # (as a function of the number of steps performed).
    # This reserves more memory than is needed,
    # but it is so cheap for a plain array that we don't worry about it.
    autocorr = np.zeros((iterations, sampler.ndim))
    autocorr_steps = np.zeros(iterations)
    autocorr_index = 0

    # Helper function
    def enough_autocorr_values_to_plot(autocorr: nptype.NDArray[np.float64]) -> bool:
        """Check whether there is any auto-correlation information to plot"""
        values_to_plot_present = np.logical_and(
            autocorr > 0.0,  # noqa: PLR2004
            np.logical_not(np.isnan(autocorr)),
        )

        return np.any(np.sum(values_to_plot_present, axis=0) > 1)

    for sample in sampler.sample(
        start_use,
        iterations=iterations,
        progress=progress,
        progress_kwargs={"leave": True},
        **sample_kwargs,
    ):
        if (
            sampler.iteration % plot_every
            or sampler.iteration < min_samples_before_plot
        ):
            continue

        for ax in axes_chain.values():
            ax.clear()

        plot_chains(
            inp=sampler,
            burnin=burnin,
            parameter_order=parameter_order,
            axes_d=axes_chain,
            neg_log_likelihood_name=neg_log_likelihood_name,
        )
        figure_chain.tight_layout()
        holder_chain.update(figure_chain)

        steps_post_burnin = max(0, sampler.iteration - burnin)
        if steps_post_burnin < 1:
            # In burn in
            acceptance_fraction = np.nan

        else:
            chain_post_burnin = sampler.get_chain(discard=burnin)
            acceptance_fraction = np.mean(get_acceptance_fractions(chain_post_burnin))

            for ax in axes_dist.values():
                ax.clear()

                plot_dist(
                    inp=sampler,
                    burnin=burnin,
                    thin=thin,
                    parameter_order=parameter_order,
                    axes_d=axes_dist,
                    warn_singular=False,
                )

            figure_dist.tight_layout()
            holder_dist.update(figure_dist)

            # Not sure why this was wrapped in try-except in the notebooks.
            # Might want to re-instate in future.
            # try:
            figure_corner.clear()
            plot_corner(
                inp=sampler,
                burnin=burnin,
                thin=thin,
                parameter_order=parameter_order,
                fig=figure_corner,
                **corner_kwargs,
            )
            figure_corner.tight_layout()
            holder_corner.update(figure_corner)
            # except AssertionError:
            #     pass

            autocorr_bits = get_autocorrelation_info(
                sampler,
                burnin=burnin,
                thin=thin,
                convergence_ratio=convergence_ratio,
            )

            if autocorr_bits.any_non_nan_tau():
                autocorr[autocorr_index, :] = autocorr_bits.tau
                autocorr_steps[autocorr_index] = autocorr_bits.steps_post_burnin
                autocorr_index += 1

            if enough_autocorr_values_to_plot(autocorr):
                ax_tau.clear()
                plot_tau(
                    ax=ax_tau,
                    autocorr=autocorr[:autocorr_index, :],
                    steps=autocorr_steps[:autocorr_index],
                    parameter_order=parameter_order,
                    convergence_ratio=convergence_ratio,
                    convergence_ratio_line_kwargs=dict(color="k", linestyle="--"),
                )

                figure_tau.tight_layout()
                holder_tau.update(figure_tau)

        yield ChainProgressInfo(
            steps=sampler.iteration,
            steps_post_burnin=steps_post_burnin,
            acceptance_fraction=acceptance_fraction,
        )
