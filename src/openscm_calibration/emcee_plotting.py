"""
Support for plotting emcee during run and afterwards
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from openscm_calibration.emcee_utils import get_labelled_chain_data

try:
    import corner

    HAS_CORNER = True
except ImportError:  # pragma: no cover
    HAS_CORNER = False

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:  # pragma: no cover
    HAS_SEABORN = False

if TYPE_CHECKING:
    import emcee
    import matplotlib
    import numpy.typing as nptype


def plot_chains(  # pylint:disable=too-many-arguments
    inp: Union[emcee.backends.Backend],
    burnin: int,
    parameter_order: List[str],
    neg_log_likelihood_name: str,
    axes_d: Dict[str, matplotlib.axes.Axes],
    get_neg_log_likelihood_ylim: Optional[
        Callable[[nptype.NDArray[Union[np.float_, np.int_]]], Tuple[float, float]]
    ] = None,
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
        ax = axes_d[label]  # pylint:disable=invalid-name
        plot_parameter_chains(
            ax,
            to_plot,
            burnin=burnin,
            **kwargs,
        )
        ax.set_ylabel(label)

        if label == neg_log_likelihood_name:
            ax.set_ylim(*get_neg_log_likelihood_ylim(to_plot))


def plot_parameter_chains(  # pylint:disable=too-many-arguments
    ax: matplotlib.Axes.axes,  # pylint:disable=invalid-name
    chain_values: nptype.NDArray[Union[np.float_, np.int_]],
    burnin: int,
    alpha_chain: float = 0.3,
    linewidth: float = 0.5,
    color: str = "0.2",
    alpha_vspan: float = 0.3,
    kwargs_chain: Optional[Dict[str, Any]] = None,
    kwargs_vspan: Optional[Dict[str, Any]] = None,
) -> matplotlib.Axes.axes:
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
    neg_ll_values: nptype.NDArray[Union[np.float_, np.int_]],
    median_scaling: float = 1.5,
    max_scaling: float = 2.0,
) -> Tuple[float, float]:
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
    """
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


def plot_dist(  # pylint:disable=too-many-arguments
    inp: Union[emcee.backends.Backend],
    burnin: int,
    thin: int,
    parameter_order: List[str],
    axes_d: Dict[str, matplotlib.axes.Axes],
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
    if not HAS_SEABORN:
        raise ImportError("``plot_dist`` requires seaborn to be installed")

    burnt_in_samples_labelled = get_labelled_chain_data(
        inp,
        parameter_order,
        burnin=burnin,
        thin=thin,
    )

    for label, to_plot in burnt_in_samples_labelled.items():
        ax = axes_d[label]  # pylint:disable=invalid-name

        sns.kdeplot(
            data=to_plot,
            ax=ax,
            common_norm=common_norm,
            fill=fill,
            legend=legend,
            **kwargs,
        )
        ax.set_xlabel(label)


DEFAULT_PLOT_CORNER_TITLE_KWARGS: Dict[str, Any] = {"fontsize": 12}
"""
Default value for ``title_kwargs`` used by ``plot_corner``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


DEFAULT_PLOT_CORNER_LABEL_KWARGS: Dict[str, Any] = {"fontsize": "x-small"}
"""
Default value for ``label_kwargs`` used by ``plot_corner``

Provided to give the user an easy to modify these defaults if they wish or a
starting point
"""


def plot_corner(  # pylint:disable=too-many-arguments,too-many-locals
    inp: Union[emcee.backends.Backend],
    burnin: int,
    thin: int,
    parameter_order: List[str],
    fig: matplotlib.figure.Figure,
    bins: int = 30,
    plot_contours: bool = True,
    smooth: bool = True,
    quantiles: Sequence[float] = (0.05, 0.17, 0.5, 0.83, 0.95),
    show_titles: bool = True,
    title_quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    title_kwargs: Optional[Dict[str, Any]] = None,
    title_fmt: str = ".3f",
    label_kwargs: Optional[Dict[str, Any]] = None,
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
    if not HAS_CORNER:
        raise ImportError("``plot_corner`` requires corner to be installed")

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
