"""
Matplotlib utility functions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openscm_calibration.exceptions import MissingRequiredDependencyError

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

try:
    import IPython.display

    HAS_IPYTHON = True
except ImportError:  # pragma: no cover
    HAS_IPYTHON = False


if TYPE_CHECKING:
    from typing import Any

    import IPython
    import matplotlib


def get_fig_axes_holder_from_mosaic(
    mosaic: list[list[str]],
    **kwargs: Any,
) -> tuple[
    matplotlib.figure.Figure,
    dict[str, matplotlib.axes.Axes],
    IPython.core.display_functions.DisplayHandle,
]:
    """
    Get figure, axes and holder from mosaic

    This is a convenience function

    Parameters
    ----------
    mosaic
        Mosaic to use

    **kwargs
        Passed to :func:`matplotlib.pyplot.subplot_mosaic`

    Returns
    -------
        Created figure, axes and holder

    Raises
    ------
    ImportError
        ``matplotlib`` is not installed or ``IPython`` is not installed
    """
    if not HAS_MATPLOTLIB:
        raise MissingRequiredDependencyError(
            "get_fig_axes_holder_from_mosaic", requirement="matplotlib"
        )

    if not HAS_IPYTHON:
        raise MissingRequiredDependencyError(
            "get_fig_axes_holder_from_mosaic", requirement="IPython"
        )

    fig, axes = plt.subplot_mosaic(
        mosaic=mosaic,  # type: ignore # matplotlib's type hints are unclear
        **kwargs,
    )
    holder = IPython.display.display(fig, display_id=True)  # type: ignore

    return fig, axes, holder
