"""
Matplotlib utility functions
"""
from __future__ import annotations

from typing import TYPE_CHECKING

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
    from typing import Any, Dict, List, Tuple

    import IPython
    import matplotlib


def get_fig_axes_holder_from_mosaic(
    mosaic: List[List[str]],
    **kwargs: Any,
) -> Tuple[
    matplotlib.figure.Figure,
    Dict[str, matplotlib.axes.Axes],
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
        raise ImportError(
            "``get_fig_axes_holder_from_mosaic`` requires matplotlib to be installed"
        )

    if not HAS_IPYTHON:
        raise ImportError(
            "``get_fig_axes_holder_from_mosaic`` requires IPython to be installed"
        )

    fig, axes = plt.subplot_mosaic(
        mosaic=mosaic,
        **kwargs,
    )
    holder = IPython.display.display(fig, display_id=True)  # type: ignore

    return fig, axes, holder
