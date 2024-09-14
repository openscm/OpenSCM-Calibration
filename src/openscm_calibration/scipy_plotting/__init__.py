"""
Support for plotting during scipy optimisation
"""

from __future__ import annotations

from openscm_calibration.scipy_plotting.base import (
    CallbackProxy,
    OptPlotter,
    get_optimisation_mosaic,
    get_runs_to_plot,
    get_ymax_default,
    plot_costs,
    plot_parameters,
)

__all__ = [
    "CallbackProxy",
    "OptPlotter",
    "get_optimisation_mosaic",
    "get_runs_to_plot",
    "get_ymax_default",
    "plot_costs",
    "plot_parameters",
]
