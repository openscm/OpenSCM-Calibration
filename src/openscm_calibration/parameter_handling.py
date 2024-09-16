"""
Handling of parameters

In particular, helping to ensure that we use the correct parameter order and units
in the many places where the order and units are not checked by the receiving function
(e.g. all of scipy's optimisation routines, which work on bare arrays).
"""

from __future__ import annotations

from typing import TypeVar

import attr
import numpy as np
import numpy.typing as nptype
import pint
from attrs import define, field

BoundsValue = TypeVar(
    "BoundsValue", float, np.float64, pint.registry.UnitRegistry.Quantity
)


@define
class BoundDefinition:
    """Definition of bounds for a parameter"""

    lower: BoundsValue = field()
    """Lower bound"""

    upper: BoundsValue = field()
    """Upper bound"""

    @upper.validator
    def check_greather_than_lower(
        instance: BoundDefinition,
        attribute: attr.Attribute[pint.registry.UnitRegistry.Quantity],
        value: pint.registry.UnitRegistry.Quantity,
    ) -> None:
        """Check that the value is greater than the lower bound"""
        if value <= instance.lower:
            msg = (
                "`upper` must be greater than `lower`. "
                f"Received {instance.lower=}, instance.upper={value=}`."
            )
            raise ValueError(msg)


@define
class ParameterDefinition:
    """Definition of a parameter"""

    name: str
    """The parameter's name"""

    unit: str | None
    """
    The parameter's default unit for usage that isn't unit-aware

    If this is `None`, we assume that no units apply to the parameter.
    """

    bounds: BoundDefinition | None = None
    """The bounds for the parameter"""

    def bounds_m(
        self, unit: str | None = None
    ) -> tuple[
        pint.registry.UnitRegistry.Quantity, pint.registry.UnitRegistry.Quantity
    ]:
        """
        Get the magnitude of `self.bounds`

        Parameters
        ----------
        unit
            Unit in which to return the magnitude.

            Only applies if `self.unit` is not `None`.

            If `self.unit` is not `None` and `unit` is not supplied,
            we use `self.unit` instead
            (i.e. `self.unit` functions as a default).

        Returns
        -------
        :
            Magnitude of the bounds (lower, upper).
        """
        if self.bounds is None:
            msg = (
                f"No bounds provided for {self.name}. "
                "Please re-initialise this parameter definition."
            )
            raise ValueError(msg)

        if self.unit is None:
            return tuple([v for v in [self.bounds.lower, self.bounds.upper]])

        unit_h = unit if unit is not None else self.unit

        return tuple([v.to(unit_h).m for v in [self.bounds.lower, self.bounds.upper]])


@define
class ParameterOrder:
    """Parameter order definition"""

    parameters: tuple[ParameterDefinition, ...]
    """
    Parameters

    These are stored in a tuple, hence order is preserved.
    """

    def bounds_m(
        self, units: dict[str, str] | None = None
    ) -> nptype.NDArray[np.float64]:
        """
        Get the magnitude of the bounds of the parameters

        Parameters
        ----------
        units
            Units to use for the parameters.

            Keys are parameter names,
            values are the units to use for the parameter's bounds.

            If not supplied at all,
            we simply use the default unit for each parameter
            i.e. the default behaviour of
            [`ParameterDefinition.bounds_m`][openscm_calibration.parameter_handling.ParameterDefinition.bounds_m].

            If not supplied for a given parameter,
            we simply use the parameter's default unit
            i.e. the default behaviour of
            [`ParameterDefinition.bounds_m`][openscm_calibration.parameter_handling.ParameterDefinition.bounds_m].

        Returns
        -------
        :
            Magnitude of the bounds of the parameters (in order)
        """
        out_l = []
        for parameter in self.parameters:
            if units is not None and parameter.name in units:
                unit = units[parameter.name]

            else:
                unit = None

            out_l.append(parameter.bounds_m(unit))

        return np.array(out_l)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Get the names of the parameters

        Returns
        -------
        :
            Names of the parameters (in order)
        """
        return tuple([v.name for v in self.parameters])
