"""
Model runner class
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any, Callable, Generic, Protocol

import numpy as np
import pint
from attrs import define

from openscm_calibration.typing import DataContainer_co


class XToNamedPintConvertor(Protocol):
    """
    Callable that supports converting the x-vector to Pint quantities
    """

    def __call__(
        self,
        x: np.typing.NDArray[np.number[Any]],
    ) -> dict[str, pint.registry.UnitRegistry.Quantity]:
        """
        Convert x to pint quantities
        """


class ModelRunsInputGenerator(Protocol):
    """
    Callable that supports generating model run inputs
    """

    def __call__(self, **kwargs: pint.registry.UnitRegistry.Quantity) -> dict[str, Any]:
        """
        Generate model run inputs
        """


class ModelRunner(Protocol[DataContainer_co]):
    """
    Callable that supports running the model
    """

    def __call__(self, **kwargs: Any) -> DataContainer_co:
        """
        Run the model
        """


@define
class OptModelRunner(Generic[DataContainer_co]):
    """
    Model runner used during optimisation
    """

    convert_x_to_names_with_units: XToNamedPintConvertor
    """
    Callable to translate the x-vector into input for `self.do_model_runs_input_generator`

    This translates from the x-vector used internally, by e.g. scipy and emcee,
    into a dictionary with meaningful keys and quantities with units (as needed).
    It must produce named output that can be passed directly to
    `self.do_model_runs_input_generator`.
    """  # noqa: E501

    do_model_runs_input_generator: ModelRunsInputGenerator
    """
    Generator of inputs for ``do_model_runs``

    More specifically, the callable used to translate the parameters
    (already converted to [`pint.Quantity`][])
    into the keyword arguments required by `self.do_model_runs`.
    """

    do_model_runs: ModelRunner[DataContainer_co]
    """
    Function that runs the model

    Runs the desired experiments based on inputs generated by
    `self.do_model_runs_input_generator`.
    """

    @classmethod
    def from_parameters(
        cls,
        params: Iterable[tuple[str, str | pint.Unit | None]],
        do_model_runs_input_generator: ModelRunsInputGenerator,
        do_model_runs: ModelRunner[DataContainer_co],
        get_unit_registry: Callable[[], pint.UnitRegistry] | None = None,
    ) -> OptModelRunner[DataContainer_co]:
        """
        Initialise from list of parameters

        This is a convenience method

        Parameters
        ----------
        params
            List of parameters

        do_model_runs_input_generator
            Generator of input for `do_model_runs`.
            See docstring of `self` for more details.

        do_model_runs
            Callable which does the model runs.
            See docstring of  `self` for more details.

        get_unit_registry
            Function to get unit registry.

            Passed to
            [`x_and_parameters_to_named_with_units`][openscm_calibration.model_runner.x_and_parameters_to_named_with_units].
            See docstring of that function for further details.

        Returns
        -------
        :
            Initialised instance
        """
        convert_x_to_names_with_units = partial(
            x_and_parameters_to_named_with_units,
            params=params,
            get_unit_registry=get_unit_registry,
        )

        return OptModelRunner(
            convert_x_to_names_with_units=convert_x_to_names_with_units,
            do_model_runs_input_generator=do_model_runs_input_generator,
            do_model_runs=do_model_runs,
        )

    def run_model(
        self,
        x: np.typing.NDArray[np.number[Any]],
    ) -> Any:
        """
        Run the model

        Parameters
        ----------
        x
            Vector of calibration parameter values (the x-vector)

        Returns
        -------
        :
            Results of run
        """
        x_converted_name = self.convert_x_to_names_with_units(x)

        do_model_runs_inputs = self.do_model_runs_input_generator(**x_converted_name)

        res = self.do_model_runs(**do_model_runs_inputs)

        return res


def x_and_parameters_to_named_with_units(
    x: np.typing.NDArray[np.number[Any]],
    params: Iterable[tuple[str, str | pint.Unit | None]],
    get_unit_registry: Callable[[], pint.UnitRegistry] | None = None,
) -> dict[str, pint.registry.UnitRegistry.Quantity]:
    """
    Convert the x-vector to a dictionary and add units

    Parameters
    ----------
    x
        Vector of calibration parameter values

    params
        parameters to be calibrated

        This defines both the names and units of the parameters.
        If the units are `None`, the values are assumed to be plain numpy quantities.

    get_unit_registry
        Function to get unit registry.
        This allows the user to do a delayed import of the unit registry,
        which is important because pint's unit registries don't parallelise well.

        If not provided, [`pint.get_application_registry`][] is used.

    Returns
    -------
    :
        Parameters, named and converted to [`pint.Quantity`][]
        where appropriate

    Examples
    --------
    It also possible to inject a different registry as needed
    >>> import pint
    >>> ur_plus_pop = pint.UnitRegistry()
    >>> ur_plus_pop.define("thousands = [population]")
    >>> def get_ur_with_pop():
    ...     return ur_plus_pop
    >>> # Withoout the injection, an error is raised
    >>> x_and_parameters_to_named_with_units(
    ...     [1.1, 3.2],
    ...     [("para_a", "m"), ("pop_weight", "thousands")],
    ... )
    Traceback (most recent call last):
    ...
    pint.errors.UndefinedUnitError: 'thousands' is not defined in the unit registry
    >>> # With the injection, this works nicely
    >>> x_and_parameters_to_named_with_units(
    ...     [1.1, 3.2, 4.0],
    ...     [("para_a", "m"), ("pop_weight", "thousands"), ("factor", None)],
    ...     get_ur_with_pop,
    ... )
    {'para_a': <Quantity(1.1, 'meter')>, 'pop_weight': <Quantity(3.2, 'thousands')>, 'factor': 4.0}
    """  # noqa: E501
    unit_reg = (
        get_unit_registry() if get_unit_registry else pint.get_application_registry()  # type: ignore
    )

    out: dict[str, pint.registry.UnitRegistry.Quantity] = {}
    for val, (key, unit) in zip(x, params):
        if unit is not None:
            val_out = unit_reg.Quantity(val, unit)
        else:
            val_out = val

        out[key] = val_out

    return out
