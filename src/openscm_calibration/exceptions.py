"""
Exceptions that are used throughout
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


class MissingRequiredDependencyError(ImportError):
    """
    Raised when a dependency is missing

    This is only needed for optional dependencies e.g. plotting dependencies
    like seaborn
    """

    def __init__(self, callable_name: str, requirement: str) -> None:
        """
        Initialise the error

        Parameters
        ----------
        callable_name
            The name of the callable that requires the dependency

        requirement
            The name of the requirement
        """
        error_msg = f"``{callable_name}`` requires {requirement} to be installed"
        super().__init__(error_msg)


class NotExpectedValueError(ValueError):
    """
    Raised when the value is not what we expect

    This is a very verbose version of an assertion error
    """

    def __init__(self, ref_name: str, val: Any, expected_val: Any) -> None:
        """
        Initialise the error

        Parameters
        ----------
        ref_name
            The name of the thing being referenced (variable, attribute etc.)

        val
            The value of ``ref_name``

        expected_val
            The value we expected
        """
        error_msg = f"``{ref_name}`` must have value: {expected_val}, received: {val}"
        super().__init__(error_msg)


class NotExpectedAllSameValueError(ValueError):
    """
    Raised when the values are not all the same expected value
    """

    def __init__(self, ref_name: str, expected_val: Any) -> None:
        """
        Initialise the error

        Parameters
        ----------
        ref_name
            The name of the thing being referenced (variable, attribute etc.)

        expected_val
            The value we expected all elements of ``ref_name`` to have
        """
        error_msg = f"All values in ``{ref_name}`` should be ``{expected_val}``"
        super().__init__(error_msg)


class MismatchLengthError(ValueError):
    """
    Raised when an object should have the same length as something else, but doesn't
    """

    def __init__(
        self, name: str, length: int, expected_name: Any, expected_length: int
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        name
            The name of the thing being referenced (variable, attribute etc.)

        length
            The length of the thing being referenced

        expected_name
            The name of the thing we expect to match

        expected_length
            The expected length of the thing ``name`` references
        """
        error_msg = (
            f"``{name}`` has length {length}, it should have length "
            f"{expected_length}, the same as ``{expected_name}``"
        )
        super().__init__(error_msg)


class MissingValueError(ValueError):
    """
    Raised when a sequence is missing a value(s) that we expect it to have
    """

    def __init__(self, name: str, vals: Sequence[Any], missing_vals: Any) -> None:
        """
        Initialise the error

        Parameters
        ----------
        name
            The name of the thing being referenced (variable, attribute etc.)

        vals
            The values in ``name``

        missing_vals
            The value(s) that are missing
        """
        error_msg = (
            f"``{name}`` is missing values: ``{missing_vals}``. "
            f"Available values: ``{vals}``"
        )
        super().__init__(error_msg)


class AlignmentError(ValueError):
    """
    Raised when two [`pandas.DataFrame`][pandas.DataFrame] do not align smoothly
    """

    def __init__(  # noqa: PLR0913
        self,
        name_left: str,
        val_left: pd.DataFrame | pd.Series[Any],
        name_right: str,
        val_right: pd.DataFrame | pd.Series[Any],
        extra_context: str | None = None,
    ) -> None:
        """
        Initialise the error

        Parameters
        ----------
        name_left
            The name of the first thing being referenced (variable, attribute
            etc.)

        val_left
            The values referred to by ``name_left``

        name_right
            The name of the other thing being referenced (variable, attribute
            etc.)

        name_right
            The values referred to by ``name_left``

        extra_context
            Any extra context to include in the message
        """
        error_msg = (
            f"Please check.\n"
            f"{name_left}:\n{val_left}\n"
            f"{name_right}:\n{val_right}\n"
        )

        if extra_context:
            error_msg = f"{extra_context}. {error_msg}"

        super().__init__(error_msg)
