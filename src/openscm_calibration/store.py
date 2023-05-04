"""
Storage class
"""
from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from attrs import define, field

from openscm_calibration.exceptions import (
    MismatchLengthError,
    NotExpectedAllSameValueError,
    NotExpectedValueError,
)

if TYPE_CHECKING:
    import attr
    import scmdata.run

    from openscm_calibration.type_hints import NPArrayFloatOrInt


class SupportsListLikeHandling(Protocol):
    """
    Class that supports handling a list-like
    """

    def list(self, list_to_handle: MutableSequence[Any]) -> MutableSequence[Any]:
        """
        Get a new object that behaves like a :obj:`MutableSequence`
        """


def _all_none_to_start(
    instance: OptResStore,
    attribute: attr.Attribute[MutableSequence[Any]],
    value: MutableSequence[Any],
) -> None:
    """
    Check all values are ``None``

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
        Not all elements in ``value`` are ``None``
    """
    expected_val = None
    if not all(v is expected_val for v in value):
        raise NotExpectedAllSameValueError(attribute.name, expected_val=expected_val)


def _same_length_as_res(
    instance: OptResStore,
    attribute: attr.Attribute[MutableSequence[Any]],
    value: MutableSequence[Any],
) -> None:
    """
    Check ``value`` has same length as ``instance.res``

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
        ``value`` does not have the same length as ``instance.res``
    """
    length_val = len(value)
    length_res = len(instance.res)
    if length_val != length_res:
        raise MismatchLengthError(
            attribute.name,
            length=length_val,
            expected_name="res",
            expected_length=length_res,
        )


def _contains_indices_in_res(
    instance: OptResStore,
    attribute: attr.Attribute[MutableSequence[int]],
    value: MutableSequence[int],
) -> None:
    """
    Check ``value`` has indices that line up with ``instance.res``

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
        ``value`` does not have indices that line up with ``instance.res``
    """
    exp_indices = list(range(len(instance.res)))
    if list(sorted(value)) != exp_indices:
        raise NotExpectedValueError(
            attribute.name,
            val=value,
            expected_val=exp_indices,
        )


@define
class OptResStore:
    """
    Store for results during optimisation
    """

    res: MutableSequence[None | scmdata.run.BaseScmRun] = field(
        validator=[_all_none_to_start]
    )
    """Results of runs"""

    costs: MutableSequence[None | float] = field(
        validator=[_all_none_to_start, _same_length_as_res]
    )
    """Costs of runs"""

    x_samples: MutableSequence[None | NPArrayFloatOrInt] = field(
        validator=[_all_none_to_start, _same_length_as_res]
    )
    """x vectors sampled"""

    params: tuple[str]
    """Names of the parameters being stored in ``x_samples``"""

    available_indices: MutableSequence[int] = field(
        validator=[_same_length_as_res, _contains_indices_in_res]
    )
    """Indices available to be written into"""

    @classmethod
    def from_n_runs(
        cls,
        n_runs: int,
        params: tuple[str],
    ) -> OptResStore:
        """
        Initialise based on expected number of runs

        Parameters
        ----------
        n_runs
            Expected number of runs

        params
            Names of the parameters that are being sampled

        Returns
        -------
            Initialised store
        """
        # Reverse so that using pop counts up
        available_indices = list(range(n_runs))[::-1]
        return cls(
            res=[None] * n_runs,
            costs=[None] * n_runs,
            x_samples=[None] * n_runs,
            params=params,
            available_indices=available_indices,
        )

    @classmethod
    def from_n_runs_manager(
        cls,
        n_runs: int,
        manager: SupportsListLikeHandling,
        params: tuple[str],
    ) -> OptResStore:
        """
        Initialise based on expected number of runs for use in parallel work

        Parameters
        ----------
        n_runs
            Expected number of runs

        manager
            Manager of lists (e.g. :class:`multiprocess.managers.SyncManager`)

        params
            Names of the parameters that are being sampled

        Returns
        -------
            Initialised store
        """
        # Reverse so that using pop counts up
        available_indices = list(range(n_runs))[::-1]

        return cls(
            res=manager.list([None] * n_runs),
            costs=manager.list([None] * n_runs),
            x_samples=manager.list([None] * n_runs),
            params=params,
            available_indices=manager.list(available_indices),
        )

    def get_available_index(self) -> int:
        """
        Get an available index to write into

        Returns
        -------
            Available index. This index is now no longer considered available.
        """
        return self.available_indices.pop()

    def set_result_cost_x(
        self,
        res: None | scmdata.run.BaseScmRun,
        cost: float,
        x: NPArrayFloatOrInt,
        idx: int,
    ) -> None:
        """
        Set result, cost and x at a given index

        Parameters
        ----------
        res
            Result to append (use ``None`` for a failed run)

        cost
            Cost associated with the run

        x
            Parameter array associated with the run

        idx
            Index in ``self.costs``, ``self.x_samples`` and ``self.res`` to write into
        """
        len_x = len(x)
        len_params = len(self.params)
        if len_x != len_params:
            raise MismatchLengthError(
                "x",
                length=len_x,
                expected_name="self.params",
                expected_length=len_params,
            )

        self.costs[idx] = cost
        self.x_samples[idx] = x
        self.res[idx] = res

    def append_result_cost_x(
        self,
        res: scmdata.run.BaseScmRun,
        cost: float,
        x: NPArrayFloatOrInt,
    ) -> None:
        """
        Append result, cost and x from a successful run to the results

        Parameters
        ----------
        res
            Result to append (use ``None`` for a failed run)

        cost
            Cost associated with the run

        x
            Parameter array associated with the run
        """
        iteration = self.get_available_index()
        res_keep = res.copy()
        res_keep["it"] = iteration

        self.set_result_cost_x(
            res=res_keep,
            cost=cost,
            x=x,
            idx=iteration,
        )

    def note_failed_run(
        self,
        cost: float,
        x: NPArrayFloatOrInt,
    ) -> None:
        """
        Note that a run failed

        Typically, ``cost`` will be ``np.inf``.

        The cost and x parameters are appended to the results, as well as an
        indicator that the run was a failure in ``self.res``.

        Parameters
        ----------
        cost
            Cost associated with the run

        x
            Parameter array associated with the run
        """
        iteration = self.get_available_index()
        self.set_result_cost_x(
            res=None,
            cost=cost,
            x=x,
            idx=iteration,
        )

    def get_costs_xsamples_res(
        self,
    ) -> tuple[
        tuple[float, ...],
        tuple[NPArrayFloatOrInt, ...],
        tuple[scmdata.run.BaseScmRun, ...],
    ]:
        """
        Get costs, x_samples and res from runs

        Returns
        -------
            Costs, x_samples and res from all runs which were attempted (i.e. we
            include failed runs here)
        """
        # There may be a better algorithm for this, PRs welcome :)
        if all(x is None for x in self.x_samples):
            return ((), (), ())

        tmp = tuple(
            zip(
                *[
                    (
                        self.costs[i],
                        x,
                        self.res[i],
                    )
                    for i, x in enumerate(self.x_samples)
                    # x is only None if no run was attempted yet
                    if x is not None
                ]
            )
        )

        # Help out type hinting
        costs: tuple[float, ...] = tmp[0]
        xs_out: tuple[NPArrayFloatOrInt, ...] = tmp[1]
        ress: tuple[scmdata.run.BaseScmRun, ...] = tmp[2]

        out = (costs, xs_out, ress)

        return out

    def get_costs_labelled_xsamples_res(
        self,
    ) -> tuple[
        tuple[float, ...],
        dict[str, NPArrayFloatOrInt],
        tuple[scmdata.run.BaseScmRun, ...],
    ]:
        """
        Get costs, x_samples and res from runs

        Returns
        -------
            Costs, x_samples and res from all runs which were attempted (i.e. we
            include failed runs here)
        """
        unlabelled = self.get_costs_xsamples_res()
        if not any(unlabelled):
            return (
                unlabelled[0],
                {p: np.array([]) for p in self.params},
                unlabelled[2],
            )

        x_samples_stacked = np.vstack(unlabelled[1])
        xs_labelled = {p: x_samples_stacked[:, i] for i, p in enumerate(self.params)}

        out = (unlabelled[0], xs_labelled, unlabelled[2])

        return out
