"""
Storage class
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, MutableSequence, Protocol, Tuple, Union

import numpy as np
from attrs import define, field

if TYPE_CHECKING:
    import attr
    import numpy.typing as nptype
    import scmdata.run


class SupportsListLikeHandling(Protocol):  # pylint: disable=too-few-public-methods
    """
    Class that supports handling a list-like
    """

    def list(self, list_to_handle: MutableSequence[Any]) -> MutableSequence[Any]:
        """
        Get a new object that behaves like a :obj:`MutableSequence`
        """


def _all_none_to_start(
    instance: OptResStore,  # pylint: disable=unused-argument
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
    if not all(v is None for v in value):
        raise ValueError(
            f"All values in ``{attribute.name}`` should be ``None`` to start"
        )


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
    if len(value) != len(instance.res):
        raise ValueError(f"``{attribute.name}`` must be the same length as ``res``")


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
        raise ValueError(
            f"{attribute.name} must have indices: {exp_indices}, received: {value}"
        )


@define
class OptResStore:
    """
    Store for results during optimisation
    """

    res: MutableSequence[Union[None, scmdata.run.BaseScmRun]] = field(
        validator=[_all_none_to_start]
    )
    """Results of runs"""

    costs: MutableSequence[Union[None, float]] = field(
        validator=[_all_none_to_start, _same_length_as_res]
    )
    """Costs of runs"""

    x_samples: MutableSequence[
        Union[None, nptype.NDArray[Union[np.float_, np.int_]]]
    ] = field(validator=[_all_none_to_start, _same_length_as_res])
    """x vectors sampled"""

    params: Tuple[str]
    """Names of the parameters being stored in ``x_samples``"""

    available_indices: MutableSequence[int] = field(
        validator=[_same_length_as_res, _contains_indices_in_res]
    )
    """Indices available to be written into"""

    @classmethod
    def from_n_runs(
        cls,
        n_runs: int,
        params: Tuple[str],
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
        params: Tuple[str],
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
        return self.available_indices.pop()  # pylint:disable=no-member

    def set_result_cost_x(
        self,
        res: Union[None, scmdata.run.BaseScmRun],
        cost: float,
        x: nptype.NDArray[Union[np.float_, np.int_]],  # pylint: disable=invalid-name
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
        if len(x) != len(self.params):
            raise ValueError(
                f"Length of parameter vector ({len(x)}) does not match "
                f"length of ``self.params`` ({len(self.params)})"
            )

        self.costs[idx] = cost  # pylint: disable=unsupported-assignment-operation
        self.x_samples[idx] = x  # pylint: disable=unsupported-assignment-operation
        self.res[idx] = res  # pylint: disable=unsupported-assignment-operation

    def append_result_cost_x(
        self,
        res: scmdata.run.BaseScmRun,
        cost: float,
        x: nptype.NDArray[Union[np.float_, np.int_]],  # pylint: disable=invalid-name
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
        x: nptype.NDArray[Union[np.float_, np.int_]],  # pylint: disable=invalid-name
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
    ) -> Tuple[
        Tuple[float, ...],
        Tuple[nptype.NDArray[Union[np.float_, np.int_]], ...],
        Tuple[scmdata.run.BaseScmRun, ...],
    ]:
        """
        Get costs, x_samples and res from runs

        Returns
        -------
            Costs, x_samples and res from all runs which were attempted (i.e. we
            include failed runs here)
        """
        # There may be a better algorithm for this, PRs welcome :)
        if all(x is None for x in self.x_samples):  # pylint: disable=not-an-iterable
            return ((), (), ())

        tmp = tuple(
            zip(
                *[
                    (
                        self.costs[i],  # pylint: disable=unsubscriptable-object
                        x,
                        self.res[i],  # pylint: disable=unsubscriptable-object
                    )
                    for i, x in enumerate(self.x_samples)
                    # x is only None if no run was attempted yet
                    if x is not None
                ]
            )
        )

        # Help out type hinting
        costs: Tuple[float, ...] = tmp[0]
        xs_out: Tuple[nptype.NDArray[Union[np.float_, np.int_]], ...] = tmp[1]
        ress: Tuple[scmdata.run.BaseScmRun, ...] = tmp[2]

        out = (costs, xs_out, ress)

        return out

    def get_costs_labelled_xsamples_res(
        self,
    ) -> Tuple[
        Tuple[float, ...],
        Dict[str, nptype.NDArray[Union[np.float_, np.int_]]],
        Tuple[scmdata.run.BaseScmRun, ...],
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
