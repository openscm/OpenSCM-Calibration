"""
Store of [scmdata][]-based results
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scmdata.run


def add_iteration_to_res_scmrun(
    res: scmdata.run.BaseScmRun, iteration: int, iteration_metadata_column: str = "it"
) -> scmdata.run.BaseScmRun:
    """
    Add iteration information to a result stored as [`scmdata.run.BaseScmRun`][]

    Parameters
    ----------
    res
        Result of the run

    iteration
        Iteration to assign to the run

    iteration_metadata_column
        Metadata column in which to store the iteration information

    Returns
    -------
    :
        Result with iteration information added.
    """
    out = res.copy()
    out[iteration_metadata_column] = iteration

    return out
