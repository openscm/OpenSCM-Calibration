"""
scmdata utility functions
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

if TYPE_CHECKING:
    import scmdata.run


def scmrun_as_dict(
    inp: scmdata.run.BaseScmRun, groups: Iterable[str], separator: str = "_"
) -> Dict[str, scmdata.run.BaseScmRun]:
    """
    Group an :obj:`scmdata.run.BaseScmRun` into a dictionary with keys

    Parameters
    ----------
    inp
        :obj:`scmdata.run.BaseScmRun` to group

    groups
        Metadata keys to use to make the groups

    separator
        Separator for metadata values when making the keys

    Returns
    -------
        Grouped ``inp``
    """
    res = {}
    for inp_g in inp.groupby(groups):
        key = separator.join([inp_g.get_unique_meta(g, True) for g in groups])
        res[key] = inp_g

    return res
