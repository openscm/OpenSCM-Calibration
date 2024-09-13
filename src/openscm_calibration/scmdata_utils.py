"""
scmdata utility functions
"""

from __future__ import annotations

from collections.abc import Iterable

import scmdata.run


def scmrun_as_dict(
    inp: scmdata.run.BaseScmRun, groups: Iterable[str], separator: str = "_"
) -> dict[str, scmdata.run.BaseScmRun]:
    """
    Group an input into a dictionary with keys

    Parameters
    ----------
    inp
        Data to group

    groups
        Metadata keys to use to make the groups

    separator
        Separator for metadata values when making the keys

    Returns
    -------
    :
        Grouped `inp`
    """
    res = {}
    for inp_g in inp.groupby(groups):
        key = str(separator.join([str(inp_g.get_unique_meta(g, True)) for g in groups]))
        res[key] = inp_g

    return res
