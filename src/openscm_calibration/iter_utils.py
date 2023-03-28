"""
Iterable utility functions
"""
from __future__ import annotations

import itertools
from typing import Any, Iterable, List


def repeat_elements(inp: Iterable[Any], n_rep: int) -> List[Any]:
    """
    Repeat elements of an iterable ``n`` times

    Parameters
    ----------
    inp
        Iterable of which to repeat the elements

    n_rep
        Number of times to repeat

    Returns
    -------
        List with repeated elements

    Examples
    --------
    >>> repeat_elements(["a", "b"], 2)
    ['a', 'a', 'b', 'b']
    """
    return list(
        itertools.chain(*[list(itertools.repeat(element, n_rep)) for element in inp])
    )
