"""
Helpers for emcee
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Union

import numpy as np

if TYPE_CHECKING:
    # See here for explanation of this pattern and why we don't need quotes
    # below https://docs.python.org/3/library/typing.html#constant
    import emcee.backends


def check_autocorrelation(
    inp: Union[emcee.backends.Backend],
    burnin: int,
    thin: int = 1,
    autocorr_tol: int = 0,
    convergence_ratio: int = 50,
) -> Dict[str, Union[float, int, np.typing.NDArray[np.float_], bool]]:
    """
    Check autocorrelation in chains

    Parameters
    ----------
    inp
        Object of which to check autocorrelation

    burnin
        Number of iterations to treat as burn-in

    thin
        Thinning to apply to the chains. Emcee handles
        this so that the returned output is in steps,
        not thinned steps.

    autocorr_tol
        Tolerance for auto-correlation calculations. Set
        to zero to force calculation to never fail

    convergence_ratio
        If the number of iterations (excluding burn-in) is
        greater than ``convergence_ratio`` multiplied by
        the autocorrelation (averaged over all the chains),
        we assume the chains have converged

    Returns
    -------
        Results of calculation, keys (TODO check dot points in html):

        - tau: autocorrelation in each chain
        - autocorr: average of tau
        - converged: whether the chains have converged or not based on
          ``convergence_ratio``
        - convergence_ratio: value of ``convergence_ratio``
        - steps_post_burnin: Number of steps in chains post burn-in

    """
    tau = inp.get_autocorr_time(discard=burnin, tol=autocorr_tol, thin=thin)
    autocorr = np.mean(tau)

    converged = inp.iteration - burnin > convergence_ratio * autocorr

    out = {
        "tau": tau,
        "autocorr": autocorr,
        "converged": converged,
        "convergence_ratio": convergence_ratio,
        "steps_post_burnin": inp.iteration - burnin,
    }

    # TODO: move into notebook
    # if verbose:
    #     print(f"tau = {np.round(out['tau'], 2)}")
    #     print(f"autocorr = {out['autocorr']:.2f}")
    #     print(f"inp.iteration - burnin = {inp.iteration - burnin}")
    #     print(
    #         f"convergence_ratio * autocorr = {convergence_ratio * out['autocorr']:.1f}"
    #     )

    return out
