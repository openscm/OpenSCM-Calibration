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


def get_acceptance_fractions(
    chains: np.typing.NDArray[np.float_],
) -> np.typing.NDArray[np.float_]:
    """
    Get acceptance fraction in each chain of an MCMC ensemble of chains

    Parameters
    ----------
    chains
        Chains. We expected that the the axes are ["step", "chain",
        "parameter"]

    Returns
    -------
        Acceptance fraction in each chain
    """
    # This is complicated, in short:
    # 1. find differences between steps across all calibration parameters
    # 2. check, in each chain, if there is any difference in any of the
    #   parameter values (where this happens, it means that the step was
    #   accepted)
    # 3. sum up the number of accepted steps in each chain
    accepted: np.typing.NDArray[np.int_] = np.sum(
        np.any(np.diff(chains, axis=0), axis=2), axis=0
    )
    n_proposals = chains.shape[0] - 1  # first step isn't a proposal
    acceptance_fraction = accepted / np.float_(n_proposals)

    return acceptance_fraction


def check_autocorrelation(
    inp: emcee.backends.Backend,
    burnin: int,
    thin: int = 1,
    autocorr_tol: int = 0,
    convergence_ratio: float = 50,
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

        - tau: autocorrelation in each chains
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

    return out
