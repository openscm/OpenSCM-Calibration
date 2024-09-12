"""
Helpers for emcee
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # See here for explanation of this pattern and why we don't need quotes
    # below https://docs.python.org/3/library/typing.html#constant
    from typing import Any

    import emcee.backends
    import numpy.typing as nptype


def get_acceptance_fractions(
    chains: nptype.NDArray[np.float64],
) -> nptype.NDArray[np.float64]:
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
    accepted: nptype.NDArray[np.int_] = np.sum(
        np.any(np.diff(chains, axis=0), axis=2), axis=0
    )
    n_proposals = chains.shape[0] - 1  # first step isn't a proposal
    acceptance_fractions = accepted / np.float64(n_proposals)

    return acceptance_fractions


def get_autocorrelation_info(
    inp: emcee.backends.Backend,
    burnin: int,
    thin: int = 1,
    autocorr_tol: int = 0,
    convergence_ratio: float = 50,
) -> dict[str, float | int | bool | nptype.NDArray[np.floating[Any]]]:
    """
    Get info about autocorrelation in chains

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
        Results of calculation, keys:

        - "tau": autocorrelation in each chains
        - "autocorr": average of tau
        - "converged": whether the chains have converged or not based on
          ``convergence_ratio``
        - "convergence_ratio": value of ``convergence_ratio``
        - "steps_post_burnin": Number of steps in chains post burn-in

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


def get_labelled_chain_data(
    inp: emcee.backends.Backend,
    parameter_order: list[str],
    neg_log_likelihood_name: str | None = None,
    burnin: int = 0,
    thin: int = 0,
) -> dict[str, np.typing.NDArray[np.floating[Any] | np.integer[Any]]]:
    """
    Get labelled chain data

    Parameters
    ----------
    inp
        Object from which to plot the state

    parameter_order
        Order of model parameters. This must match the order used by  ``inp``.

    neg_log_likelihood_name
        Name to use for the negative log likelihood data. If not provided,
        negative log likelihood information is not returned.

    burnin
        Number of iterations to treat as burn in

    thin
        Thinning to use when sampling the chains

    Returns
    -------
        Chain data, labelled with parameter names and, if requested,
        ``neg_log_likelihood_name``
    """
    all_samples = inp.get_chain(discard=burnin, thin=thin)

    out = {para: all_samples[:, :, i] for i, para in enumerate(parameter_order)}

    if neg_log_likelihood_name:
        all_neg_ll = inp.get_log_prob(discard=burnin, thin=thin)
        out[neg_log_likelihood_name] = all_neg_ll

    return out
