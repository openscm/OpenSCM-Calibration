"""
Helpers for emcee
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import numpy.typing as nptype
from attrs import define
from typing_extensions import TypeAlias

from openscm_calibration.parameter_handling import ParameterOrder
from openscm_calibration.typing import (
    DataContainer,
    SupportsModelRun,
    SupportsNegativeLogLikelihoodCalculation,
)

if TYPE_CHECKING:
    # See here for explanation of this pattern and why we don't need quotes
    # below https://docs.python.org/3/library/typing.html#constant
    from typing import Any

    import emcee.backends


SupportsLikelihoodCalculation: TypeAlias = Callable[[nptype.NDArray[np.float64]], float]


def get_neg_log_prior(
    parameter_order: ParameterOrder,
    kind: Literal["uniform"] = "uniform",
) -> SupportsLikelihoodCalculation:
    """
    Get a function for calculating the negative log prior

    Parameters
    ----------
    parameter_order
        The parameters being calibrated

    kind
        The kind of log prior to create.

        Options:

        - "uniform": a uniform prior within the bounds for each parameter

    Returns
    -------
    :
        Function that, given a parameter vector, returns the negative log prior
    """
    if kind == "uniform":
        bounds = np.array(
            [parameter.bounds_m() for parameter in parameter_order.parameters]
        )

        def neg_log_prior(x: nptype.NDArray[np.float64]) -> float:
            """
            Get log prior for a given parameter vector, `x`

            This was generated in `get_neg_log_prior_from_bounds`
            and assumes a uniform distribution for parameters
            within their bounds.

            Future uses must match the bounds order provided to
            `get_neg_log_prior_from_bounds`.

            Parameters
            ----------
            x
                Parameter vector.

            Returns
            -------
            :
                Negative log prior of `x`
            """
            in_bounds = (x > bounds[:, 0]) & (x < bounds[:, 1])
            if np.all(in_bounds):
                return 0

            return -np.inf

    else:
        raise NotImplementedError(kind)

    return neg_log_prior


def neg_log_info(
    x: nptype.NDArray[np.float64],
    neg_log_prior: Callable[[nptype.NDArray[np.float64]], float],
    model_runner: SupportsModelRun[DataContainer],
    negative_log_likelihood_calculator: SupportsNegativeLogLikelihoodCalculation[
        DataContainer
    ],
) -> tuple[float, float, float | None]:
    """
    Get negative log probability and likelihood information

    Specifically, negative log probability, log prior and log likelihood
    for a given parameter vector.

    Parameters
    ----------
    x
        Parameter vector.

        Be careful with the order and units of parameters.
        No checks of parameter order or units are performed in this function.

    neg_log_prior
        Function that calculates the negative log prior for a given value of `x`

    model_runner
        Runner of the model for a given value of `x`

    negative_log_likelihood_calculator
        Calculator of negative log likelihood based on the outputs of the model run

    Returns
    -------
    :
        Negative log probability of `x`,
        negative log prior of `x`
        and negative log likelihood of `x`.

        If the negative log probability of `x` is `-np.inf`,
        then no likelihood calculation is performed
        and `None` is returned for the negative log likelihood of `x`.

        If the log likelihood calculation raises a `ValueError`,
        we assume that the likelihood calculation failed
        and return `-np.inf` for the log probability
        and negative log likelihood of `x`.
    """
    neg_log_prior_x = neg_log_prior(x)

    if not np.isfinite(neg_log_prior_x):
        return -np.inf, neg_log_prior_x, None

    try:
        model_results = model_runner.run_model(x)
    except ValueError:
        return -np.inf, neg_log_prior_x, -np.inf

    neg_log_likelihood_x = (
        negative_log_likelihood_calculator.calculate_negative_log_likelihood(
            model_results
        )
    )
    neg_log_prob = neg_log_prior_x + neg_log_likelihood_x

    return neg_log_prob, neg_log_prior_x, neg_log_likelihood_x


def get_start(
    sampler: emcee.ensemble.EnsembleSampler,
    start: nptype.NDArray[np.float64] | None = None,
) -> nptype.NDArray[np.float64]:
    """
    Get starting point for emcee sampling

    Parameters
    ----------
    sampler
        Sampler which will do the sampling

    start
        Starting point to use.

        This is only used if the sampler has not already performed some iterations.

    Returns
    -------
    :
        Starting point for the sampling

    Raises
    ------
    TypeError
        `start` is `None` and the sampler has not performed any iterations yet.

    Warns
    -----
    UserWarning
        `start` is not `None` but the sampler has already performed iterations.

        In this case, we use the sampler's last iteration
        and ignore the value of `start`.
    """
    if sampler.iteration < 1:
        # Haven't used any samples yet, hence must have start
        if start is None:
            msg = (
                "The sampler has not performed any iterations yet. "
                "You must provide a value for `start`. "
                f"Received {start=}."
            )
            raise TypeError(msg)

        res = start

    else:
        # Sampler has already done iterations, hence ignore start
        if start is not None:
            # Avoidable user side-effect, hence warn
            # (see https://docs.python.org/3/howto/logging.html#when-to-use-logging)
            warn_msg = (
                "The sampler has already performed iterations. "
                "We will use its last sample as the starting point "
                "rather than the provided value for `start`."
                "(If you with to re-start the sampling, reset the sampler)."
            )
            warnings.warn(warn_msg)

        res = sampler.get_last_sample()

    return res


@define
class ChainProgressInfo:
    """Information about the progress of MCMC chains"""

    steps: int
    """Number of steps in the chain"""

    steps_post_burnin: int
    """Number of steps after the chain's burn-in period"""

    acceptance_fraction: float
    """Acceptance fraction of proposed steps"""


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


@define
class AutoCorrelationInfo:
    """Information about auto-correlation within an MCMC chain"""

    steps_post_burnin: int
    """The number of steps in the chains post burn-in."""

    tau: nptype.NDArray[np.float64]
    """
    Auto-correlation for each parameter

    I.e. the first value is the auto-correlation for the first parameter,
    second value is the auto-correlation for the second parameter,
    ...
    nth value is the auto-correlation for the nth parameter,
    ...
    """

    convergence_ratio: float
    """
    Convergence ratio used to assess convergence.

    For details, see
    [`get_autocorrelation_info`][openscm_calibration.emcee_utils.get_autocorrelation_info].
    """

    converged: tuple[bool, ...]
    """
    Whether, based on `convergence_ratio`, the chains for each parameter have converged

    The first value is whether the chain for the first parameter has converged,
    the second value is whether the chain for the second parameter has converged,
    ...
    nth value is whether the chain for the nth parameter has converged,
    ...
    """

    def any_non_nan_tau(self) -> bool:
        """
        Check if any of the tau information is not nan

        Returns
        -------
        :
            Whether any of the tau values are non-nan
        """
        return np.any(np.logical_not(np.isnan(self.tau)))


def get_autocorrelation_info(
    inp: emcee.backends.Backend,
    burnin: int,
    thin: int = 1,
    autocorr_tol: int = 0,
    convergence_ratio: float = 50,
) -> AutoCorrelationInfo:
    """
    Get info about autocorrelation in chains

    Parameters
    ----------
    inp
        Object of which to check autocorrelation

    burnin
        Number of iterations to treat as burn-in

    thin
        Thinning to apply to the chains.

        Emcee handles this such that the returned output is in steps,
        not thinned steps.

    autocorr_tol
        Tolerance for auto-correlation calculations.

        Set to zero to force calculation to never fail

    convergence_ratio
        Convergence ratio to apply when checking for convergence

        If the number of iterations (excluding burn-in)
        is greater than `convergence_ratio`
        multiplied by the autocorrelation (averaged over all the chains),
        we assume the chain has converged.

    Returns
    -------
    :
        Results of calculation
    """
    tau = inp.get_autocorr_time(discard=burnin, tol=autocorr_tol, thin=thin)

    converged = inp.iteration - burnin > convergence_ratio * tau

    out = AutoCorrelationInfo(
        steps_post_burnin=inp.iteration - burnin,
        tau=tau,
        convergence_ratio=convergence_ratio,
        converged=converged,
    )

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
        Order of model parameters. This must match the order used by  `inp`.

    neg_log_likelihood_name
        Name to use for the negative log likelihood data.

        If not provided, negative log likelihood information is not returned.

    burnin
        Number of iterations to treat as burn in

    thin
        Thinning to use when sampling the chains

    Returns
    -------
    :
        Chain data, labelled with parameter names
        and, if requested, `neg_log_likelihood_name`
    """
    all_samples = inp.get_chain(discard=burnin, thin=thin)

    out = {para: all_samples[:, :, i] for i, para in enumerate(parameter_order)}

    if neg_log_likelihood_name:
        all_neg_ll = inp.get_log_prob(discard=burnin, thin=thin)
        out[neg_log_likelihood_name] = all_neg_ll

    return out
