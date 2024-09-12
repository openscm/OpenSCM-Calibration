"""
Tests of :mod:`openscm_calibration.emcee_utils`
"""

from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from openscm_calibration.emcee_utils import (
    get_acceptance_fractions,
    get_autocorrelation_info,
    get_labelled_chain_data,
)


@pytest.mark.parametrize(
    "inp, exp",
    (
        pytest.param(
            [[[0]], [[0]], [[0.1]], [[0.2]], [[0.2]]], [2 / 4], id="1D-simple"
        ),
        pytest.param(
            [[[0, 0]], [[0, 0]], [[0.1, 0]], [[0.1, 0.1]], [[0.2, 0]]],
            [3 / 4],
            id="2-parameters-1-chain",
        ),
        pytest.param(
            [
                [
                    [0, 0],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 1.01],
                ],
                [
                    [0.1, 0],
                    [0, 1.01],
                ],
                [
                    [0.1, 0.1],
                    [0, 1.01],
                ],
                [
                    [0.2, 0],
                    [0.01, 1.03],
                ],
            ],
            [3 / 4, 2 / 4],
            id="2-parameters-2-chains",
        ),
    ),
)
def test_get_acceptance_fractions(inp, exp):
    inp = np.array(inp)
    exp = np.array(exp)
    res = get_acceptance_fractions(inp)

    npt.assert_equal(res, exp)


@pytest.mark.parametrize("burnin", (0, 10))
@pytest.mark.parametrize(
    "thin_inp, thin_exp, autocorr_tol_inp, autocorr_tol_exp",
    (
        (3, 3, 1, 1),
        (None, 1, None, 0),
    ),
)
@pytest.mark.parametrize(
    "convergence_ratio_inp, convergence_ratio_exp",
    (
        (4, 4),
        (None, 50),
    ),
)
def test_get_autocorrelation_info(
    burnin,
    thin_inp,
    thin_exp,
    autocorr_tol_inp,
    autocorr_tol_exp,
    convergence_ratio_inp,
    convergence_ratio_exp,
):
    mock_chain_autocor_times = np.array([12.4, 13.6, 18.0])
    mock_iteration = 756

    mock_inp = Mock()
    mock_inp.get_autocorr_time = Mock()
    mock_inp.get_autocorr_time.return_value = mock_chain_autocor_times
    mock_inp.iteration = mock_iteration

    call_kwargs = {"burnin": burnin}

    if thin_inp is not None:
        call_kwargs["thin"] = thin_inp

    if autocorr_tol_inp is not None:
        call_kwargs["autocorr_tol"] = autocorr_tol_inp

    if convergence_ratio_inp is not None:
        call_kwargs["convergence_ratio"] = convergence_ratio_exp

    res = get_autocorrelation_info(mock_inp, **call_kwargs)

    mock_inp.get_autocorr_time.assert_called_with(
        discard=burnin, thin=thin_exp, tol=autocorr_tol_exp
    )

    steps_post_burnin = mock_iteration - burnin
    autocorr = np.mean(mock_chain_autocor_times)
    assert res == {
        "tau": mock_chain_autocor_times,
        "autocorr": autocorr,
        "converged": steps_post_burnin > convergence_ratio_exp * autocorr,
        "convergence_ratio": convergence_ratio_exp,
        "steps_post_burnin": steps_post_burnin,
    }


@pytest.mark.parametrize(
    "parameter_order",
    (
        ["para_ab", "parabcd", "paraz"],
        ["a", "b", "c"],
    ),
)
@pytest.mark.parametrize("neg_log_likelihood_name", (None, "neg_ll", "negative_log_l"))
@pytest.mark.parametrize(
    "burnin, burnin_exp, thin, thin_exp", ((None, 0, None, 0), (100, 100, 20, 20))
)
def test_get_labelled_chain_data(
    parameter_order, neg_log_likelihood_name, burnin, burnin_exp, thin, thin_exp
):
    inp = Mock()
    inp.get_chain = Mock()
    inp.get_chain.return_value = np.array(
        [
            [
                [1, 2, -2.8],
                [1.1, 2.1, -3.03],
            ],
            [
                [1.2, 2, -3],
                [1.05, 2.15, -3],
            ],
            [
                [1, 2.01, -3.2],
                [0.9, 3.1, -3.1],
            ],
        ]
    )
    inp.get_log_prob = Mock()
    inp.get_log_prob.return_value = np.array(
        [
            [-1, -1.5],
            [-2, -3],
            [-1.3, -0.4],
        ]
    )

    # Ignore burnin and thin for this expected value because it is all mocked
    # anyway, could be part of an integration test if we wanted
    exp = {
        parameter_order[0]: np.array(
            [
                [1, 1.1],
                [1.2, 1.05],
                [1, 0.9],
            ]
        ),
        parameter_order[1]: inp.get_chain.return_value[:, :, 1],
        parameter_order[2]: inp.get_chain.return_value[:, :, 2],
    }
    if neg_log_likelihood_name is not None:
        exp[neg_log_likelihood_name] = inp.get_log_prob.return_value

    call_kwargs = {}
    for name, para in (
        ("burnin", burnin),
        ("thin", thin),
    ):
        if para is not None:
            call_kwargs[name] = para

    res = get_labelled_chain_data(
        inp, parameter_order, neg_log_likelihood_name, **call_kwargs
    )

    inp.get_chain.assert_called_with(discard=burnin_exp, thin=thin_exp)
    if neg_log_likelihood_name is not None:
        inp.get_log_prob.assert_called_with(discard=burnin_exp, thin=thin_exp)
    else:
        inp.get_log_prob.assert_not_called()

    for key, res_v in res.items():
        npt.assert_equal(res_v, exp[key])
