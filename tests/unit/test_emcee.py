from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest

from openscm_calibration.emcee import get_acceptance_fractions, get_autocorrelation_info


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
