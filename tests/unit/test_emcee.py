import numpy as np
import numpy.testing as npt
import pytest

from openscm_calibration.emcee import get_acceptance_fractions


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
