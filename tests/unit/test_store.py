"""
Tests of :mod:`openscm_calibration.store`
"""

import copy
import re
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import numpy as np
import numpy.testing as npt
import pytest
from multiprocess import Manager

from openscm_calibration.store import OptResStore


@pytest.fixture()
def dummy_res():
    return [None] * 4


@pytest.fixture()
def dummy_costs(dummy_res):
    return copy.deepcopy(dummy_res)


@pytest.fixture()
def dummy_xs(dummy_res):
    return copy.deepcopy(dummy_res)


@pytest.fixture()
def dummy_avail_indices(dummy_res):
    return list(range(len(dummy_res)))


@pytest.fixture()
def dummy_parameters():
    return ["a", "b", "c"]


@pytest.fixture()
def dummy_store(
    dummy_res, dummy_costs, dummy_xs, dummy_parameters, dummy_avail_indices
):
    return OptResStore(
        res=dummy_res,
        costs=dummy_costs,
        x_samples=dummy_xs,
        params=dummy_parameters,
        available_indices=dummy_avail_indices,
    )


def test_get_available_index_mock(dummy_store):
    with patch(
        "openscm_calibration.store.OptResStore.available_indices",
        new_callable=PropertyMock,
    ) as mock_avail_ind:
        mock_avail_ind.return_value = Mock()
        mock_avail_ind.return_value.pop = Mock()
        mock_avail_ind.return_value.pop.return_value = 3

        res = dummy_store.get_available_index()

        dummy_store.available_indices.pop.assert_called_once()

        assert res == 3


def test_get_available_index(dummy_store):
    assert dummy_store.get_available_index() == 3
    assert dummy_store.get_available_index() == 2
    assert dummy_store.get_available_index() == 1


@pytest.mark.parametrize("res", (1, 3))
@pytest.mark.parametrize("cost", (1.01, 1.12))
@pytest.mark.parametrize(
    "x, parameters",
    (
        ([1.1, 1.2], ("a", "b")),
        ([1.0, 1.3], ("c", "d")),
        ([1.0, 1.3, -0.43], ("c", "d", "a")),
    ),
)
@pytest.mark.parametrize("idx", (0, 1, 2))
def test_set_result_cost_x(res, cost, x, parameters, idx):
    store = OptResStore.from_n_runs(3, params=parameters)

    store.set_result_cost_x(
        res=res,
        cost=cost,
        x=x,
        idx=idx,
    )

    assert store.costs[idx] == cost
    assert store.x_samples[idx] == x
    assert store.res[idx] == res


@pytest.mark.parametrize(
    "parameters, x",
    (
        (["a", "b", "c"], np.array([1, 2, 3, 4])),
        (["a", "b", "c"], np.array([1, 2])),
        (["a", "b"], np.array([1, 2, 3])),
    ),
)
def test_set_result_cost_x_raises(parameters, x):
    store = OptResStore.from_n_runs(3, params=parameters)

    error_msg = re.escape(
        f"``x`` has length {len(x)}, it should have length {len(parameters)}, "
        "the same as ``self.params``"
    )
    with pytest.raises(ValueError, match=error_msg):
        store.set_result_cost_x(
            res="a",
            cost=13,
            x=x,
            idx=0,
        )


@patch("openscm_calibration.store.OptResStore.get_available_index")
@patch("openscm_calibration.store.OptResStore.set_result_cost_x")
@pytest.mark.parametrize("cost", (1.01, 1.12))
@pytest.mark.parametrize("x", ([1.1, 1.2], [1.0, 1.3]))
def test_append_result_cost_x(
    mock_set_result_cost_x,
    mock_get_available_index,
    cost,
    x,
    dummy_store,
):
    idx = 11
    mock_get_available_index.return_value = idx

    res = MagicMock()
    res_keep = MagicMock()
    res.copy.return_value = res_keep

    dummy_store.append_result_cost_x(
        res=res,
        cost=cost,
        x=x,
    )

    res.copy.assert_called_once()
    res_keep.__setitem__.assert_called_with("it", idx)

    mock_set_result_cost_x.assert_called_with(
        res=res_keep,
        cost=cost,
        x=x,
        idx=idx,
    )


@patch("openscm_calibration.store.OptResStore.get_available_index")
@patch("openscm_calibration.store.OptResStore.set_result_cost_x")
@pytest.mark.parametrize("cost", (1.01, 1.12))
@pytest.mark.parametrize("x", ([1.1, 1.2], [1.0, 1.3]))
def test_note_failed_run(
    mock_set_result_cost_x,
    mock_get_available_index,
    cost,
    x,
    dummy_store,
):
    idx = 5
    mock_get_available_index.return_value = idx

    dummy_store.note_failed_run(
        cost=cost,
        x=x,
    )

    mock_set_result_cost_x.assert_called_with(
        res=None,
        cost=cost,
        x=x,
        idx=idx,
    )


@patch("openscm_calibration.store.OptResStore.costs", new_callable=PropertyMock)
@patch("openscm_calibration.store.OptResStore.x_samples", new_callable=PropertyMock)
@patch("openscm_calibration.store.OptResStore.res", new_callable=PropertyMock)
@pytest.mark.parametrize(
    "costs, x_samples, res, exp",
    (
        pytest.param(
            [1, 2, None],
            [[1, 1], [1, 2], None],
            ["a", "b", None],
            (
                (1, 2),
                ([1, 1], [1, 2]),
                ("a", "b"),
            ),
            id="basic",
        ),
        pytest.param(
            [1, 2, 3],
            [[1, 1], [1, 2], [3, 1]],
            ["a", "b", "c"],
            (
                (1, 2, 3),
                ([1, 1], [1, 2], [3, 1]),
                ("a", "b", "c"),
            ),
            id="all-run",
        ),
        pytest.param(
            [1, None, 3],
            [[1, 1], None, [3, 1]],
            ["a", None, "c"],
            (
                (1, 3),
                ([1, 1], [3, 1]),
                ("a", "c"),
            ),
            id="second-not-attempted",
        ),
        pytest.param(
            [np.inf, None, 3],
            [[1, 1], None, [3, 1]],
            [None, None, "c"],
            (
                (np.inf, 3),
                ([1, 1], [3, 1]),
                (None, "c"),
            ),
            id="first-failed-second-not-attempted",
        ),
        pytest.param(
            [None, None],
            [None, None],
            [None, None],
            (
                (),
                (),
                (),
            ),
            id="none-attempted",
        ),
        pytest.param(
            [1.2, None],
            [[0.2, -0.3], [1, -1]],
            ["a", "c"],
            (
                (1.2, None),
                ([0.2, -0.3], [1, -1]),
                ("a", "c"),
            ),
            id="x_samples-out-of-sync-with-costs",
        ),
    ),
)
def test_get_costs_xsamples_res(
    mock_res, mock_xs, mock_costs, dummy_store, costs, x_samples, res, exp
):
    mock_costs.return_value = costs
    mock_xs.return_value = x_samples
    mock_res.return_value = res

    res = dummy_store.get_costs_xsamples_res()
    assert res == exp


@patch("openscm_calibration.store.OptResStore.costs", new_callable=PropertyMock)
@patch("openscm_calibration.store.OptResStore.x_samples", new_callable=PropertyMock)
@patch("openscm_calibration.store.OptResStore.res", new_callable=PropertyMock)
@pytest.mark.parametrize(
    "costs, parameters, x_samples, res, exp",
    (
        pytest.param(
            [1, 2, None],
            ["para_a", "para_b"],
            [[1, 1], [1, 2], None],
            ["a", "b", None],
            (
                (1, 2),
                {
                    "para_a": [1, 1],
                    "para_b": [1, 2],
                },
                ("a", "b"),
            ),
            id="basic",
        ),
        pytest.param(
            [1, 2, 3],
            ["para_a", "para_b"],
            [[1, 2], [1, 1], [3, 2]],
            ["a", "b", "c"],
            (
                (1, 2, 3),
                {
                    "para_a": [1, 1, 3],
                    "para_b": [2, 1, 2],
                },
                ("a", "b", "c"),
            ),
            id="all-run",
        ),
        pytest.param(
            [1, None, 3],
            ["para_a", "para_b"],
            [[1, 1], None, [3, 1]],
            ["a", None, "c"],
            (
                (1, 3),
                {
                    "para_a": [1, 3],
                    "para_b": [1, 1],
                },
                ("a", "c"),
            ),
            id="second-not-attempted",
        ),
        pytest.param(
            [np.inf, None, 3],
            ["para_a", "para_b"],
            [[1, 1], None, [3, 1]],
            [None, None, "c"],
            (
                (np.inf, 3),
                {
                    "para_a": [1, 3],
                    "para_b": [1, 1],
                },
                (None, "c"),
            ),
            id="first-failed-second-not-attempted",
        ),
        pytest.param(
            [None, None],
            ["para_a", "para_b"],
            [None, None],
            [None, None],
            (
                (),
                {"para_a": np.array([]), "para_b": np.array([])},
                (),
            ),
            id="none-attempted",
        ),
        pytest.param(
            [1.2, None],
            ["para_a", "para_b"],
            [[0.2, -0.3], [1, -1]],
            ["a", "c"],
            (
                (1.2, None),
                {
                    "para_a": [0.2, 1],
                    "para_b": [-0.3, -1],
                },
                ("a", "c"),
            ),
            id="x_samples-out-of-sync-with-costs",
        ),
        pytest.param(
            [1.2, 10, 13],
            ["para_a", "para_b"],
            [[0.2, -0.3], [1, -1], [1.3, -2]],
            ["a", "c", "d"],
            (
                (1.2, 10, 13),
                {
                    "para_a": [0.2, 1, 1.3],
                    "para_b": [-0.3, -1, -2],
                },
                ("a", "c", "d"),
            ),
            id="2-parameters_3-samples",
        ),
    ),
)
def test_get_costs_labelled_xsamples_res(
    mock_res, mock_xs, mock_costs, dummy_store, costs, parameters, x_samples, res, exp
):
    dummy_store.params = parameters

    mock_costs.return_value = costs
    mock_xs.return_value = x_samples
    mock_res.return_value = res

    res = dummy_store.get_costs_labelled_xsamples_res()

    if not any(exp):
        # Graceful handling if no runs have been attempted
        assert res == exp

    else:
        assert res[0] == exp[0]

        for key, res_val in res[1].items():
            exp_val = exp[1][key]
            npt.assert_equal(res_val, exp_val)

        assert res[2] == exp[2]


@pytest.mark.parametrize("in_name", ("costs", "x_samples"))
def test_init_wrong_length(
    dummy_res, dummy_costs, dummy_xs, dummy_parameters, dummy_avail_indices, in_name
):
    inp = {
        "costs": dummy_costs,
        "x_samples": dummy_xs,
    }
    inp[in_name] = inp[in_name][:-1]

    error_msg = re.escape(
        f"``{in_name}`` has length {len(inp[in_name])}, it should have length "
        f"{len(dummy_res)}, the same as ``res``"
    )
    with pytest.raises(ValueError, match=error_msg):
        OptResStore(
            res=dummy_res,
            params=dummy_parameters,
            available_indices=dummy_avail_indices,
            **inp,
        )


@pytest.mark.parametrize("in_name", ("res", "costs", "x_samples"))
def test_init_not_none(
    dummy_res, dummy_costs, dummy_xs, dummy_parameters, dummy_avail_indices, in_name
):
    inp = {
        "res": dummy_res,
        "costs": dummy_costs,
        "x_samples": dummy_xs,
    }
    inp[in_name][1] = 3

    error_msg = re.escape(f"All values in ``{in_name}`` should be ``None``")
    with pytest.raises(ValueError, match=error_msg):
        OptResStore(
            available_indices=dummy_avail_indices,
            params=dummy_parameters,
            **inp,
        )


def test_init_indices_can_shuffle(
    dummy_res, dummy_costs, dummy_xs, dummy_parameters, dummy_avail_indices
):
    dummy_avail_indices = dummy_avail_indices[::-1]

    OptResStore(
        res=dummy_res,
        costs=dummy_costs,
        x_samples=dummy_xs,
        params=dummy_parameters,
        available_indices=dummy_avail_indices,
    )


def test_init_indices_wrong(
    dummy_res, dummy_costs, dummy_xs, dummy_parameters, dummy_avail_indices
):
    dummy_avail_indices[-1] = len(dummy_avail_indices) + 1

    error_msg = re.escape(
        f"``available_indices`` must have value: {list(range(len(dummy_res)))}, "
        f"received: {dummy_avail_indices}"
    )
    with pytest.raises(ValueError, match=error_msg):
        OptResStore(
            res=dummy_res,
            costs=dummy_costs,
            x_samples=dummy_xs,
            params=dummy_parameters,
            available_indices=dummy_avail_indices,
        )


@pytest.mark.parametrize("n_runs", (3, 5, 23))
def test_from_n_runs(n_runs, dummy_parameters):
    init = OptResStore.from_n_runs(
        n_runs,
        params=dummy_parameters,
    )

    assert len(init.res) == n_runs
    assert len(init.costs) == n_runs
    assert len(init.x_samples) == n_runs
    assert init.params == dummy_parameters
    assert init.available_indices == list(range(n_runs))[::-1]


@pytest.mark.parametrize("n_runs", (3, 5, 23))
def test_from_n_runs_manager(n_runs, dummy_parameters):
    mock_manager = Mock()
    mock_manager.list = Mock()
    mock_manager.list.side_effect = list

    init = OptResStore.from_n_runs_manager(
        n_runs, manager=mock_manager, params=dummy_parameters
    )

    exp_res = [None] * n_runs
    exp_avail_indices = list(range(n_runs))[::-1]
    mock_manager.list.assert_has_calls(
        [
            call(exp_res),
            call(exp_res),
            call(exp_res),
            call(exp_avail_indices),
        ]
    )

    assert init.res == exp_res
    assert init.costs == exp_res
    assert init.x_samples == exp_res
    assert init.params == dummy_parameters
    assert init.available_indices == exp_avail_indices


@pytest.mark.parametrize("n_runs", (3, 5, 23))
def test_from_n_runs_manager_multiprocess(n_runs, dummy_parameters):
    exp_res = [None] * n_runs
    exp_avail_indices = list(range(n_runs))[::-1]

    with Manager() as manager:
        init = OptResStore.from_n_runs_manager(
            n_runs, manager=manager, params=dummy_parameters
        )

        assert list(init.res) == exp_res
        assert list(init.costs) == exp_res
        assert list(init.x_samples) == exp_res
        assert list(init.available_indices) == exp_avail_indices
