import inspect
from unittest.mock import Mock

import numpy as np
import pytest

from openscm_calibration.minimize import to_minimize_full


def get_mock_model_runner(inp_x):
    mock_model_runner = Mock()
    mock_model_runner.run_model = Mock()

    if inspect.isclass(inp_x) and issubclass(inp_x, Exception):
        mock_model_runner.run_model.side_effect = inp_x
    elif not inspect.isclass(inp_x):
        mock_model_runner.run_model.return_value = inp_x
    else:
        raise NotImplementedError(inp_x)

    return mock_model_runner


def get_mock_cost_calc(cost):
    mock_cost_calc = Mock()
    mock_cost_calc.calculate_cost = Mock()
    mock_cost_calc.calculate_cost.return_value = cost

    return mock_cost_calc


def test_no_known_error_no_store_return():
    x = np.array([1, 3.2, 4.1])
    run_model_return = "scmrun goes here"
    cost = 3.01

    mock_model_runner = get_mock_model_runner(run_model_return)
    mock_cost_calc = get_mock_cost_calc(cost)

    res = to_minimize_full(
        x=x,
        cost_calculator=mock_cost_calc,
        model_runner=mock_model_runner,
    )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_called_with(
        mock_model_runner.run_model.return_value
    )
    assert res == cost


def test_no_known_error_store_return():
    x = np.array([1, 3.2, 4.1])
    run_model_return = "scmrun goes here"
    cost = 3.01

    mock_model_runner = get_mock_model_runner(run_model_return)
    mock_cost_calc = get_mock_cost_calc(cost)
    mock_store = Mock()

    res = to_minimize_full(
        x=x,
        cost_calculator=mock_cost_calc,
        model_runner=mock_model_runner,
        store=mock_store,
    )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_called_with(
        mock_model_runner.run_model.return_value
    )
    mock_store.append_result_cost_x.assert_called_with(
        mock_model_runner.run_model.return_value,
        cost,
        x,
    )
    assert res == cost


def test_no_known_error_no_store_raise():
    x = np.array([1, 3.2, 4.1])
    run_model_return = ValueError

    mock_model_runner = get_mock_model_runner(run_model_return)
    mock_cost_calc = get_mock_cost_calc(3)

    with pytest.raises(run_model_return):
        to_minimize_full(
            x=x,
            cost_calculator=mock_cost_calc,
            model_runner=mock_model_runner,
        )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_not_called()


def test_no_known_error_store_raise():
    x = np.array([1, 3.2, 4.1])
    run_model_return = ValueError

    mock_model_runner = get_mock_model_runner(run_model_return)
    mock_cost_calc = get_mock_cost_calc(3)
    mock_store = Mock()

    with pytest.raises(run_model_return):
        to_minimize_full(
            x=x,
            cost_calculator=mock_cost_calc,
            model_runner=mock_model_runner,
            store=mock_store,
        )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_not_called()
    mock_store.append_result_cost_x.assert_not_called()


def test_known_error_no_store_return():
    x = np.array([1, 3.2, 4.1])
    run_model_return = "scmrun goes here"
    cost = 3.01
    known_error = ValueError

    mock_model_runner = get_mock_model_runner(run_model_return)
    mock_cost_calc = get_mock_cost_calc(cost)

    res = to_minimize_full(
        x=x,
        cost_calculator=mock_cost_calc,
        model_runner=mock_model_runner,
        known_error=known_error,
    )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_called_with(
        mock_model_runner.run_model.return_value
    )
    assert res == cost


def test_known_error_store_return():
    x = np.array([1, 3.2, 4.1])
    run_model_return = "scmrun goes here"
    cost = 3.01
    known_error = ValueError

    mock_model_runner = get_mock_model_runner(run_model_return)
    mock_cost_calc = get_mock_cost_calc(cost)
    mock_store = Mock()

    res = to_minimize_full(
        x=x,
        cost_calculator=mock_cost_calc,
        model_runner=mock_model_runner,
        known_error=known_error,
        store=mock_store,
    )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_called_with(
        mock_model_runner.run_model.return_value
    )
    mock_store.append_result_cost_x.assert_called_with(
        mock_model_runner.run_model.return_value,
        cost,
        x,
    )
    assert res == cost


def test_known_error_no_store_raises():
    x = np.array([1, 3.2, 4.1])
    known_error = ValueError

    mock_model_runner = get_mock_model_runner(known_error)
    mock_cost_calc = get_mock_cost_calc(1.01)

    res = to_minimize_full(
        x=x,
        cost_calculator=mock_cost_calc,
        model_runner=mock_model_runner,
        known_error=known_error,
    )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_not_called()

    assert np.isinf(res)


def test_known_error_store_raises():
    x = np.array([1, 3.2, 4.1])
    known_error = ValueError

    mock_model_runner = get_mock_model_runner(known_error)
    mock_cost_calc = get_mock_cost_calc(1.01)
    mock_store = Mock()

    res = to_minimize_full(
        x=x,
        cost_calculator=mock_cost_calc,
        model_runner=mock_model_runner,
        known_error=known_error,
        store=mock_store,
    )

    mock_model_runner.run_model.assert_called_with(x)
    mock_cost_calc.calculate_cost.assert_not_called()
    mock_store.note_failed_run.assert_called_with(np.inf, x)

    assert np.isinf(res)
