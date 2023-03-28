import re
from functools import partial
from unittest.mock import Mock, patch

import pint
import pint.errors
import pint.testing
import pytest
from openscm_units import unit_registry

from openscm_calibration.model_runner import (
    OptModelRunner,
    x_and_parameters_to_named_with_units,
)

UNIT_REG = unit_registry


def test_run_model():
    x = [1.1, 1.9]

    converted_x = {"a": "res_here", "b": "res_there"}
    mock_convert_x_to_names_with_units = Mock()
    mock_convert_x_to_names_with_units.return_value = converted_x

    model_run_input = {"model": "input", "here": "now"}
    mock_model_run_input_generator = Mock()
    mock_model_run_input_generator.return_value = model_run_input

    mock_do_model_runs = Mock()
    mock_do_model_runs.return_value = 3

    runner = OptModelRunner(
        mock_convert_x_to_names_with_units,
        mock_model_run_input_generator,
        mock_do_model_runs,
    )

    res = runner.run_model(x)

    mock_convert_x_to_names_with_units.assert_called_with(x)
    mock_model_run_input_generator.assert_called_with(
        **mock_convert_x_to_names_with_units.return_value
    )
    mock_do_model_runs.assert_called_with(**mock_model_run_input_generator.return_value)
    assert res == mock_do_model_runs.return_value


@patch("openscm_calibration.model_runner.x_and_parameters_to_named_with_units")
def test_from_parameters_run_model(mock_get_params):
    x = [1.1, 1.9]
    parameters = [("a", "kg"), ("b", "m")]

    converted_paras = {"a": "res_here", "b": "res_there"}
    mock_get_params.return_value = converted_paras

    model_run_input = {"model": "input", "here": "now"}
    mock_model_run_input_generator = Mock()
    mock_model_run_input_generator.return_value = model_run_input

    mock_do_model_runs = Mock()
    mock_do_model_runs.return_value = 3

    runner = OptModelRunner.from_parameters(
        parameters,
        mock_model_run_input_generator,
        mock_do_model_runs,
    )

    res = runner.run_model(x)

    mock_get_params.assert_called_with(x, params=parameters)
    mock_model_run_input_generator.assert_called_with(**mock_get_params.return_value)
    mock_do_model_runs.assert_called_with(**mock_model_run_input_generator.return_value)
    assert res == mock_do_model_runs.return_value


@pytest.mark.parametrize(
    "x_in, params_in, exp",
    (
        (
            [1.1],
            (("a", "kg"),),
            {
                "a": UNIT_REG.Quantity(1.1, "kg"),
            },
        ),
        (
            [1.2],
            (("a", None),),
            {
                "a": 1.2,
            },
        ),
        (
            [1.1, 2.2, 3.3],
            (
                ("a", "kg"),
                ("b", " W / m^2"),
                ("c", None),
            ),
            {
                "a": UNIT_REG.Quantity(1.1, "kg"),
                "b": UNIT_REG.Quantity(2.2, "W / m^2"),
                "c": 3.3,
            },
        ),
    ),
)
def test_x_and_parameters_to_named_with_units(x_in, params_in, exp):
    res = x_and_parameters_to_named_with_units(
        x=x_in,
        params=params_in,
    )

    for k, exp_v in exp.items():
        if isinstance(exp_v, pint.Quantity):
            pint.testing.assert_allclose(exp_v, res[k])
        else:
            assert exp_v == res[k]


def test_convert_x_to_names_with_units_dependency_injection():
    x = [1.1, 2.2, 0.3]
    parameters = [("a", "kg"), ("b", None), ("c", "thousands")]
    mock_model_run_input_generator = Mock()
    mock_model_run_input_generator.return_value = {}
    mock_do_model_runs = Mock()

    convert_x_to_names_with_units_pop = partial(
        x_and_parameters_to_named_with_units,
        params=parameters,
    )
    no_pop = OptModelRunner(
        convert_x_to_names_with_units=convert_x_to_names_with_units_pop,
        do_model_runs_input_generator=mock_model_run_input_generator,
        do_model_runs=mock_do_model_runs,
    )

    error_msg = re.escape("'thousands' is not defined in the unit registry")
    with pytest.raises(pint.errors.UndefinedUnitError, match=error_msg):
        no_pop.run_model(x)

    ur_plus_pop = pint.UnitRegistry()
    ur_plus_pop.define("thousands = [population]")

    def get_unit_registry_pop():
        return ur_plus_pop

    convert_x_to_names_with_units_pop = partial(
        x_and_parameters_to_named_with_units,
        params=parameters,
        get_unit_registry=get_unit_registry_pop,
    )

    with_pop = OptModelRunner(
        convert_x_to_names_with_units=convert_x_to_names_with_units_pop,
        do_model_runs_input_generator=mock_model_run_input_generator,
        do_model_runs=mock_do_model_runs,
    )

    # Should run without issue
    with_pop.run_model(x)

    exp_call_inp = convert_x_to_names_with_units_pop(x)
    mock_model_run_input_generator.assert_called_with(**exp_call_inp)
