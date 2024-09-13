"""
Tests of :mod:`openscm_calibration.cost`
"""

import re

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import scmdata.run
import scmdata.testing

from openscm_calibration.cost.scmdata import AlignmentError, OptCostCalculatorSSE
from openscm_calibration.exceptions import MissingValueError


@pytest.fixture()
def dummy_model_col():
    return "climate_model"


@pytest.fixture()
def dummy_target(dummy_model_col):
    timeseries = pd.DataFrame(
        [[1, 1.2], [2.3, 2.6]],
        columns=[2010, 2020],
        index=pd.MultiIndex.from_tuples(
            [
                ("Temperature", "K", "test_target"),
                ("Heat Uptake", "W / m^2", "test_target"),
            ],
            names=("variable", "unit", dummy_model_col),
        ),
    )

    return scmdata.run.BaseScmRun(timeseries)


@pytest.fixture()
def dummy_normalisation(dummy_target, dummy_model_col):
    norm = dummy_target.timeseries()
    norm.loc[:, :] = 1.3

    return type(dummy_target)(norm)


def test_from_unit_normalisation(dummy_target, dummy_model_col):
    calc = OptCostCalculatorSSE.from_unit_normalisation(
        dummy_target,
        dummy_model_col,
    )

    npt.assert_allclose(calc.calculate_cost(dummy_target), 0.0)


def test_dummy_setup(dummy_target, dummy_normalisation, dummy_model_col):
    calc = OptCostCalculatorSSE(
        target=dummy_target,
        normalisation=dummy_normalisation,
        model_col=dummy_model_col,
    )

    npt.assert_allclose(calc.calculate_cost(dummy_target), 0.0)


def test_normalisation_misaligned_with_target(
    dummy_target, dummy_normalisation, dummy_model_col
):
    normalisation_in = dummy_normalisation.copy()
    normalisation_in["variable"] = normalisation_in["variable"].replace(
        {"Temperature": "Ocean Temperature"}
    )

    error_msg = re.escape(
        "Note we have aligned the timeseries so that nan values appear where "
        "there are alignment issues"
    )
    with pytest.raises(AlignmentError, match=error_msg):
        OptCostCalculatorSSE(
            target=dummy_target,
            normalisation=normalisation_in,
            model_col=dummy_model_col,
        )


def test_normalisation_unit_agnostic(
    dummy_target, dummy_normalisation, dummy_model_col
):
    normalisation_in = dummy_normalisation.copy()
    normalisation_different_units = normalisation_in.convert_unit(
        "mK", variable="Temperature"
    )

    calc_k = OptCostCalculatorSSE(
        target=dummy_target,
        normalisation=dummy_normalisation,
        model_col=dummy_model_col,
    )

    calc_mk = OptCostCalculatorSSE(
        target=dummy_target,
        normalisation=normalisation_different_units,
        model_col=dummy_model_col,
    )

    assert calc_k.calculate_cost(dummy_target * 1.1) == calc_mk.calculate_cost(
        dummy_target * 1.1
    )


def test_model_col_missing_raises(dummy_target, dummy_normalisation):
    error_msg = re.escape(
        "``instance.target.meta_attributes`` is missing values: ``junk``. "
        "Available values: ``['climate_model', 'unit', 'variable']``"
    )
    with pytest.raises(MissingValueError, match=error_msg):
        OptCostCalculatorSSE(
            target=dummy_target,
            normalisation=dummy_normalisation,
            model_col="junk",
        )


def test_from_series_normalisation(dummy_model_col):
    years = [2010, 2020, 2030]
    ts_idx = pd.MultiIndex.from_tuples(
        [
            ("Temperature", "K", "1pctCO2", "test_target"),
            ("Heat Uptake", "W / m^2", "1pctCO2", "test_target"),
            ("Temperature", "K", "abrupt-2xCO2", "test_target"),
            ("Heat Uptake", "W / m^2", "abrupt-2xCO2", "test_target"),
        ],
        names=("variable", "unit", "scenario", dummy_model_col),
    )

    target = scmdata.run.BaseScmRun(
        pd.DataFrame(
            [
                [1, 1.2, 2.0],
                [2.3, 2.6, 2.0],
                [2, 2.2, 2.3],
                [3.1, 3.3, 2.8],
            ],
            columns=years,
            index=ts_idx,
        )
    )

    exp_normalisation = scmdata.run.BaseScmRun(
        pd.DataFrame(
            [
                [1.1, 1.1, 1.1],
                [2.2, 2.2, 2.2],
                [1.1, 1.1, 1.1],
                [2.2, 2.2, 2.2],
            ],
            columns=years,
            index=ts_idx,
        )
    )

    norm_series = pd.Series(
        [1.1, 2.2],
        index=pd.MultiIndex.from_tuples(
            [
                ("Temperature", "K"),
                ("Heat Uptake", "W / m^2"),
            ],
            names=("variable", "unit"),
        ),
    )

    calc = OptCostCalculatorSSE.from_series_normalisation(
        target=target,
        model_col=dummy_model_col,
        normalisation_series=norm_series,
    )

    scmdata.testing.assert_scmdf_almost_equal(
        exp_normalisation,
        calc.normalisation,
        allow_unordered=True,
        check_ts_names=False,
    )
    npt.assert_allclose(calc.calculate_cost(target), 0.0)


@pytest.mark.parametrize(
    "norm_values, exp_cost",
    (
        ([1.0, 10.0], np.sum([0.01, 0.01, 0.01, 0.01, 0.04])),
        ([1.0, 1e10], np.sum([0.01, 0.01])),
        ([1e10, 1.0], np.sum([1, 1, 4])),
    ),
)
def test_from_series_normalisation_weights(norm_values, exp_cost, dummy_model_col):
    years = [2010, 2020, 2030]
    ts_idx = pd.MultiIndex.from_tuples(
        [
            ("Temperature", "K", "test_target"),
            ("Heat Uptake", "W / m^2", "test_target"),
        ],
        names=("variable", "unit", dummy_model_col),
    )

    target_temp_vals = np.array([1, 1.5, 2.0])
    target_ohu_vals = np.array([20.0, 26.0, 30.0])
    target = scmdata.run.BaseScmRun(
        pd.DataFrame(
            np.vstack([target_temp_vals, target_ohu_vals]),
            columns=years,
            index=ts_idx,
        )
    )

    results_temp_vals = np.array([1.1, 1.4, 2.0])
    results_ohu_vals = np.array([21.0, 25.0, 32.0])
    results = scmdata.run.BaseScmRun(
        pd.DataFrame(
            np.vstack([results_temp_vals, results_ohu_vals]),
            columns=years,
            index=ts_idx,
        )
    )

    norm_series = pd.Series(
        norm_values,
        index=pd.MultiIndex.from_tuples(
            [
                ("Temperature", "K"),
                ("Heat Uptake", "W / m^2"),
            ],
            names=("variable", "unit"),
        ),
    )

    res_cost = OptCostCalculatorSSE.from_series_normalisation(
        target=target,
        model_col=dummy_model_col,
        normalisation_series=norm_series,
    ).calculate_cost(results)

    npt.assert_allclose(res_cost, exp_cost)


@pytest.mark.parametrize("required_col", (["variable"], ["unit"], ("variable", "unit")))
def test_from_series_normalisation_missing_required_col(
    required_col, dummy_target, dummy_model_col
):
    norm_series = pd.Series(
        [1.1, 2.2],
        index=pd.MultiIndex.from_tuples(
            [
                ("Temperature", "K"),
                ("Heat Uptake", "W / m^2"),
            ],
            names=("variable", "unit"),
        ),
    )
    norm_series = norm_series.reset_index(required_col, drop=True)

    error_msg = re.escape(
        "``normalisation_series.index.names`` is missing values: "
        f"``{sorted(set(required_col))}``. Available values: "
        f"``{sorted(set(norm_series.index.names))}``"
    )
    with pytest.raises(MissingValueError, match=error_msg):
        OptCostCalculatorSSE.from_series_normalisation(
            target=dummy_target,
            normalisation_series=norm_series,
            model_col=dummy_model_col,
        )


def test_from_series_normalisation_misaligned(dummy_target, dummy_model_col):
    norm_series = pd.Series(
        [1.1, 2.2],
        index=pd.MultiIndex.from_tuples(
            [
                ("Ocean Temperature", "K"),
                ("Heat Uptake", "W / m^2"),
            ],
            names=("variable", "unit"),
        ),
    )

    error_msg = re.escape("Even after aligning, there are still nan values")
    with pytest.raises(AlignmentError, match=error_msg):
        OptCostCalculatorSSE.from_series_normalisation(
            target=dummy_target,
            normalisation_series=norm_series,
            model_col=dummy_model_col,
        )


def test_from_series_normalisation_extra_val(dummy_target, dummy_model_col):
    norm_series = pd.Series(
        [1.1, 2.2, 3.0],
        index=pd.MultiIndex.from_tuples(
            [
                ("Temperature", "K"),
                ("Heat Uptake", "W / m^2"),
                ("Ocean Temperature", "K"),
            ],
            names=("variable", "unit"),
        ),
    )

    error_msg = re.escape(
        "After aligning, there are more rows in the normalisation than "
        "in the target."
    )
    with pytest.raises(AlignmentError, match=error_msg):
        OptCostCalculatorSSE.from_series_normalisation(
            target=dummy_target,
            normalisation_series=norm_series,
            model_col=dummy_model_col,
        )
