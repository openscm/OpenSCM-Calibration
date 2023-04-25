import logging
import re
from functools import partial
from unittest.mock import Mock, call, patch

import more_itertools
import numpy as np
import pandas as pd
import pytest
import scmdata.run
import scmdata.testing

from openscm_calibration.scipy_plotting import (
    DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS,
    DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS,
    DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS,
    CallbackProxy,
    OptPlotter,
    convert_target_to_model_output_units,
    get_optimisation_mosaic,
    get_runs_to_plot,
    get_ymax_default,
    plot_costs,
    plot_parameters,
    plot_timeseries,
)
from openscm_calibration.scmdata_utils import scmrun_as_dict


@pytest.fixture
def dummy_init_kwargs():
    cost_key = "cost"
    parameters = ("a", "A", "b")
    timeseries_axes = ("ts1", "ts2")

    axes = {k: Mock() for k in [cost_key] + list(parameters) + list(timeseries_axes)}
    convert_scmrun_to_plot_dict = Mock(
        return_value={k: Mock() for k in timeseries_axes}
    )

    return {
        "holder": Mock(),
        "fig": Mock(),
        "axes": axes,
        "cost_key": cost_key,
        "parameters": parameters,
        "timeseries_axes": timeseries_axes,
        "convert_scmrun_to_plot_dict": convert_scmrun_to_plot_dict,
        "target": Mock(),
        "store": Mock(),
        "thin_ts_to_plot": 30,
        "plot_cost_kwargs": {"lw": 5},
        "plot_parameters_kwargs": {"lw": 5, "legend": True},
        "plot_timeseries_kwargs": {"alpha": 0.5},
        "get_timeseries": Mock(),
    }


def test_dummy_setup(dummy_init_kwargs):
    OptPlotter(**dummy_init_kwargs)


@pytest.mark.parametrize("missing_paras", (["b"], ["a", "b"], ["a", "A"]))
def test_parameter_axis_missing(missing_paras, dummy_init_kwargs):
    dummy_init_kwargs["axes"] = {
        k: v for k, v in dummy_init_kwargs["axes"].items() if k not in missing_paras
    }

    error_msg = re.escape(
        "Error setting 'parameters'. "
        f"The following keys are missing from ``self.axes``: ``{missing_paras}``. "
        f"``self.axes.keys()`` is ``{dummy_init_kwargs['axes'].keys()}``."
    )
    with pytest.raises(KeyError, match=error_msg):
        OptPlotter(**dummy_init_kwargs)


@pytest.mark.parametrize("missing_ts", (["ts1"], ["ts2"], ["ts1", "ts2"]))
def test_timeseries_axes_missing(missing_ts, dummy_init_kwargs):
    dummy_init_kwargs["axes"] = {
        k: v for k, v in dummy_init_kwargs["axes"].items() if k not in missing_ts
    }

    error_msg = re.escape(
        "Error setting 'timeseries_axes'. "
        f"The following keys are missing from ``self.axes``: ``{missing_ts}``. "
        f"``self.axes.keys()`` is ``{dummy_init_kwargs['axes'].keys()}``."
    )
    with pytest.raises(KeyError, match=error_msg):
        OptPlotter(**dummy_init_kwargs)


def test_timeseries_axes_convert_run_target_incompatible(dummy_init_kwargs):
    ts_axes = dummy_init_kwargs["timeseries_axes"]

    years = [2010, 2020]
    index = pd.MultiIndex.from_tuples(
        [(name, "K") for name in ts_axes[:-1]], names=("variable", "unit")
    )
    df = pd.DataFrame(
        np.random.random(size=(index.size, len(years))),
        columns=years,
        index=index,
    )
    target = scmdata.run.BaseScmRun(df)

    convert_scmrun_to_plot_dict = partial(scmrun_as_dict, groups=("variable",))
    target_converted = convert_scmrun_to_plot_dict(target)

    dummy_init_kwargs["target"] = target
    dummy_init_kwargs["convert_scmrun_to_plot_dict"] = convert_scmrun_to_plot_dict
    error_msg = re.escape(
        "Error setting 'timeseries_axes'. The following keys are missing when "
        "``self.convert_scmrun_to_plot_dict`` is applied to "
        f"``self.target``: ``{list(ts_axes[-1:])}``. "
        f"``target_converted.keys()`` is ``{target_converted.keys()}``."
    )
    with pytest.raises(ValueError, match=error_msg):
        OptPlotter(**dummy_init_kwargs)


@pytest.mark.parametrize(
    ",".join(
        [
            "cost_key",
            "parameters",
            "target",
            "store",
        ]
    ),
    (
        ("cost_key", ["param_a", "param_b"], Mock(), Mock()),
        ("cost", ["123", "abc"], "target", "store"),
    ),
)
@pytest.mark.parametrize(
    ",".join(
        [
            "kwargs_create_mosaic",
            "kwargs_create_mosaic_exp",
            "kwargs_get_fig_axes_holder",
            "kwargs_get_fig_axes_holder_exp",
            "extra_kwargs",
            "extra_kwargs_exp",
        ]
    ),
    (
        ({"cost_col_relwidth": 2}, {"cost_col_relwidth": 2}, None, {}, None, {}),
        (
            None,
            {},
            {"figsize": (6, 6)},
            {"figsize": (6, 6)},
            {"thin_ts_to_plot": 10},
            {"thin_ts_to_plot": 10},
        ),
    ),
)
@patch("openscm_calibration.scipy_plotting.OptPlotter.__init__", return_value=None)
@patch("openscm_calibration.scipy_plotting.get_fig_axes_holder_from_mosaic")
@patch("openscm_calibration.scipy_plotting.get_optimisation_mosaic")
def test_from_autogenerated_figure(
    mock_get_optimisation_mosaic,
    mock_get_fig_axes_holder_from_mosaic,
    mock_init,
    cost_key,
    parameters,
    target,
    store,
    kwargs_create_mosaic,
    kwargs_create_mosaic_exp,
    kwargs_get_fig_axes_holder,
    kwargs_get_fig_axes_holder_exp,
    extra_kwargs,
    extra_kwargs_exp,
):
    ts_axes_exp = ["ts_a", "ts_b"]
    convert_scmrun_to_plot_dict = Mock(return_value={k: 1 for k in ts_axes_exp})

    mock_get_optimisation_mosaic.return_value = [["mosaic", "here"]]
    mock_get_fig_axes_holder_from_mosaic.return_value = ("fig", "axes", "holder")

    call_kwargs = {}

    for name, para in (
        ("kwargs_create_mosaic", kwargs_create_mosaic),
        ("kwargs_get_fig_axes_holder", kwargs_get_fig_axes_holder),
    ):
        if para is not None:
            call_kwargs[name] = para

    if extra_kwargs is not None:
        call_kwargs = call_kwargs | extra_kwargs

    res = OptPlotter.from_autogenerated_figure(
        cost_key,
        parameters,
        convert_scmrun_to_plot_dict,
        target,
        store,
        **call_kwargs,
    )

    mock_get_optimisation_mosaic.assert_called_once_with(
        cost_key=cost_key,
        params=parameters,
        timeseries=tuple(ts_axes_exp),
        **kwargs_create_mosaic_exp,
    )

    mock_get_fig_axes_holder_from_mosaic.assert_called_once_with(
        mock_get_optimisation_mosaic.return_value, **kwargs_get_fig_axes_holder_exp
    )

    mock_init.assert_called_once_with(
        holder=mock_get_fig_axes_holder_from_mosaic.return_value[2],
        fig=mock_get_fig_axes_holder_from_mosaic.return_value[0],
        axes=mock_get_fig_axes_holder_from_mosaic.return_value[1],
        cost_key=cost_key,
        parameters=parameters,
        timeseries_axes=tuple(ts_axes_exp),
        convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
        target=target,
        store=store,
        **extra_kwargs_exp,
    )

    assert isinstance(res, OptPlotter)


@patch("openscm_calibration.scipy_plotting.OptPlotter.__init__", return_value=None)
@patch("openscm_calibration.scipy_plotting.OptPlotter.update_plots")
def test_callback_minimize(mock_update_plots, mock_init):
    plotter = OptPlotter()
    plotter.callback_minimize([1, 3, 4])
    mock_update_plots.assert_called_once_with()


@patch("openscm_calibration.scipy_plotting.OptPlotter.__init__", return_value=None)
@patch("openscm_calibration.scipy_plotting.OptPlotter.update_plots")
def test_callback_differential_evolution(mock_update_plots, mock_init):
    plotter = OptPlotter()
    plotter.callback_differential_evolution([1, 3, 4], 0.4)
    mock_update_plots.assert_called_once_with()


@pytest.mark.parametrize(
    "thin_ts_to_plot, thin_ts_to_plot_exp, plot_cost_kwargs, "
    "plot_cost_kwargs_exp, plot_parameters_kwargs, "
    "plot_parameters_kwargs_exp, plot_timeseries_kwargs, "
    "plot_timeseries_kwargs_exp, get_timeseries",
    (
        (None, 20, None, {}, None, {}, None, {}, None),
        (
            10,
            10,
            {"lw": 3},
            {"lw": 3},
            {"alpha": 0.7},
            {"alpha": 0.7},
            {"zorder": 3},
            {"zorder": 3},
            "mocked",
        ),
    ),
)
@patch("openscm_calibration.scipy_plotting.get_timeseries_default")
@patch("openscm_calibration.scipy_plotting.plot_timeseries")
@patch("openscm_calibration.scipy_plotting.get_runs_to_plot")
@patch("openscm_calibration.scipy_plotting.plot_parameters")
@patch("openscm_calibration.scipy_plotting.plot_costs")
def test_update_plots(
    mock_plot_costs,
    mock_plot_parameters,
    mock_get_runs_to_plot,
    mock_plot_timeseries,
    mock_get_timeseries_default,
    thin_ts_to_plot,
    thin_ts_to_plot_exp,
    plot_cost_kwargs,
    plot_cost_kwargs_exp,
    plot_parameters_kwargs,
    plot_parameters_kwargs_exp,
    plot_timeseries_kwargs,
    plot_timeseries_kwargs_exp,
    get_timeseries,
):
    if get_timeseries is None:
        get_timeseries_exp = mock_get_timeseries_default
    elif get_timeseries == "mocked":
        get_timeseries = Mock()
        get_timeseries_exp = get_timeseries
    else:
        raise NotImplementedError()

    mock_get_runs_to_plot.return_value = ("best", "others")

    holder = Mock()
    fig = Mock()
    cost_key = "cost"
    parameters = ["a", "b", "c"]
    timeseries_axes = ["out_1", "out_2"]
    convert_scmrun_to_plot_dict = Mock(
        return_value={k: Mock() for k in timeseries_axes}
    )
    target = Mock()

    axes = {k: Mock() for k in [cost_key] + parameters + timeseries_axes}

    store = Mock()
    store.get_costs_labelled_xsamples_res.return_value = (
        (10, np.inf, 31, 1.0),
        [[1.0, 10.0, 31.2], [-0.3, 11.1, 30.2], [2.1, 9.8, 23.0], [2.03, 4.3, 22.0]],
        ["passed", "to", "mocked", "functions"],
    )

    init_kwargs = {}

    for name, para in (
        ("thin_ts_to_plot", thin_ts_to_plot),
        ("plot_cost_kwargs", plot_cost_kwargs),
        ("plot_parameters_kwargs", plot_parameters_kwargs),
        ("plot_timeseries_kwargs", plot_timeseries_kwargs),
        ("get_timeseries", get_timeseries),
    ):
        if para is not None:
            init_kwargs[name] = para

    plotter = OptPlotter(
        holder=holder,
        fig=fig,
        axes=axes,
        cost_key=cost_key,
        parameters=parameters,
        timeseries_axes=timeseries_axes,
        convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
        target=target,
        store=store,
        **init_kwargs,
    )

    assert plotter.update_plots() is None

    store.get_costs_labelled_xsamples_res.assert_called_once_with()

    mock_plot_costs.assert_called_once_with(
        ax=axes[cost_key],
        ylabel=cost_key,
        costs=store.get_costs_labelled_xsamples_res.return_value[0],
        **plot_cost_kwargs_exp,
    )

    mock_plot_parameters.assert_called_once_with(
        axes=axes,
        para_vals=store.get_costs_labelled_xsamples_res.return_value[1],
        **plot_parameters_kwargs_exp,
    )

    mock_get_runs_to_plot.assert_called_once_with(
        store.get_costs_labelled_xsamples_res.return_value[0],
        store.get_costs_labelled_xsamples_res.return_value[2],
        thin_ts_to_plot_exp,
    )

    mock_plot_timeseries.assert_called_once_with(
        best_run="best",
        others_to_plot="others",
        target=target,
        convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
        timeseries_keys=timeseries_axes,
        axes=axes,
        get_timeseries=get_timeseries_exp,
        **plot_timeseries_kwargs_exp,
    )

    for ax in axes.values():
        ax.clear.assert_called_once_with()

    fig.tight_layout.assert_called_once_with()
    holder.update.assert_called_once_with(fig)


@patch("openscm_calibration.scipy_plotting.plot_costs")
@patch("openscm_calibration.scipy_plotting.OptPlotter.__init__", return_value=None)
def test_update_plots_no_successful_runs(mock_init, mock_plot_costs, caplog):
    caplog.set_level(logging.INFO)

    store = Mock()
    store.get_costs_labelled_xsamples_res.return_value = (
        (np.inf, np.inf, np.inf),
        None,
        None,
    )

    plotter = OptPlotter()
    plotter.store = store
    assert plotter.update_plots() is None

    mock_plot_costs.assert_not_called()
    assert caplog.record_tuples == [
        (
            "openscm_calibration.scipy_plotting",
            logging.INFO,
            "No runs succeeded, nothing to plot",
        )
    ]


@pytest.mark.parametrize("get_ymax", (None, "mock"))
@pytest.mark.parametrize(
    "get_ymax_return_value, ymax_exp", ((32.1, 32.1), (np.inf, 10**3))
)
@pytest.mark.parametrize(
    "ymin, ymin_exp, alpha, alpha_exp, extra_kwargs, extra_kwargs_exp",
    ((None, 0.0, 0.8, 0.8, None, {}), (10.0, 10.0, None, 0.7, {"lw": 3}, {"lw": 3})),
)
@patch("openscm_calibration.scipy_plotting.get_ymax_default")
def test_plot_costs(
    mock_get_ymax_default,
    get_ymax,
    get_ymax_return_value,
    ymax_exp,
    ymin,
    ymin_exp,
    alpha,
    alpha_exp,
    extra_kwargs,
    extra_kwargs_exp,
):
    ylabel = "some string"
    costs = (1.3, 44.3, 4.9)

    ax = Mock()

    if get_ymax is None:
        mock_get_ymax_default.return_value = get_ymax_return_value
        get_ymax_exp = mock_get_ymax_default
    elif get_ymax == "mock":
        get_ymax = Mock()
        get_ymax.return_value = get_ymax_return_value
        get_ymax_exp = get_ymax
    else:
        raise NotImplementedError()

    call_kwargs = {}

    for name, para in (
        ("ymin", ymin),
        ("get_ymax", get_ymax),
        ("alpha", alpha),
    ):
        if para is not None:
            call_kwargs[name] = para

    if extra_kwargs is not None:
        call_kwargs = call_kwargs | extra_kwargs

    plot_costs(
        ax,
        ylabel,
        costs,
        **call_kwargs,
    )

    ax.scatter.assert_called_once_with(
        range(len(costs)), costs, alpha=alpha_exp, **extra_kwargs_exp
    )
    ax.set_ylabel.assert_called_once_with(ylabel)

    get_ymax_exp.assert_called_once_with(costs)

    ax.set_ylim.assert_called_once_with(ymin=ymin_exp, ymax=ymax_exp)


@pytest.mark.parametrize(
    "costs, min_scale_factor, min_v_median_scale_factor, exp",
    (
        pytest.param((1.3, 43.0, 24.0), None, None, 13.0, id="defaults"),
        pytest.param(
            (1.3, 43.0, 24.0), 100.0, None, 24.0, id="big-scaled-min_median-dominates"
        ),
        pytest.param(
            (13.0, 43.0, 24.0), 100.0, None, 26.0, id="big-scaled-min_min-dominates"
        ),
        pytest.param(
            (13.0, 43.0, 24.0), None, 1, 24.0, id="big-scaled-min_force-median"
        ),
        pytest.param((13.0, 43.0, 24.0), None, 3, 39.0, id="big-scaled-min_force-min"),
        pytest.param((13.0, 43.0, 24.0), 1.5, 3.0, 13.0 * 1.5, id="all-inputs"),
    ),
)
def test_get_ymax_default(costs, min_scale_factor, min_v_median_scale_factor, exp):
    call_kwargs = {}

    for name, para in (
        ("min_scale_factor", min_scale_factor),
        ("min_v_median_scale_factor", min_v_median_scale_factor),
    ):
        if para is not None:
            call_kwargs[name] = para

    res = get_ymax_default(costs, **call_kwargs)

    assert res == exp


@pytest.mark.parametrize(
    "alpha, alpha_exp, extra_kwargs, extra_kwargs_exp",
    ((0.8, 0.8, None, {}), (None, 0.7, {"lw": 3}, {"lw": 3})),
)
def test_plot_parameters(alpha, alpha_exp, extra_kwargs, extra_kwargs_exp):
    axes = {
        "a": Mock(),
        "b": Mock(),
    }
    para_vals = {
        "a": [1, 2, 3],
        "b": [3, 6, -3],
    }

    call_kwargs = {}

    for name, para in (("alpha", alpha),):
        if para is not None:
            call_kwargs[name] = para

    if extra_kwargs is not None:
        call_kwargs = call_kwargs | extra_kwargs

    plot_parameters(
        axes,
        para_vals,
        **call_kwargs,
    )

    for para, vals in para_vals.items():
        axes[para].scatter.assert_called_once_with(
            range(len(vals)), vals, alpha=alpha_exp, **extra_kwargs_exp
        )
        axes[para].set_ylabel.assert_called_once_with(para)


@pytest.mark.parametrize("exp_warn", (True, False))
@pytest.mark.parametrize("others_empty", (True, False))
@pytest.mark.parametrize(
    "background_ts_kwargs, background_ts_kwargs_exp, "
    "target_ts_kwargs, target_ts_kwargs_exp, "
    "best_ts_kwargs, best_ts_kwargs_exp, "
    "ylabel_kwargs, ylabel_kwargs_exp",
    (
        (
            None,
            DEFAULT_PLOT_TIMESERIES_BACKGROUND_TS_KWARGS,
            None,
            DEFAULT_PLOT_TIMESERIES_TARGET_TS_KWARGS,
            {"zorder": 3},
            {"zorder": 3},
            {"color": "k"},
            {"color": "k"},
        ),
        (
            {"lw": 3},
            {"lw": 3},
            {"alpha": 0.3},
            {"alpha": 0.3},
            None,
            DEFAULT_PLOT_TIMESERIES_BEST_TS_KWARGS,
            None,
            {},
        ),
    ),
)
def test_plot_timeseries(
    exp_warn,
    others_empty,
    background_ts_kwargs,
    background_ts_kwargs_exp,
    target_ts_kwargs,
    target_ts_kwargs_exp,
    best_ts_kwargs,
    best_ts_kwargs_exp,
    ylabel_kwargs,
    ylabel_kwargs_exp,
    recwarn,
):
    # Given how complicated this test is, there is a pretty strong argument to
    # say that plot_timeseries does too much and should be refactored. PRs
    # welcome :)

    def convert_scmrun_to_plot_dict(inputs):
        return inputs

    timeseries_keys = ("a", "b")
    axes = {"a": Mock(), "b": Mock()}
    mock_get_timeseries_return = Mock()
    get_timeseries = Mock(return_value=mock_get_timeseries_return)

    # Setup
    def _get_scmrun_mock(name, unit, empty=False):
        out = Mock()
        out.name = name
        out.empty = empty
        out.convert_unit = Mock(return_value=f"{id(out)}_converted")
        out.get_unique_meta = Mock(return_value=unit)

        return out

    unit = "K"
    best_run = {
        "a": _get_scmrun_mock("a_best", unit),
        "b": _get_scmrun_mock("b_best", unit),
    }
    others_to_plot = {
        "a": _get_scmrun_mock("a_other", unit, empty=others_empty),
        "b": _get_scmrun_mock("b_other", unit, empty=others_empty),
    }
    if exp_warn:
        target_unit = "mK"
    else:
        target_unit = unit

    target = {
        "a": _get_scmrun_mock("a_target", target_unit),
        "b": _get_scmrun_mock("b_target", target_unit),
    }

    call_kwargs = {}

    for name, para in (
        ("background_ts_kwargs", background_ts_kwargs),
        ("target_ts_kwargs", target_ts_kwargs),
        ("best_ts_kwargs", best_ts_kwargs),
        ("ylabel_kwargs", ylabel_kwargs),
    ):
        if para is not None:
            call_kwargs[name] = para

    # Run
    plot_timeseries(
        best_run,
        others_to_plot,
        target,
        convert_scmrun_to_plot_dict,
        timeseries_keys,
        axes,
        get_timeseries,
        **call_kwargs,
    )

    if exp_warn:
        assert len(recwarn) == 1
        warn = recwarn.pop(UserWarning)
        exp_warn_msg = (
            f"Converting target units ('{target_unit}') to model output "
            f"units ('{unit}'), this will happen every time you "
            "plot and is slow. Please convert the target units to the "
            "model's units before doing the optimisation for increased "
            "performance (the function "
            "``convert_target_to_model_output_units`` may be helpful)."
        )
        assert str(warn.message) == exp_warn_msg

    get_timeseries_exp_calls = []
    get_timeseries_return_exp_calls = []
    for k in timeseries_keys:
        if not others_empty:
            get_timeseries_exp_calls.append(
                call(convert_scmrun_to_plot_dict(others_to_plot)[k])
            )
            get_timeseries_return_exp_calls.append(
                call(ax=axes[k], **background_ts_kwargs_exp)
            )

        if exp_warn:
            get_timeseries_exp_calls.append(
                call(f"{id(convert_scmrun_to_plot_dict(target)[k])}_converted")
            )

        else:
            get_timeseries_exp_calls.append(
                call(convert_scmrun_to_plot_dict(target)[k])
            )
        get_timeseries_return_exp_calls.append(call(ax=axes[k], **target_ts_kwargs_exp))

        get_timeseries_exp_calls.append(call(convert_scmrun_to_plot_dict(best_run)[k]))
        get_timeseries_return_exp_calls.append(call(ax=axes[k], **best_ts_kwargs_exp))

        axes[k].set_ylabel.assert_called_once_with(k, **ylabel_kwargs_exp)

    get_timeseries.assert_has_calls(get_timeseries_exp_calls)
    mock_get_timeseries_return.plot.line.assert_has_calls(
        get_timeseries_return_exp_calls
    )


@pytest.mark.parametrize(
    "costs, res, thin_ts_to_plot, exp_best, exp_others",
    (
        pytest.param(
            [3, 5, 1, 8], ["a", "b", "c", "d"], 1, "c", ["d", "b", "a"], id="simple"
        ),
        pytest.param([3, 5, 1, 8], ["a", "b", "c", "d"], 2, "c", ["d", "b"], id="thin"),
        pytest.param(
            [3, np.inf, 1, 8],
            ["a", None, "c", "d"],
            1,
            "c",
            ["d", "a"],
            id="including-failed-run",
        ),
        pytest.param(
            [3, np.inf, 1, 8],
            ["a", None, "c", "d"],
            3,
            "c",
            ["d"],
            id="thin-including-failed-run",
        ),
    ),
)
@patch("openscm_calibration.scipy_plotting.scmdata")
def test_get_runs_to_plot(
    mock_scmdata, costs, res, thin_ts_to_plot, exp_best, exp_others
):
    mock_scmdata.run_append = Mock(return_value="m".join(exp_others))

    res = get_runs_to_plot(
        costs,
        res,
        thin_ts_to_plot,
    )

    mock_scmdata.run_append.assert_called_once_with(exp_others)

    assert res[0] == exp_best
    assert res[1] == mock_scmdata.run_append.return_value


def test_get_runs_to_plot_no_success():
    error_msg = re.escape("No successful runs, please check")
    with pytest.raises(ValueError, match=error_msg):
        get_runs_to_plot(
            [np.inf, np.inf],
            [None, None],
            3,
        )


@pytest.mark.parametrize("time_to_call_real_callback_return_value", (True, False))
@patch("openscm_calibration.scipy_plotting.CallbackProxy.time_to_call_real_callback")
def test_callback_proxy_callback_minimize(
    mock_time_to_call_real_callback, time_to_call_real_callback_return_value
):
    mock_time_to_call_real_callback.return_value = (
        time_to_call_real_callback_return_value
    )

    xk = [1, 3, 4]

    real_callback = Mock()
    proxy = CallbackProxy(
        real_callback,
        Mock(),
    )
    proxy.callback_minimize(xk)

    if time_to_call_real_callback_return_value:
        real_callback.callback_minimize.assert_called_once_with(xk)
    else:
        real_callback.callback_minimize.assert_not_called()


@pytest.mark.parametrize("time_to_call_real_callback_return_value", (True, False))
@patch("openscm_calibration.scipy_plotting.CallbackProxy.time_to_call_real_callback")
def test_callback_proxy_callback_differential_evolution(
    mock_time_to_call_real_callback, time_to_call_real_callback_return_value
):
    mock_time_to_call_real_callback.return_value = (
        time_to_call_real_callback_return_value
    )

    xk = [1, 3, 4]
    convergence = 0.4

    real_callback = Mock()
    proxy = CallbackProxy(
        real_callback,
        Mock(),
    )
    proxy.callback_differential_evolution(xk, convergence)

    if time_to_call_real_callback_return_value:
        real_callback.callback_differential_evolution.assert_called_once_with(
            xk, convergence
        )
    else:
        real_callback.callback_differential_evolution.assert_not_called()


@pytest.mark.parametrize(
    "last_callback_val, last_callback_val_exp, "
    "update_every, update_every_exp, x_samples, exp_n_calls, exp",
    (
        pytest.param(
            None,
            0,
            None,
            50,
            [[1, 3], [3, 1], None, None],
            2,
            False,
            id="defaults-not-called",
        ),
        pytest.param(None, 0, None, 50, ["a"] * 100, 100, True, id="defaults-called"),
        pytest.param(
            2, 2, 3, 3, [[1, 3], [3, 1], None, None], 2, False, id="basic-not-called"
        ),
        pytest.param(
            -1, -1, 2, 2, [[1, 3], [3, 1], None, [3, 3]], 3, True, id="basic-called"
        ),
    ),
)
@pytest.mark.parametrize(
    "progress_bar, progress_bar_exp",
    (
        (None, None),
        ("mock", "mock"),
    ),
)
@patch("openscm_calibration.scipy_plotting.CallbackProxy.update_progress_bar")
def test_time_to_call_real_callback(
    mock_update_progress_bar,
    last_callback_val,
    last_callback_val_exp,
    update_every,
    update_every_exp,
    x_samples,
    exp_n_calls,
    exp,
    progress_bar,
    progress_bar_exp,
):
    if progress_bar == "mock":
        progress_bar = Mock()
        progress_bar_exp = progress_bar

    store = Mock()
    store.x_samples = x_samples

    init_kwargs = {}

    for name, para in (
        ("last_callback_val", last_callback_val),
        ("update_every", update_every),
        ("progress_bar", progress_bar),
    ):
        if para is not None:
            init_kwargs[name] = para

    proxy = CallbackProxy(
        real_callback=Mock(),
        store=store,
        **init_kwargs,
    )

    assert proxy.last_callback_val == last_callback_val_exp
    assert proxy.update_every == update_every_exp
    assert proxy.progress_bar == progress_bar_exp

    res = proxy.time_to_call_real_callback()

    if progress_bar_exp:
        mock_update_progress_bar.assert_called_once_with(exp_n_calls)
    else:
        mock_update_progress_bar.assert_not_called()

    if exp:
        assert proxy.last_callback_val == exp_n_calls

    assert res == exp


@pytest.mark.parametrize("n_calls", (5, 10))
@pytest.mark.parametrize("last_print_n", (1, 3))
def test_progress_bar(n_calls, last_print_n):
    progress_bar = Mock()
    progress_bar.last_print_n = last_print_n

    proxy = CallbackProxy(
        Mock(),
        Mock(),
        progress_bar=progress_bar,
    )

    proxy.update_progress_bar(n_calls)

    progress_bar.update.assert_called_once_with(n_calls - progress_bar.last_print_n)


def test_no_progress_bar():
    proxy = CallbackProxy(
        Mock(),
        Mock(),
    )

    error_msg = re.escape("``self.progress_bar`` is not set")
    with pytest.raises(ValueError, match=error_msg):
        proxy.update_progress_bar(3)


def test_convert_target_to_model_output_units():
    years = [2010, 2020, 2030]

    index_target = pd.MultiIndex.from_tuples(
        [(name, unit) for name, unit in (("dT", "K"), ("OHU", "J / yr"))],
        names=("variable", "unit"),
    )
    df_target = pd.DataFrame(
        np.random.random(size=(index_target.size, len(years))),
        columns=years,
        index=index_target,
    )
    target = scmdata.run.BaseScmRun(df_target)

    index_sample = pd.MultiIndex.from_tuples(
        [(name, unit) for name, unit in (("dT", "mK"), ("OHU", "ZJ / yr"))],
        names=("variable", "unit"),
    )
    df_sample = pd.DataFrame(
        np.random.random(size=(index_sample.size, len(years))),
        columns=years,
        index=index_sample,
    )
    sample = scmdata.run.BaseScmRun(df_sample)

    convert_scmrun_to_plot_dict = partial(scmrun_as_dict, groups=("variable",))

    res = convert_target_to_model_output_units(
        target=target,
        model_output=sample,
        convert_scmrun_to_plot_dict=convert_scmrun_to_plot_dict,
    )

    exp = target.convert_unit("mK", variable="dT").convert_unit(
        "ZJ / yr", variable="OHU"
    )

    scmdata.testing.assert_scmdf_almost_equal(
        res, exp, allow_unordered=True, check_ts_names=False
    )


@pytest.mark.parametrize(
    ",".join(
        [
            "cost_key",
            "parameters",
            "timeseries",
            "cost_col_relwidth",
            "n_parameters_per_row",
            "n_timeseries_per_row",
            "exp",
        ]
    ),
    (
        pytest.param(
            "cost",
            ["a", "b"],
            ["c", "d"],
            None,
            None,
            None,
            [
                ["cost", "a"],
                ["cost", "b"],
                ["c", "c"],
                ["d", "d"],
            ],
            id="defaults",
        ),
        pytest.param(
            "cost",
            ["a", "b"],
            ["c", "d"],
            None,
            2,
            None,
            [
                ["cost", "a", "b"],
                ["c", "c", "c"],
                ["d", "d", "d"],
            ],
            id="2-params-per-row",
        ),
        pytest.param(
            "cost",
            ["a", "b"],
            ["c", "d"],
            2,
            None,
            2,
            [
                ["cost", "cost", "cost", "cost", "a", "a"],
                ["cost", "cost", "cost", "cost", "b", "b"],
                ["c", "c", "c", "d", "d", "d"],
            ],
            id="double-cost-width-2-timeseries-per-row",
        ),
        pytest.param(
            "cost",
            ["a", "b"],
            ["c", "d"],
            2,
            2,
            2,
            [
                list(more_itertools.repeat_each(["cost", "cost", "a", "b"], 2)),
                list(more_itertools.repeat_each(["c", "c", "d", "d"], 2)),
            ],
            id="double-cost-width-2-paras-per-row-2-timeseries-per-row",
        ),
    ),
)
def test_get_optimisation_mosaic(
    cost_key,
    parameters,
    timeseries,
    cost_col_relwidth,
    n_parameters_per_row,
    n_timeseries_per_row,
    exp,
):
    call_kwargs = {}

    for name, para in (
        ("cost_col_relwidth", cost_col_relwidth),
        ("n_parameters_per_row", n_parameters_per_row),
        ("n_timeseries_per_row", n_timeseries_per_row),
    ):
        if para is not None:
            call_kwargs[name] = para

    res = get_optimisation_mosaic(cost_key, parameters, timeseries, **call_kwargs)

    assert res == exp
