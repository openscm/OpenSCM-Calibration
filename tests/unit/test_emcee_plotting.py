"""
Tests of :mod:`openscm_calibration.emcee_plotting`
"""
import re
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openscm_calibration.emcee_plotting import (
    get_neg_log_likelihood_ylim_default,
    plot_chains,
    plot_corner,
    plot_dist,
    plot_parameter_chains,
)


@pytest.mark.parametrize(
    "get_neg_log_likelihood_ylim", (None, Mock(return_value=[-1, -3]))
)
@pytest.mark.parametrize("burnin", (10, 3))
@pytest.mark.parametrize("parameter_order", (["para_ab", "parabcd"], ["x", "y"]))
@pytest.mark.parametrize("neg_log_likelihood_name", ("neg_ll", "neg_log_likelihood"))
@pytest.mark.parametrize(
    "extra_kwargs, extra_kwargs_exp",
    (
        (None, {}),
        ({"linewidth": 3}, {"linewidth": 3}),
    ),
)
@patch("openscm_calibration.emcee_plotting.plot_parameter_chains")
@patch("openscm_calibration.emcee_plotting.get_labelled_chain_data")
@patch("openscm_calibration.emcee_plotting.get_neg_log_likelihood_ylim_default")
def test_plot_chains(
    mock_get_neg_log_likelihood_ylim_default,
    mock_get_labelled_chain_data,
    mock_plot_parameter_chains,
    get_neg_log_likelihood_ylim,
    burnin,
    parameter_order,
    neg_log_likelihood_name,
    extra_kwargs,
    extra_kwargs_exp,
):
    inp = "passed_to_mock"

    labels = [*parameter_order, neg_log_likelihood_name]
    axes_d = {para: Mock() for para in labels}
    mock_get_labelled_chain_data.return_value = {
        para: i for i, para in enumerate(labels)
    }

    call_kwargs = {}
    if get_neg_log_likelihood_ylim is not None:
        call_kwargs["get_neg_log_likelihood_ylim"] = get_neg_log_likelihood_ylim
        mock_exp_get_neg_log_likelihood_ylim = get_neg_log_likelihood_ylim

    else:
        # If not passed, should use the default instead
        mock_exp_get_neg_log_likelihood_ylim = mock_get_neg_log_likelihood_ylim_default

    if extra_kwargs is not None:
        call_kwargs = call_kwargs | extra_kwargs

    plot_chains(
        inp,
        burnin,
        parameter_order,
        neg_log_likelihood_name,
        axes_d,
        **call_kwargs,
    )

    mock_get_labelled_chain_data.assert_called_with(
        inp,
        parameter_order,
        neg_log_likelihood_name,
        burnin=0,
        thin=1,
    )

    for label in labels:
        expected_vals = mock_get_labelled_chain_data.return_value[label]

        mock_plot_parameter_chains.assert_any_call(
            axes_d[label],
            expected_vals,
            burnin=burnin,
            **extra_kwargs_exp,
        )
        axes_d[label].set_ylabel.assert_called_with(label)

        if label == neg_log_likelihood_name:
            mock_exp_get_neg_log_likelihood_ylim.assert_called_with(expected_vals)
            axes_d[label].set_ylim.assert_called_with(
                *mock_exp_get_neg_log_likelihood_ylim.return_value
            )


@pytest.mark.parametrize(
    "alpha_chain, alpha_chain_exp",
    (
        (None, 0.3),
        (0.5, 0.5),
    ),
)
@pytest.mark.parametrize(
    "linewidth, linewidth_exp",
    (
        (None, 0.5),
        (1, 1),
    ),
)
@pytest.mark.parametrize(
    "color, color_exp",
    (
        (None, "0.2"),
        ("0.3", "0.3"),
    ),
)
@pytest.mark.parametrize(
    "kwargs_chain, kwargs_chain_exp",
    (
        (None, {}),
        ({"zorder": 3}, {"zorder": 3}),
    ),
)
@pytest.mark.parametrize(
    "alpha_vspan, alpha_vspan_exp",
    (
        (None, 0.3),
        (0.5, 0.5),
    ),
)
@pytest.mark.parametrize(
    "kwargs_vspan, kwargs_vspan_exp",
    (
        (None, {}),
        ({"hatch": "//"}, {"hatch": "//"}),
    ),
)
@pytest.mark.parametrize("burnin", (3, 5))
def test_plot_parameter_chains(
    alpha_chain,
    alpha_chain_exp,
    linewidth,
    linewidth_exp,
    color,
    color_exp,
    kwargs_chain,
    kwargs_chain_exp,
    alpha_vspan,
    alpha_vspan_exp,
    kwargs_vspan,
    kwargs_vspan_exp,
    burnin,
):
    call_kwargs = {}

    for name, para in (
        ("alpha_chain", alpha_chain),
        ("linewidth", linewidth),
        ("color", color),
        ("kwargs_chain", kwargs_chain),
        ("alpha_vspan", alpha_vspan),
        ("kwargs_vspan", kwargs_vspan),
    ):
        if para is not None:
            call_kwargs[name] = para

    ax = Mock()
    vals = Mock()
    vals.shape = [3, 4]

    plot_parameter_chains(
        ax,
        vals,
        burnin,
        **call_kwargs,
    )

    ax.plot.assert_called_with(
        vals,
        lw=linewidth_exp,
        alpha=alpha_chain_exp,
        color=color_exp,
        **kwargs_chain_exp,
    )

    ax.axvspan.assert_called_with(
        0,
        burnin - 0.5,
        alpha=alpha_vspan_exp,
        **kwargs_vspan_exp,
    )

    ax.set_xlim.assert_called_with(
        0,
        vals.shape[0],
    )


@pytest.mark.parametrize(
    "inp, median_scaling, median_scaling_exp, max_scaling, max_scaling_exp, exp",
    (
        pytest.param(
            [-2, -4, -5], None, 1.5, None, 2.0, (-6, 0), id="median-dominates"
        ),
        pytest.param(
            [-2, -4, -5], None, 1.5, 10, 10.0, (-20, 0), id="force-max-dominates"
        ),
        pytest.param([-4, -4.5, -5], None, 1.5, None, 2.0, (-8, 0), id="max-dominates"),
        pytest.param(
            [-4, -4.5, -5], 4, 4, None, 2.0, (-18, 0), id="force-median-dominates"
        ),
        pytest.param(
            [-2, -4, -np.inf, -np.inf, -np.inf],
            None,
            1.5,
            None,
            2.0,
            (-4, 0),
            id="inf-median",
        ),
        pytest.param(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            None,
            1.5,
            None,
            2.0,
            (0, 0),
            id="inf-median-max",
        ),
        pytest.param(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            None,
            1.5,
            None,
            2.0,
            (0, 0),
            id="inf-median-max",
        ),
    ),
)
def test_get_neg_log_likelihood_ylim_default(
    inp, median_scaling, median_scaling_exp, max_scaling, max_scaling_exp, exp
):
    call_kwargs = {}

    for name, para in (
        ("median_scaling", median_scaling),
        ("max_scaling", max_scaling),
    ):
        if para is not None:
            call_kwargs[name] = para

    res = get_neg_log_likelihood_ylim_default(
        inp,
        **call_kwargs,
    )

    assert res == exp


@patch("openscm_calibration.emcee_plotting.HAS_SEABORN", False)
def test_no_seaborn():
    error_msg = re.escape("``plot_dist`` requires seaborn to be installed")
    with pytest.raises(ImportError, match=error_msg):
        plot_dist(
            inp=Mock(),
            burnin=3,
            thin=3,
            parameter_order=["a"],
            axes_d={"a", Mock()},
        )


@pytest.mark.parametrize("parameter_order", (["para_ab", "parabcd"], ["x", "y"]))
@pytest.mark.parametrize("burnin", (0, 10))
@pytest.mark.parametrize("thin", (0, 20))
@pytest.mark.parametrize("common_norm, common_norm_exp", ((None, False), (True, True)))
@pytest.mark.parametrize("fill, fill_exp", ((None, True), (False, False)))
@pytest.mark.parametrize("legend, legend_exp", ((None, False), (True, True)))
@pytest.mark.parametrize(
    "extra_kwargs, extra_kwargs_exp",
    (
        (None, {}),
        ({"linewidth": 3}, {"linewidth": 3}),
    ),
)
@patch("openscm_calibration.emcee_plotting.sns")
@patch("openscm_calibration.emcee_plotting.get_labelled_chain_data")
def test_plot_dist(
    mock_get_labelled_chain_data,
    mock_sns,
    parameter_order,
    burnin,
    thin,
    common_norm,
    common_norm_exp,
    fill,
    fill_exp,
    legend,
    legend_exp,
    extra_kwargs,
    extra_kwargs_exp,
):
    inp = "passed_to_mock"

    axes_d = {para: Mock() for para in parameter_order}
    mock_get_labelled_chain_data.return_value = {
        para: i for i, para in enumerate(parameter_order)
    }

    call_kwargs = {}

    for name, para in (
        ("common_norm", common_norm),
        ("fill", fill),
        ("legend", legend),
    ):
        if para is not None:
            call_kwargs[name] = para

    if extra_kwargs is not None:
        call_kwargs = call_kwargs | extra_kwargs

    plot_dist(
        inp,
        burnin,
        thin,
        parameter_order,
        axes_d,
        **call_kwargs,
    )

    mock_get_labelled_chain_data.assert_called_with(
        inp,
        parameter_order,
        burnin=burnin,
        thin=thin,
    )

    for label in parameter_order:
        expected_vals = mock_get_labelled_chain_data.return_value[label]

        mock_sns.kdeplot.assert_any_call(
            data=expected_vals,
            ax=axes_d[label],
            common_norm=common_norm_exp,
            fill=fill_exp,
            legend=legend_exp,
            **extra_kwargs_exp,
        )
        axes_d[label].set_xlabel.assert_called_with(label)


@patch("openscm_calibration.emcee_plotting.HAS_CORNER", False)
def test_no_corner():
    error_msg = re.escape("``plot_corner`` requires corner to be installed")
    with pytest.raises(ImportError, match=error_msg):
        plot_corner(
            inp=Mock(),
            burnin=3,
            thin=3,
            parameter_order=["a"],
            fig=Mock(),
        )


@pytest.mark.parametrize("burnin", (0, 10))
@pytest.mark.parametrize("thin", (0, 20))
@pytest.mark.parametrize("parameter_order", (["a", "b"], ["de", "f", "gh"]))
@pytest.mark.parametrize(
    "bins, bins_exp, plot_contours, plot_contours_exp",
    ((None, 30, None, True), (15, 15, False, False)),
)
@pytest.mark.parametrize(
    "smooth, smooth_exp, quantiles, quantiles_exp",
    (
        (None, True, None, (0.05, 0.17, 0.5, 0.83, 0.95)),
        (False, False, [0.25, 0.75], [0.25, 0.75]),
    ),
)
@pytest.mark.parametrize(
    "show_titles, show_titles_exp, title_quantiles, title_quantiles_exp",
    ((None, True, [0.25, 0.75], [0.25, 0.75]), (False, False, None, (0.05, 0.5, 0.95))),
)
@pytest.mark.parametrize(
    "title_kwargs, title_kwargs_exp, title_fmt, title_fmt_exp, label_kwargs, label_kwargs_exp",
    (
        (None, {"fontsize": 12}, ".2d", ".2d", None, {"fontsize": "x-small"}),
        (
            {"fontsize": 10},
            {"fontsize": 10},
            None,
            ".3f",
            {"fontsize": "x-large"},
            {"fontsize": "x-large"},
        ),
    ),
)
@pytest.mark.parametrize(
    "extra_kwargs, extra_kwargs_exp",
    (
        (None, {}),
        ({"linewidth": 3}, {"linewidth": 3}),
    ),
)
@patch("openscm_calibration.emcee_plotting.corner")
def test_plot_corner(
    mock_corner,
    burnin,
    thin,
    parameter_order,
    bins,
    bins_exp,
    plot_contours,
    plot_contours_exp,
    smooth,
    smooth_exp,
    quantiles,
    quantiles_exp,
    show_titles,
    show_titles_exp,
    title_quantiles,
    title_quantiles_exp,
    title_kwargs,
    title_kwargs_exp,
    title_fmt,
    title_fmt_exp,
    label_kwargs,
    label_kwargs_exp,
    extra_kwargs,
    extra_kwargs_exp,
):
    inp = Mock()
    inp.get_chain = Mock(return_value=[31, 33])

    fig = Mock()

    call_kwargs = {}

    for name, para in (
        ("bins", bins),
        ("plot_contours", plot_contours),
        ("smooth", smooth),
        ("quantiles", quantiles),
        ("show_titles", show_titles),
        ("title_quantiles", title_quantiles),
        ("title_kwargs", title_kwargs),
        ("title_fmt", title_fmt),
        ("label_kwargs", label_kwargs),
    ):
        if para is not None:
            call_kwargs[name] = para

    if extra_kwargs is not None:
        call_kwargs = call_kwargs | extra_kwargs

    plot_corner(
        inp,
        burnin,
        thin,
        parameter_order,
        fig,
        **call_kwargs,
    )

    mock_corner.corner.assert_called_with(
        inp.get_chain.return_value,
        labels=parameter_order,
        fig=fig,
        bins=bins_exp,
        plot_contours=plot_contours_exp,
        smooth=smooth_exp,
        quantiles=quantiles_exp,
        show_titles=show_titles_exp,
        title_quantiles=title_quantiles_exp,
        title_kwargs=title_kwargs_exp,
        title_fmt=title_fmt_exp,
        label_kwargs=label_kwargs_exp,
        **extra_kwargs_exp,
    )
