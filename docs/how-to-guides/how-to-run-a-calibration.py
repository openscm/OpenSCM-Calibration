# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calibration demo
#
# Here we give a basic demo of how to run a calibration with OpenSCM Calibration.
#
# ## Imports

# %%
from functools import partial

import emcee
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import pandas as pd
import pint
import scipy.integrate
import scmdata.run
from emcwrap import DIMEMove
from multiprocess import Manager, Pool
from openscm_units import unit_registry as UREG
from tqdm.notebook import tqdm

from openscm_calibration import emcee_plotting
from openscm_calibration.cost.scmdata import OptCostCalculatorSSE
from openscm_calibration.emcee_utils import (
    get_acceptance_fractions,
    get_autocorrelation_info,
)
from openscm_calibration.minimize import to_minimize_full
from openscm_calibration.model_runner import OptModelRunner
from openscm_calibration.scipy_plotting import (
    CallbackProxy,
    OptPlotter,
    get_ymax_default,
    plot_costs,
)
from openscm_calibration.scipy_plotting.scmdata import (
    get_timeseries_scmrun,
    plot_timeseries_scmrun,
)
from openscm_calibration.scmdata_utils import scmrun_as_dict
from openscm_calibration.store import OptResStore
from openscm_calibration.store.scmdata import add_iteration_to_res_scmrun

# %%
# Set the seed to ensure reproducibility
seed = 424242
np.random.seed(seed)  # noqa: NPY002 # want to set global seed for emcee
RNG = np.random.default_rng(seed=seed)

# %% [markdown]
# ## Background
#
# In this notebook we're going to run a simple model
# to solve the following equation for the motion of a mass on a damped spring
#
# \begin{align*}
# v &= \frac{dx}{dt} \\
# m \frac{dv}{dt} &= -k (x - x_0) - \beta v
# \end{align*}
#
# where $v$ is the velocity of the mass,
# $x$ is the position of the mass,
# $t$ is time,
# $m$ is the mass of the mass,
# $k$ is the spring constant,
# $x_0$ is the equilibrium position
# and $\beta$ is a damping constant.

# %% [markdown]
# We are going to solve this system using
# [scipy's solve initial value problem](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
#
# We want to support units,
# so we implement this using [pint](https://pint.readthedocs.io/).
# To make this work, we also have to define some wrappers.
# We define these in this notebook to show you the full details.
# If you find them distracting,
# please [raise an issue](https://github.com/openscm/OpenSCM-Calibration/issues/new)
# or submit a pull request.

# %% [markdown]
# ## Experiments
#
# We're going to calibrate the model's response in two experiments:
#
# - starting out of equilibrium
# - starting at the equilibrium position but already moving
#
# We're going to fix the mass of the spring
# because the system is underconstrained if it isn't fixed.

# %%
LENGTH_UNITS = "m"
MASS_UNITS = "Pt"
TIME_UNITS = "yr"
time_axis = UREG.Quantity(np.arange(1850, 2000, 1), TIME_UNITS)
mass = UREG.Quantity(100, MASS_UNITS)


# %%
def do_experiments(
    k: pint.Quantity, x_zero: pint.Quantity, beta: pint.Quantity
) -> scmdata.run.BaseScmRun:
    """
    Run model experiments

    Parameters
    ----------
    k
        Spring constant [kg / s^2]

    x_zero
        Equilibrium position [m]

    beta
        Damping constant [kg / s]

    Returns
    -------
        Results
    """
    # Avoiding pint conversions in the function actually
    # being solved makes things much faster, but also
    # harder to keep track of. We recommend starting with
    # pint everywhere first, then optimising second (either
    # via using numpy quantities throughout, optional numpy
    # quantities using e.g. pint.wrap or using Fortran or
    # C wrappers).
    k_m = k.to(f"{MASS_UNITS} / {TIME_UNITS}^2").m
    x_zero_m = x_zero.to(LENGTH_UNITS).m
    beta_m = beta.to(f"{MASS_UNITS} / {TIME_UNITS}").m
    m_m = mass.to(MASS_UNITS).m

    def to_solve(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of our equation i.e. dN/dt

        Parameters
        ----------
        t
            time

        y
            Current stat of the system

        Returns
        -------
            dN/dt
        """
        x = y[0]
        v = y[1]

        dv_dt = (-k_m * (x - x_zero_m) - beta_m * v) / m_m
        dx_dt = v

        out = np.array([dx_dt, dv_dt])

        return out

    res = {}
    time_axis_m = time_axis.to(TIME_UNITS).m
    for name, to_solve_l, x_start, v_start in (
        (
            "non-eqm-start",
            to_solve,
            UREG.Quantity(1.2, "m"),
            UREG.Quantity(0, "m / s"),
        ),
        ("eqm-start", to_solve, x_zero, UREG.Quantity(0.3, "m / yr")),
    ):
        res[name] = scipy.integrate.solve_ivp(
            to_solve_l,
            t_span=[time_axis_m[0], time_axis_m[-1]],
            y0=[
                x_start.to(LENGTH_UNITS).m,
                v_start.to(f"{LENGTH_UNITS} / {TIME_UNITS}").m,
            ],
            t_eval=time_axis_m,
        )
        if not res[name].success:
            msg = "Model failed to solve"
            raise ValueError(msg)

    out = scmdata.run.BaseScmRun(
        pd.DataFrame(
            np.vstack([res["non-eqm-start"].y[0, :], res["eqm-start"].y[0, :]]),
            index=pd.MultiIndex.from_arrays(
                (
                    ["position", "position"],
                    [LENGTH_UNITS, LENGTH_UNITS],
                    ["non-eqm-start", "eqm-start"],
                ),
                names=["variable", "unit", "scenario"],
            ),
            columns=time_axis.to(TIME_UNITS).m,
        )
    )
    out["model"] = "example"

    return out


# %% [markdown]
# ### Target
#
# For this example, we're going to use a known configuration as our target
# so we can make sure that we optimise to the right spot.
# In practice, we won't know the correct answer before we start
# so this setup will generally look a bit different
# (typically we would be loading data
# from some other source to which we want to calibrate).

# %%
truth = {
    "k": UREG.Quantity(3000, "kg / s^2"),
    "x_zero": UREG.Quantity(0.5, "m"),
    "beta": UREG.Quantity(1e11, "kg / s"),
}

target = do_experiments(**truth)
target["model"] = "target"
target.lineplot(time_axis="year-month")
target

# %% [markdown]
# ### Cost calculation
#
# The next thing is to decide how we're going to calculate the cost function.
# There are many options here,
# in this case we're going to use the sum of squared errors.

# %%
normalisation = pd.Series(
    [0.1],
    index=pd.MultiIndex.from_arrays(
        (
            [
                "position",
            ],
            ["m"],
        ),
        names=["variable", "unit"],
    ),
)

cost_calculator = OptCostCalculatorSSE.from_series_normalisation(
    target=target, normalisation_series=normalisation, model_col="model"
)
assert cost_calculator.calculate_cost(target) == 0
assert cost_calculator.calculate_cost(target * 1.1) > 0
cost_calculator

# %% [markdown]
# ### Model runner
#
# Scipy does everything using numpy arrays.
# Here we use a wrapper that converts them to pint quantities before running.

# %% [markdown]
# Firstly, we define the parameters we're going to optimise.
# This will be used to ensure a consistent order throughout.

# %%
parameters = [
    ("k", f"{MASS_UNITS} / {TIME_UNITS} ^ 2"),
    ("x_zero", LENGTH_UNITS),
    ("beta", f"{MASS_UNITS} / {TIME_UNITS}"),
]
parameters


# %% [markdown]
# Next we define a function which, given pint quantities,
# returns the inputs needed for our `do_experiments` function.
# In this case this is not a very interesting function,
# but in other use cases the flexibility is helpful.


# %%
def do_model_runs_input_generator(
    k: pint.Quantity, x_zero: pint.Quantity, beta: pint.Quantity
) -> dict[str, pint.Quantity]:
    """
    Create the inputs for :func:`do_experiments`

    Parameters
    ----------
    k
        k

    x_zero
        x_zero

    beta
        beta

    Returns
    -------
        Inputs for :func:`do_experiments`
    """
    return {"k": k, "x_zero": x_zero, "beta": beta}


# %%
model_runner = OptModelRunner.from_parameters(
    params=parameters,
    do_model_runs_input_generator=do_model_runs_input_generator,
    do_model_runs=do_experiments,
)
model_runner

# %% [markdown]
# Now we can run from a plain numpy array (like scipy will use)
# and get a result that will be understood by our cost calculator.

# %%
cost_calculator.calculate_cost(model_runner.run_model([3, 0.5, 3]))

# %% [markdown]
# Now we're ready to optimise.

# %% [markdown]
# ## Global optimisation
#
# Scipy has many [global optimisation options](https://docs.scipy.org/doc/scipy/reference/optimize.html#global-optimization).
# Here we show how to do this with differential evolution,
# but using others would be equally simple.

# %% [markdown]
# We have to define where to start the optimisation.

# %%
start = np.array([4, 0.6, 2])
start

# %% [markdown]
# For this optimisation, we must also define bounds for each parameter.

# %%
bounds_dict = {
    "k": [
        UREG.Quantity(300, "kg / s^2"),
        UREG.Quantity(1e4, "kg / s^2"),
    ],
    "x_zero": [
        UREG.Quantity(-2, "m"),
        UREG.Quantity(2, "m"),
    ],
    "beta": [
        UREG.Quantity(1e10, "kg / s"),
        UREG.Quantity(1e12, "kg / s"),
    ],
}
bounds_dict

# %%
bounds = [[v.to(unit).m for v in bounds_dict[k]] for k, unit in parameters]
bounds

# %% [markdown]
# Now we're ready to run our optimisation.

# %%
# Number of parallel processes to use
processes = 4

# Random seed (use if you want reproducibility)
seed = 12849

## Optimisation parameters - here we use short runs
## TODO: other repo with full runs
# Tolerance to set for convergance
atol = 1
tol = 0.02
# Maximum number of iterations to use
maxiter = 8
# Lower mutation means faster convergence but smaller
# search radius
mutation = (0.1, 0.8)
# Higher recombination means faster convergence but
# might miss global minimum
recombination = 0.8
# Size of population to use (higher number means more searching
# but slower convergence)
popsize = 4
# There are also the strategy and init options
# which might be needed for some problems

# Maximum number of runs to store
max_n_runs = (maxiter + 1) * popsize * len(parameters)


# Visualisation options
update_every = 4
thin_ts_to_plot = 5


# Create axes to plot on (could also be created as part of a factory
# or class method)
convert_scmrun_to_plot_dict = partial(scmrun_as_dict, groups=["variable", "scenario"])

cost_name = "cost"
timeseries_axes = list(convert_scmrun_to_plot_dict(target).keys())

parameters_names = [v[0] for v in parameters]
parameters_mosiac = list(more_itertools.repeat_each(parameters_names, 1))
timeseries_axes_mosiac = list(more_itertools.repeat_each(timeseries_axes, 1))

fig, axd = plt.subplot_mosaic(
    mosaic=[
        [cost_name, *timeseries_axes_mosiac],
        parameters_mosiac,
    ],
    figsize=(6, 6),
)
holder = display(fig, display_id=True)  # noqa: F821 # used in a notebook


with Manager() as manager:
    store = OptResStore.from_n_runs_manager(
        max_n_runs,
        manager,
        params=parameters_names,
        add_iteration_to_res=add_iteration_to_res_scmrun,
    )

    # Create objects and functions to use
    to_minimize = partial(
        to_minimize_full,
        store=store,
        cost_calculator=cost_calculator,
        model_runner=model_runner,
        known_error=ValueError,
    )

    with manager.Pool(processes=processes) as pool:
        with tqdm(total=max_n_runs) as pbar:
            opt_plotter = OptPlotter(
                holder=holder,
                fig=fig,
                axes=axd,
                cost_key=cost_name,
                parameters=parameters_names,
                timeseries_axes=timeseries_axes,
                target=target,
                store=store,
                get_timeseries=get_timeseries_scmrun,
                plot_timeseries=plot_timeseries_scmrun,
                convert_results_to_plot_dict=convert_scmrun_to_plot_dict,
                thin_ts_to_plot=thin_ts_to_plot,
            )

            proxy = CallbackProxy(
                real_callback=opt_plotter,
                store=store,
                update_every=update_every,
                progress_bar=pbar,
                last_callback_val=0,
            )

            # This could be wrapped up too
            optimize_res = scipy.optimize.differential_evolution(
                to_minimize,
                bounds,
                maxiter=maxiter,
                x0=start,
                tol=tol,
                atol=atol,
                seed=seed,
                # Polish as a second step if you want
                polish=False,
                workers=pool.map,
                updating="deferred",  # as we run in parallel, this has to be used
                mutation=mutation,
                recombination=recombination,
                popsize=popsize,
                callback=proxy.callback_differential_evolution,
            )

plt.close()
optimize_res

# %% [markdown]
# ## Local optimisation
#
# Scipy also has
# [local optimisation](https://docs.scipy.org/doc/scipy/reference/optimize.html#local-multivariate-optimization)
# (e.g. Nelder-Mead) options. Here we show how to do this.

# %% [markdown]
# Again, we have to define where to start the optimisation
# (this has a greater effect on local optimisation).

# %%
# Here we imagine that we're polishing from the results of the DE above,
# but we make the start slightly worse first
start_local = optimize_res.x
start_local

# %%
# Optimisation parameters
tol = 1e-4
# Maximum number of iterations to use
maxiter = 30

# I think this is how this works
max_n_runs = len(parameters) + 2 * maxiter

# Lots of options here
method = "Nelder-mead"

# Visualisation options
update_every = 10
thin_ts_to_plot = 5
parameters_names = [v[0] for v in parameters]

# Create other objects
store = OptResStore.from_n_runs(
    max_n_runs,
    params=parameters_names,
    add_iteration_to_res=add_iteration_to_res_scmrun,
)
to_minimize = partial(
    to_minimize_full,
    store=store,
    cost_calculator=cost_calculator,
    model_runner=model_runner,
)


with tqdm(total=max_n_runs) as pbar:
    # Here we use a class method which auto-generates the figure
    # for us. This is just a convenience thing, it does the same
    # thing as the previous example under the hood.
    opt_plotter = OptPlotter.from_autogenerated_figure(
        cost_key=cost_name,
        params=parameters_names,
        convert_results_to_plot_dict=convert_scmrun_to_plot_dict,
        target=target,
        store=store,
        get_timeseries=get_timeseries_scmrun,
        plot_timeseries=plot_timeseries_scmrun,
        thin_ts_to_plot=thin_ts_to_plot,
        kwargs_create_mosaic=dict(
            n_parameters_per_row=3,
            n_timeseries_per_row=1,
            cost_col_relwidth=2,
        ),
        kwargs_get_fig_axes_holder=dict(figsize=(10, 6)),
        plot_costs=partial(
            plot_costs,
            alpha=0.7,
            get_ymax=partial(
                get_ymax_default, min_scale_factor=1e6, min_v_median_scale_factor=0
            ),
        ),
    )

    proxy = CallbackProxy(
        real_callback=opt_plotter,
        store=store,
        update_every=update_every,
        progress_bar=pbar,
        last_callback_val=0,
    )

    optimize_res_local = scipy.optimize.minimize(
        to_minimize,
        x0=start_local,
        tol=tol,
        method=method,
        options={"maxiter": maxiter},
        callback=proxy.callback_minimize,
    )

plt.close()
optimize_res_local


# %% [markdown]
# ## MCMC
#
# To run MCMC, we use the [emcee](https://emcee.readthedocs.io/) package.
# This has heaps of options for running MCMC and is really user friendly.
# All the different available moves/samplers are listed [here](https://emcee.readthedocs.io/en/stable/user/moves/).


# %%
def neg_log_prior_bounds(x: np.ndarray, bounds: np.ndarray) -> float:
    """
    Log prior that just checks proposal is in bounds

    Parameters
    ----------
    x
        Parameter array

    bounds
        Bounds for each parameter (must have same
        order as x)
    """
    in_bounds = (x > bounds[:, 0]) & (x < bounds[:, 1])
    if np.all(in_bounds):
        return 0

    return -np.inf


neg_log_prior = partial(neg_log_prior_bounds, bounds=np.array(bounds))


# %%
def log_prob(x) -> tuple[float, float, float]:
    """
    Get log probability for a given parameter vector

    Returns (negative) log probability of x,
    (negative) log likelihood of x based on the prior
    and (negative) log likelihood of x.

    """
    neg_ll_prior_x = neg_log_prior(x)

    if not np.isfinite(neg_ll_prior_x):
        return -np.inf, None, None

    try:
        model_results = model_runner.run_model(x)
    except ValueError:
        return -np.inf, None, None

    sses = cost_calculator.calculate_cost(model_results)
    neg_ll_x = -sses / 2
    neg_log_prob = neg_ll_x + neg_ll_prior_x

    return neg_log_prob, neg_ll_prior_x, neg_ll_x


# %% [markdown]
# We're using the DIME proposal from [emcwrap](https://github.com/gboehl/emcwrap).
# This claims to have an adaptive proposal distribution
# so requires less fine tuning and is less sensitive to the starting point.

# %%
ndim = len(bounds)
# emcwrap docs suggest 5 * ndim
nwalkers = 5 * ndim

start_emcee = [s + s / 100 * RNG.random(nwalkers) for s in optimize_res_local.x]
start_emcee = np.vstack(start_emcee).T

move = DIMEMove()

# %%
# Use HDF5 backend
filename = "basic-demo-mcmc.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# %%
# How many parallel process to use
processes = 4

## MCMC options
# Unclear at the start how many iterations are needed to sample
# the posterior appropriately, normally requires looking at the
# chains and then just running them for longer if needed.
# This number is definitely too small
max_iterations = 60
burnin = 10
thin = 2

## Visualisation options
plot_every = 15
convergence_ratio = 50
parameter_order = [p[0] for p in parameters]
neg_log_likelihood_name = "neg_ll"
labels_chain = [neg_log_likelihood_name, *parameter_order]

# Stores for autocorr over steps
autocorr = np.zeros(max_iterations)
autocorr_steps = np.zeros(max_iterations)
index = 0

## Setup plots
fig_chain, axd_chain = plt.subplot_mosaic(
    mosaic=[[lc] for lc in labels_chain],
    figsize=(10, 5),
)
holder_chain = display(fig_chain, display_id=True)  # noqa: F821 # used in a notebook

fig_dist, axd_dist = plt.subplot_mosaic(
    mosaic=[[parameter] for parameter in parameter_order],
    figsize=(10, 5),
)
holder_dist = display(fig_dist, display_id=True)  # noqa: F821 # used in a notebook

fig_corner = plt.figure(figsize=(6, 6))
holder_corner = display(fig_dist, display_id=True)  # noqa: F821 # used in a notebook

fig_tau, ax_tau = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(4, 4),
)
holder_tau = display(fig_tau, display_id=True)  # noqa: F821 # used in a notebook

# Plottting helper
truths_corner = [truth[k].to(u).m for k, u in parameters]

with Pool(processes=processes) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob,
        moves=move,
        backend=backend,
        blobs_dtype=[("neg_log_prior", float), ("neg_log_likelihood", float)],
        pool=pool,
    )

    for sample in sampler.sample(
        # If statement in case we're continuing a run rather than starting fresh
        start_emcee if sampler.iteration < 1 else sampler.get_last_sample(),
        iterations=max_iterations,
        progress="notebook",
        progress_kwargs={"leave": True},
    ):
        min_samples_before_plot = 2
        if (
            sampler.iteration % plot_every
            or sampler.iteration < min_samples_before_plot
        ):
            continue

        if sampler.iteration < burnin + 1:
            in_burn_in = True
        else:
            in_burn_in = False

        for ax in axd_chain.values():
            ax.clear()

        emcee_plotting.plot_chains(
            inp=sampler,
            burnin=burnin,
            parameter_order=parameter_order,
            axes_d=axd_chain,
            neg_log_likelihood_name=neg_log_likelihood_name,
        )
        fig_chain.tight_layout()
        holder_chain.update(fig_chain)

        if not in_burn_in:
            chain_post_burnin = sampler.get_chain(discard=burnin)
            if chain_post_burnin.shape[0] > 0:
                acceptance_fraction = np.mean(
                    get_acceptance_fractions(chain_post_burnin)
                )
                print(
                    f"{chain_post_burnin.shape[0]} steps post burnin, "
                    f"acceptance fraction: {acceptance_fraction}"
                )

            for ax in axd_dist.values():
                ax.clear()

            emcee_plotting.plot_dist(
                inp=sampler,
                burnin=burnin,
                thin=thin,
                parameter_order=parameter_order,
                axes_d=axd_dist,
                warn_singular=False,
            )
            fig_dist.tight_layout()
            holder_dist.update(fig_dist)

            try:
                fig_corner.clear()
                emcee_plotting.plot_corner(
                    inp=sampler,
                    burnin=burnin,
                    thin=thin,
                    parameter_order=parameter_order,
                    fig=fig_corner,
                    truths=truths_corner,
                )
                fig_corner.tight_layout()
                holder_corner.update(fig_corner)
            except AssertionError:
                pass

            autocorr_bits = get_autocorrelation_info(
                sampler,
                burnin=burnin,
                thin=thin,
                autocorr_tol=0,
                convergence_ratio=convergence_ratio,
            )
            autocorr[index] = autocorr_bits["autocorr"]
            autocorr_steps[index] = sampler.iteration - burnin
            index += 1

            if np.sum(autocorr > 0) > 1 and np.sum(~np.isnan(autocorr)) > 1:
                # plot autocorrelation, pretty specific to setup so haven't
                # created separate function
                ax_tau.clear()
                ax_tau.plot(
                    autocorr_steps[:index],
                    autocorr[:index],
                )
                ax_tau.axline(
                    (0, 0), slope=1 / convergence_ratio, color="k", linestyle="--"
                )
                ax_tau.set_ylabel("Mean tau")
                ax_tau.set_xlabel("Number steps (post burnin)")
                holder_tau.update(fig_tau)

# Close all the figures
for _ in range(4):
    plt.close()
