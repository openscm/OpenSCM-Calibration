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
# # Calibration demo - scmdata
#
# Here we give a basic demo of how to run a calibration with OpenSCM Calibration
# for a model that uses [scmdata](https://scmdata.readthedocs.io/en/latest)
# for data handling.
#
# ## Imports

# %%
from functools import partial
from typing import Callable

import emcee
import matplotlib.pyplot as plt
import more_itertools
import numpy as np
import numpy.typing as nptype
import pandas as pd
import pint
import scipy.integrate
import scmdata.run
from attrs import define
from emcwrap import DIMEMove
from multiprocess import Manager, Pool
from openscm_units import unit_registry as UREG
from tqdm.notebook import tqdm

from openscm_calibration import emcee_plotting as oc_emcee_plotting
from openscm_calibration.cost.scmdata import OptCostCalculatorSSE
from openscm_calibration.emcee_utils import (
    get_neg_log_prior,
    neg_log_info,
)
from openscm_calibration.minimize import to_minimize_full
from openscm_calibration.model_runner import OptModelRunner
from openscm_calibration.parameter_handling import (
    BoundDefinition,
    ParameterDefinition,
    ParameterOrder,
)
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
# Here we implement this using a setup which uses
# [pint](https://pint.readthedocs.io/)
# for unit support and
# [scmdata](https://scmdata.readthedocs.io/en/latest)
# for handling model output.
# This means we have to define some wrappers too,
# so that all the different data types can work with each other.
# This is a bit of extra work,
# but we think it is worth it to avoid the unit headaches
# and we do it here so you can see how this could work with your own data containers.

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
mass = UREG.Quantity(250, MASS_UNITS)


# %%
@define
class ExperimentDefinition:
    """
    Definition of an experiment to run
    """

    experiment_id: str
    """ID of the experiment"""

    dy_dt: Callable[
        [nptype.NDArray[np.float64], nptype.NDArray[np.float64]],
        nptype.NDArray[np.float64],
    ]
    """
    The function which defines dy_dt(t, y)

    This is what we solve.
    """

    x_0: pint.registry.UnitRegistry.Quantity
    """Initial position of the mass"""

    v_0: pint.registry.UnitRegistry.Quantity
    """Initial velocity of the mass"""


# %%
def run_experiments(
    k: float,
    x_zero: float,
    beta: float,
    mass: float = mass,
) -> scmdata.run.BaseScmRun:
    """
    Run experiments for a given set of parameters

    Parameters
    ----------
    k
        Spring constant [kg / s^2]

    x_zero
        Equilibrium position [m]

    beta
        Damping constant [kg / s]

    mass
        Mass on the spring [kg]

    Returns
    -------
    :
        Results of the experiments
    """
    # Avoiding pint conversions in the function actually
    # being solved makes things much faster,
    # but also harder to keep track of.
    # We recommend starting with pint everywhere first,
    # then optimising second
    # (either via using numpy quantities throughout,
    # or using Fortran or C wrappers).
    # We don't recommend usint pint's `wraps` functionalities,
    # because these don't place nice with parallelism.
    k_m = k.to(f"{MASS_UNITS} / {TIME_UNITS}^2").m
    x_zero_m = x_zero.to(LENGTH_UNITS).m
    beta_m = beta.to(f"{MASS_UNITS} / {TIME_UNITS}").m
    m_m = mass.to(MASS_UNITS).m

    def to_solve(
        t: nptype.NDArray[np.float64], y: nptype.NDArray[np.float64]
    ) -> nptype.NDArray[np.float64]:
        """
        Right-hand side of our equation i.e. dy/dt

        Parameters
        ----------
        t
            time

        y
            Current state of the system

        Returns
        -------
        :
            dy/dt
        """
        x = y[0]
        v = y[1]

        dv_dt = (-k_m * (x - x_zero_m) - beta_m * v) / m_m
        dx_dt = v

        out = np.array([dx_dt, dv_dt])

        return out

    res = {}
    # Grabbed out of global scope,
    # not ideal but ok for this example.
    time_axis_m = time_axis.to(TIME_UNITS).m
    for exp_definition in [
        ExperimentDefinition(
            experiment_id="non-eqm-start",
            dy_dt=to_solve,
            x_0=UREG.Quantity(1.8, "m"),
            v_0=UREG.Quantity(0, "m / s"),
        ),
        ExperimentDefinition(
            experiment_id="eqm-start",
            dy_dt=to_solve,
            x_0=UREG.Quantity(x_zero_m, LENGTH_UNITS),
            v_0=UREG.Quantity(1.3, "m / yr"),
        ),
    ]:
        res[exp_definition.experiment_id] = scipy.integrate.solve_ivp(
            exp_definition.dy_dt,
            t_span=[time_axis_m[0], time_axis_m[-1]],
            y0=[
                exp_definition.x_0.to(LENGTH_UNITS).m,
                exp_definition.v_0.to(f"{LENGTH_UNITS} / {TIME_UNITS}").m,
            ],
            t_eval=time_axis_m,
        )
        if not res[exp_definition.experiment_id].success:
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
    "k": UREG.Quantity(2100, "kg / s^2"),
    "x_zero": UREG.Quantity(-0.5, "m"),
    "beta": UREG.Quantity(6.3e11, "kg / s"),
}

target = run_experiments(**truth)
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
# This will be used to ensure a consistent order and units throughout.
# We also define the bounds we're going to use here
# because we need them later
# (but we don't always have to provide bounds).

# %%
parameter_order = ParameterOrder(
    (
        ParameterDefinition(
            "k",
            f"{MASS_UNITS} / {TIME_UNITS} ^ 2",
            BoundDefinition(
                lower=UREG.Quantity(300.0, "kg / s^2"),
                upper=UREG.Quantity(10000.0, "kg / s^2"),
            ),
        ),
        ParameterDefinition(
            "x_zero",
            LENGTH_UNITS,
            BoundDefinition(
                lower=UREG.Quantity(-2, "m"),
                upper=UREG.Quantity(2, "m"),
            ),
        ),
        ParameterDefinition(
            "beta",
            f"{MASS_UNITS} / {TIME_UNITS}",
            BoundDefinition(
                lower=UREG.Quantity(1e10, "kg / s"),
                upper=UREG.Quantity(1e12, "kg / s"),
            ),
        ),
    )
)
parameter_order.names


# %% [markdown]
# Next we define a function which, given pint quantities,
# returns the inputs needed for our `run_experiments` function.
# In this case this is not a very interesting function,
# but in other use cases the flexibility is helpful.


# %%
def run_experiments_input_generator(
    k: pint.Quantity, x_zero: pint.Quantity, beta: pint.Quantity
) -> dict[str, pint.Quantity]:
    """
    Create the inputs for `run_experiments`

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
    :
        Inputs for `run_experiments`
    """
    return {"k": k, "x_zero": x_zero, "beta": beta}


# %%
model_runner = OptModelRunner.from_parameter_order(
    parameter_order=parameter_order,
    do_model_runs_input_generator=run_experiments_input_generator,
    do_model_runs=run_experiments,
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
# Now we're ready to run our optimisation.

# %% [markdown]
# There are lots of choices that can be made during optimisation.
# At the moment, we show how to change many of them in this documentation.
# This does make the next cell somewhat overwhelming,
# but it also shows you all the choices you have available.
# As we do this more in future, we may create further abstractions.
# If these would be helpful,
# please [create an issue](https://github.com/openscm/OpenSCM-Calibration/issues/new?assignees=&labels=feature&projects=&template=feature_request.md&title=).

# %%
# Number of parallel processes to use
processes = 4


## Optimisation parameters - here we use short runs
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

## Maximum number of runs to store
# We think this is the right way to calculate this.
# If in doubt, you can always just increase this by some factor
# (at the cost of potentially reserving more memory than you need).
max_n_runs = (maxiter + 1) * popsize * len(parameter_order.parameters)


# Visualisation options
update_every = 4
thin_ts_to_plot = 5

# Function for converting output into a dict of runs and axes to plot on
convert_scmrun_to_plot_dict = partial(scmrun_as_dict, groups=["variable", "scenario"])

# Create axes to plot on
cost_name = "cost"
timeseries_axes = list(convert_scmrun_to_plot_dict(target).keys())

parameters_names = parameter_order.names
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
                parameter_order.bounds_m(),
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
start_local = optimize_res.x * 1.5
start_local

# %% [markdown]
# As for global optimisation above,
# there are lots of choices that can be made during optimisation
# and we show how to change many of them in this documentation.
# Once again, this can be somewhat overwhelming
# and if you would find further abstractions helpful,
# please [create an issue](https://github.com/openscm/OpenSCM-Calibration/issues/new?assignees=&labels=feature&projects=&template=feature_request.md&title=).

# %%
# Optimisation parameters
tol = 1e-4
# Maximum number of iterations to use
maxiter = 50

# We think this is how this works.
# As above, if you hit memory errors,
# just increase this by some factor.
max_n_runs = len(parameter_order.names) + 2 * maxiter

# Lots of options here
method = "Nelder-mead"

# Visualisation options
update_every = 10
thin_ts_to_plot = 5

# Create other objects
store = OptResStore.from_n_runs(
    max_n_runs,
    params=parameter_order.names,
    add_iteration_to_res=add_iteration_to_res_scmrun,
)
to_minimize = partial(
    to_minimize_full,
    store=store,
    cost_calculator=cost_calculator,
    model_runner=model_runner,
)


with tqdm(total=max_n_runs) as pbar:
    # Here we use a class method which auto-generates the figure for us.
    # This is just a convenience thing,
    # it does the same thing as the previous example under the hood.
    opt_plotter = OptPlotter.from_autogenerated_figure(
        cost_key=cost_name,
        params=parameter_order.names,
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
neg_log_prior = get_neg_log_prior(
    parameter_order,
    kind="uniform",
)
neg_log_info = partial(
    neg_log_info,
    neg_log_prior=neg_log_prior,
    model_runner=model_runner,
    negative_log_likelihood_calculator=cost_calculator,
)

# %% [markdown]
# We're using the DIME proposal from [emcwrap](https://github.com/gboehl/emcwrap).
# This claims to have an adaptive proposal distribution
# so requires less fine tuning and is less sensitive to the starting point.

# %%
ndim = len(parameter_order.names)
# emcwrap docs suggest 5 * ndim
nwalkers = 5 * ndim

start_emcee = [s + s / 100 * RNG.random(nwalkers) for s in optimize_res_local.x]
start_emcee = np.vstack(start_emcee).T

move = DIMEMove()

# %%
# Use HDF5 backend
filename = "how-to-run-a-calibration-scmdata-mcmc.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# %% [markdown]
# As for global and local optimisation above,
# there are lots of choices that can be made during MCMC
# and we show how to change many of them in this documentation.
# Once again, this can be somewhat overwhelming
# and if you would find further abstractions helpful,
# please [create an issue](https://github.com/openscm/OpenSCM-Calibration/issues/new?assignees=&labels=feature&projects=&template=feature_request.md&title=).

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
neg_log_likelihood_name = "neg_ll"
labels_chain = [neg_log_likelihood_name, *parameter_order.names]

## Setup plots
fig_chain, axd_chain = plt.subplot_mosaic(
    mosaic=[[lc] for lc in labels_chain],
    figsize=(10, 5),
)
holder_chain = display(fig_chain, display_id=True)  # noqa: F821 # used in a notebook

fig_dist, axd_dist = plt.subplot_mosaic(
    mosaic=[[parameter] for parameter in parameter_order.names],
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
truths_corner = [
    truth[parameter.name].to(parameter.unit).m
    for parameter in parameter_order.parameters
]

with Pool(processes=processes) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_prob_fn=neg_log_info,
        # Could be handy, but requires changes in other functions.
        # One for the future.
        # parameter_names=parameter_order,
        moves=move,
        backend=backend,
        blobs_dtype=[("neg_log_prior", float), ("neg_log_likelihood", float)],
        pool=pool,
    )

    for step_info in oc_emcee_plotting.plot_emcee_progress(
        sampler=sampler,
        iterations=max_iterations,
        burnin=burnin,
        thin=thin,
        plot_every=plot_every,
        parameter_order=parameter_order.names,
        neg_log_likelihood_name=neg_log_likelihood_name,
        start=start_emcee if sampler.iteration < 1 else None,
        holder_chain=holder_chain,
        figure_chain=fig_chain,
        axes_chain=axd_chain,
        holder_dist=holder_dist,
        figure_dist=fig_dist,
        axes_dist=axd_dist,
        holder_corner=holder_corner,
        figure_corner=fig_corner,
        holder_tau=holder_tau,
        figure_tau=fig_tau,
        ax_tau=ax_tau,
        corner_kwargs=dict(truths=truths_corner),
    ):
        print(f"{step_info.steps_post_burnin=}. {step_info.acceptance_fraction=:.3f}")

# Close all the figures to avoid them appearing twice
for _ in range(4):
    plt.close()
