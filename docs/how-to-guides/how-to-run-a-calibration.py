# ---
# jupyter:
#   jupytext:
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
from typing import Callable

import emcee
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as nptype
import pint
import scipy.integrate
import tqdm.autonotebook as tqdman
from attrs import define
from emcwrap import DIMEMove
from multiprocess import Manager, Pool
from openscm_units import unit_registry as UREG

from openscm_calibration import emcee_plotting as oc_emcee_plotting
from openscm_calibration.calibration_demo import (
    CostCalculator,
    ExperimentResult,
    ExperimentResultCollection,
    Timeseries,
    add_iteration_info,
    convert_results_to_plot_dict,
    get_timeseries,
    plot_timeseries,
)
from openscm_calibration.minimize import to_minimize_full
from openscm_calibration.model_runner import OptModelRunner
from openscm_calibration.scipy_plotting import (
    CallbackProxy,
    OptPlotter,
    get_optimisation_mosaic,
    get_ymax_default,
    plot_costs,
)
from openscm_calibration.store import OptResStore

# %%
# Adjust plotting defaults
plt.rcParams["axes.xmargin"] = 0.0

# %%
# Set the seed to ensure reproducibility
seed = 4729523
np.random.seed(seed)  # noqa: NPY002 # want to set global seed for emcee
RNG = np.random.default_rng(seed=seed)

# %%
# Ensure that pint uses the desired unit registry throughout
pint.set_application_registry(UREG)

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
# In our experience, units are too easy to get wrong.
# Hence, we implement this using [pint](https://pint.readthedocs.io/).
# This means we have to define some wrappers too,
# so that scipy can work with our pint quantities.
# This is a bit of extra work,
# but we think it is worth it to avoid the unit headaches.

# %% [markdown]
# ## Experiments
#
# We're going to calibrate the model's response in two experiments:
#
# - starting out of equilibrium
# - starting at the equilibrium position but already moving
# - starting out of equilibrium and already moving
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
    mass: float = mass.to(MASS_UNITS).m,
) -> ExperimentResultCollection:
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

        dv_dt = (-k * (x - x_zero) - beta * v) / mass
        dx_dt = v

        out = np.array([dx_dt, dv_dt])

        return out

    res_l = []
    # Grabbed out of global scope,
    # not ideal but ok for this example.
    time_axis_m = time_axis.to(TIME_UNITS).m

    for exp_definition in [
        ExperimentDefinition(
            experiment_id="non-eqm-start",
            dy_dt=to_solve,
            x_0=UREG.Quantity(1.2, "m"),
            v_0=UREG.Quantity(0, "m / s"),
        ),
        ExperimentDefinition(
            experiment_id="eqm-start-moving",
            dy_dt=to_solve,
            x_0=UREG.Quantity(x_zero, LENGTH_UNITS),
            v_0=UREG.Quantity(0.3, "m / yr"),
        ),
        ExperimentDefinition(
            experiment_id="non-eqm-start-moving",
            dy_dt=to_solve,
            x_0=UREG.Quantity(0.6, "m"),
            v_0=UREG.Quantity(1.0, "m / yr"),
        ),
    ]:
        solve_res = scipy.integrate.solve_ivp(
            exp_definition.dy_dt,
            t_span=[time_axis_m[0], time_axis_m[-1]],
            y0=[
                exp_definition.x_0.to(LENGTH_UNITS).m,
                exp_definition.v_0.to(f"{LENGTH_UNITS} / {TIME_UNITS}").m,
            ],
            t_eval=time_axis_m,
        )
        if not solve_res.success:
            msg = "Model failed to solve"
            raise ValueError(msg)

        res_l.append(
            ExperimentResult(
                experiment_id=exp_definition.experiment_id,
                result=Timeseries(
                    values=UREG.Quantity(solve_res.y[0, :], LENGTH_UNITS),
                    time=time_axis,
                ),
            )
        )

    out = ExperimentResultCollection(tuple(res_l))

    return out


# %% [markdown]
# Next we define a function which, given pint quantities,
# returns the inputs needed for our `do_experiments` function.
# In this case this is not a very interesting function,
# but in other use cases the flexibility is helpful.
# For example, by converting the quantities to plain floats.
#
# This is a bit like writing our own version of
# [pint's wrapping functions](https://pint.readthedocs.io/en/0.10.1/wrapping.html#wrapping-and-checking-functions).
# One of the key advantages of doing it this way
# is that we can parallelise our experiment running function.
# Unfortunately, pint's unit registry does not parallelise happily
# because it uses weak references, which can't be pickled.


# %%
def do_model_runs_input_generator(
    k: pint.registry.UnitRegistry.Quantity,
    x_zero: pint.registry.UnitRegistry.Quantity,
    beta: pint.registry.UnitRegistry.Quantity,
) -> dict[str, pint.registry.UnitRegistry.Quantity]:
    """
    Create the inputs for `do_experiments`

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
        Inputs for `do_experiments`
    """
    return {
        "k": k.to(f"{MASS_UNITS} / {TIME_UNITS}^2").m,
        "x_zero": x_zero.to(LENGTH_UNITS).m,
        "beta": beta.to(f"{MASS_UNITS} / {TIME_UNITS}").m,
    }


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
    "x_zero": UREG.Quantity(-1.0, "m"),
    "beta": UREG.Quantity(1.7e11, "kg / s"),
}

target = run_experiments(**do_model_runs_input_generator(**truth))
fig, ax = plt.subplots()
target.lineplot(ax=ax)
ax.legend()
ax.set_yticks(np.arange(-4, 4.01))
ax.grid()
plt.show()
# target

# %% [markdown]
# ### Cost calculation
#
# The next thing is to decide how we're going to calculate the cost function.
# There are many options here,
# in this case we're going to use the sum of squared errors.

# %%
cost_calculator = CostCalculator(
    target=target,
    normalisation=UREG.Quantity(0.5, "m"),
)

assert cost_calculator.calculate_cost(target) == 0

not_target = run_experiments(
    **do_model_runs_input_generator(
        k=UREG.Quantity(3000, "kg / s^2"),
        x_zero=UREG.Quantity(1.0, "m"),
        beta=UREG.Quantity(1.7e11, "kg / s"),
    )
)
assert cost_calculator.calculate_cost(not_target) > 0

# %% [markdown]
# ### Model runner
#
# Scipy does everything using numpy arrays.
# Here we use a wrapper that converts them to pint quantities before running.

# %% [markdown]
# Firstly, we define the parameters we're going to optimise.
# This will be used to ensure a consistent order and units throughout.

# %%
parameters = [
    ("k", f"{MASS_UNITS} / {TIME_UNITS} ^ 2"),
    ("x_zero", LENGTH_UNITS),
    ("beta", f"{MASS_UNITS} / {TIME_UNITS}"),
]
parameters

# %%
model_runner = OptModelRunner.from_parameters(
    params=parameters,
    do_model_runs_input_generator=do_model_runs_input_generator,
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
bounds = [[v.to(unit).m for v in bounds_dict[k]] for k, unit in parameters]
bounds_dict

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
max_n_runs = (maxiter + 1) * popsize * len(parameters)


# Visualisation options
update_every = 4
thin_ts_to_plot = 5


# Create axes to plot on
cost_name = "cost"
timeseries_axes = list(convert_results_to_plot_dict(target).keys())
parameters_names = [v[0] for v in parameters]

mosaic = get_optimisation_mosaic(
    cost_key=cost_name,
    params=parameters_names,
    timeseries=timeseries_axes,
    cost_col_relwidth=2,
    n_parameters_per_row=3,
)

fig, axd = plt.subplot_mosaic(
    mosaic=mosaic,
    figsize=(8, 6),
)
holder = display(fig, display_id=True)  # noqa: F821 # used in a notebook


with Manager() as manager:
    store = OptResStore.from_n_runs_manager(
        max_n_runs,
        manager,
        params=parameters_names,
        add_iteration_to_res=add_iteration_info,
    )

    to_minimize = partial(
        to_minimize_full,
        store=store,
        cost_calculator=cost_calculator,
        model_runner=model_runner,
        known_error=ValueError,
    )

    with manager.Pool(processes=processes) as pool:
        with tqdman.tqdm(total=max_n_runs) as pbar:
            opt_plotter = OptPlotter(
                holder=holder,
                fig=fig,
                axes=axd,
                cost_key=cost_name,
                parameters=parameters_names,
                timeseries_axes=timeseries_axes,
                target=target,
                store=store,
                get_timeseries=get_timeseries,
                plot_timeseries=plot_timeseries,
                convert_results_to_plot_dict=convert_results_to_plot_dict,
                thin_ts_to_plot=thin_ts_to_plot,
                plot_costs=partial(
                    plot_costs,
                    # If you want a fixed y-axis on the costs axis,
                    # you can uncomment the below.
                    # get_ymax=lambda _: 5000
                ),
            )

            proxy = CallbackProxy(
                real_callback=opt_plotter,
                store=store,
                update_every=update_every,
                progress_bar=pbar,
                last_callback_val=0,
            )

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
                # If you get pickle errors, comment this out to see what is going on.
                # Then, we can recommend reading this stack overflow answer:
                # https://stackoverflow.com/a/30529992
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
start_local = optimize_res.x * 1.3
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
maxiter = 100

# We think this is how this works.
# As above, if you hit memory errors,
# just increase this by some factor.
max_n_runs = len(parameters) + 2 * maxiter

# Lots of options here
method = "Nelder-mead"

# Visualisation options
update_every = 20
thin_ts_to_plot = 5
parameters_names = [v[0] for v in parameters]

# Create other objects
store = OptResStore.from_n_runs(
    max_n_runs,
    params=parameters_names,
    add_iteration_to_res=add_iteration_info,
)
to_minimize = partial(
    to_minimize_full,
    store=store,
    cost_calculator=cost_calculator,
    model_runner=model_runner,
)


with tqdman.tqdm(total=max_n_runs) as pbar:
    # Here we use a class method which auto-generates the figure
    # for us. This is just a convenience thing, it does the same
    # thing as the previous example under the hood.
    opt_plotter = OptPlotter.from_autogenerated_figure(
        cost_key=cost_name,
        params=parameters_names,
        convert_results_to_plot_dict=convert_results_to_plot_dict,
        target=target,
        store=store,
        get_timeseries=get_timeseries,
        plot_timeseries=plot_timeseries,
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


# %%
assert False, "put neg_log_prior_bounds in package and check names against emcee docs"
assert False, "put log_prob in package and check names against emcee docs"

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
filename = "how-to-run-a-calibration-mcmc.h5"
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
# This number is definitely too small.
max_iterations = 125
burnin = 25
thin = 5

## Visualisation options
plot_every = 30
convergence_ratio = 50
parameter_order = [p[0] for p in parameters]
neg_log_likelihood_name = "neg_ll"
labels_chain = [neg_log_likelihood_name, *parameter_order]

# Stores for autocorrelation values
# (as a function of the number of steps performed).
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
        # Handy, but requires changes in other functions.
        # One for the future.
        # parameter_names=parameter_order,
        moves=move,
        backend=backend,
        blobs_dtype=[("neg_log_prior", float), ("neg_log_likelihood", float)],
        pool=pool,
    )

    # Split this logic out too
    # (into some sort of iterator i.e. function with yield)
    # for step in oc_emcee_plotting.plot_emcee_progress
    for step_info in oc_emcee_plotting.plot_emcee_progress(
        sampler=sampler,
        iterations=max_iterations,
        burnin=burnin,
        thin=thin,
        plot_every=plot_every,
        parameter_order=parameter_order,
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
        print(f"{step_info.steps_post_burnin=}")
        print(f"{step_info.acceptance_fraction=:.3f}")

# Close all the figures to avoid them appearing twice
for _ in range(4):
    plt.close()
