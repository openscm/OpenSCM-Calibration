"""
Integration tests of :mod:`openscm_calibration.scipy_plotting`
"""

import numpy as np
import pytest

from openscm_calibration.scipy_plotting.scmdata import get_timeseries_scmrun

pd = pytest.importorskip("pandas")
pdt = pytest.importorskip("pandas.testing")
scmdata_run = pytest.importorskip("scmdata.run")

RNG = np.random.default_rng()


@pytest.mark.parametrize(
    "time_axis, years, years_exp",
    (
        (None, [2010, 2010.5, 2011], [2010 + 1 / 24, 2010 + 13 / 24, 2011 + 1 / 24]),
        (
            "year-month",
            [2010, 2010.5, 2011],
            [2010 + 1 / 24, 2010 + 13 / 24, 2011 + 1 / 24],
        ),
        ("year", [2010.5, 2011.3], [2010, 2011]),
        ("year", [2010, 2020], [2010, 2020]),
        ("days since 1970-01-01", [2010, 2020], [14610, 18262]),
    ),
)
def test_get_timeseries_default(time_axis, years, years_exp):
    index = pd.MultiIndex.from_tuples(
        [(name, "K") for name in ["dT", "dT_Ocean"]], names=("variable", "unit")
    )

    inp_ts = pd.DataFrame(
        RNG.random(size=(index.size, len(years))),
        columns=years,
        index=index,
    )
    inp = scmdata_run.BaseScmRun(inp_ts)

    call_kwargs = {}

    for name, para in (("time_axis", time_axis),):
        if para is not None:
            call_kwargs[name] = para

    res = get_timeseries_scmrun(inp, **call_kwargs)

    exp = inp_ts.set_index(["variable", "unit"]).T
    exp.index = years_exp
    exp.index.name = "time"
    exp.columns = exp.columns.reorder_levels(res.columns.names)

    pdt.assert_frame_equal(res, exp)
