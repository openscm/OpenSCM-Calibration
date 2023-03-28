import pandas as pd
import pytest
import scmdata.testing
from scmdata.run import BaseScmRun

from openscm_calibration.scmdata_utils import scmrun_as_dict


@pytest.fixture()
def basic_scmrun():
    df = pd.DataFrame(
        [
            [
                "a_model",
                "a_iam",
                "a_scenario",
                "World",
                "Primary Energy",
                "EJ/yr",
                1,
                6.0,
                6.0,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario",
                "World",
                "Primary Energy|Coal",
                "EJ/yr",
                0.5,
                3,
                3.0,
            ],
            [
                "a_model",
                "a_iam",
                "a_scenario2",
                "World",
                "Primary Energy",
                "EJ/yr",
                2,
                7,
                7.0,
            ],
        ],
        columns=[
            "climate_model",
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
            2005,
            2010,
            2015,
        ],
    )

    return BaseScmRun(df)


@pytest.mark.parametrize(
    "separator, exp_separator, groups, exp_keys",
    (
        (None, "_", ["climate_model"], ["a_model"]),
        (None, "_", ["scenario"], ["a_scenario", "a_scenario2"]),
        (None, "_", ["scenario", "model"], ["a_scenario_a_iam", "a_scenario2_a_iam"]),
        ("-", "-", ["scenario", "model"], ["a_scenario-a_iam", "a_scenario2-a_iam"]),
    ),
)
def test_groups(separator, exp_separator, groups, exp_keys, basic_scmrun):
    call_kwargs = {}
    if separator is not None:
        call_kwargs["separator"] = separator

    res = scmrun_as_dict(basic_scmrun, groups=groups, **call_kwargs)

    exp = {}
    for group in basic_scmrun.groupby(groups):
        key = exp_separator.join([group.get_unique_meta(g, True) for g in groups])
        exp[key] = group

    assert res.keys() == exp.keys()
    assert set(res.keys()) == set(exp_keys)
    for key in exp_keys:
        scmdata.testing.assert_scmdf_almost_equal(res[key], exp[key])


def test_non_existent_group(basic_scmrun):
    groups = ["climate_model", "junk"]

    with pytest.raises(KeyError):
        scmrun_as_dict(basic_scmrun, groups=groups)


def test_non_str_separator(basic_scmrun):
    with pytest.raises(AttributeError):
        scmrun_as_dict(basic_scmrun, groups=["climate_model"], separator=3)
