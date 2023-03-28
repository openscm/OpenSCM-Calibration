import pytest

from openscm_calibration.iter_utils import repeat_elements


@pytest.mark.parametrize(
    "inp, n_repeats, exp",
    (
        (["a", "b"], 2, ["a", "a", "b", "b"]),
        (["a", "b", "c"], 3, ["a", "a", "a", "b", "b", "b", "c", "c", "c"]),
    ),
)
def test_repeat_elements(inp, n_repeats, exp):
    res = repeat_elements(inp, n_repeats)

    assert res == exp
