import re
from unittest.mock import Mock, patch

import pytest

from openscm_calibration.matplotlib_utils import get_fig_axes_holder_from_mosaic


@pytest.mark.parametrize("mosaic", ([["a", "b"], ["c" "."]], [["a", "b", "c"]]))
@pytest.mark.parametrize(
    "kwargs, kwargs_exp",
    (
        (None, {}),
        ({"figsize": (6, 6)}, {"figsize": (6, 6)}),
    ),
)
@patch("openscm_calibration.matplotlib_utils.IPython")
@patch("openscm_calibration.matplotlib_utils.plt")
def test_get_fig_axes_holder_from_mosaic(
    mock_plt, mock_ipython, mosaic, kwargs, kwargs_exp
):
    mock_plt.subplot_mosaic = Mock(return_value=["abc", "bcd"])
    mock_ipython.display.display = Mock(return_value="holder")

    if kwargs is not None:
        call_kwargs = kwargs
    else:
        call_kwargs = {}

    res = get_fig_axes_holder_from_mosaic(mosaic, **call_kwargs)

    mock_plt.subplot_mosaic.assert_called_once_with(mosaic=mosaic, **kwargs_exp)
    mock_ipython.display.display.assert_called_once_with(
        mock_plt.subplot_mosaic.return_value[0], display_id=True
    )

    assert len(res) == 3
    assert res[0] == mock_plt.subplot_mosaic.return_value[0]
    assert res[1] == mock_plt.subplot_mosaic.return_value[1]
    assert res[2] == mock_ipython.display.display.return_value


@patch("openscm_calibration.matplotlib_utils.HAS_MATPLOTLIB", False)
def test_no_matplotlib():
    error_msg = re.escape(
        "``get_fig_axes_holder_from_mosaic`` requires matplotlib " "to be installed"
    )
    with pytest.raises(ImportError, match=error_msg):
        get_fig_axes_holder_from_mosaic([["a"]])


@patch("openscm_calibration.matplotlib_utils.HAS_IPYTHON", False)
def test_no_ipython():
    error_msg = re.escape(
        "``get_fig_axes_holder_from_mosaic`` requires IPython to be installed"
    )
    with pytest.raises(ImportError, match=error_msg):
        get_fig_axes_holder_from_mosaic([["a"]])
