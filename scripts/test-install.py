"""
Test that all of our modules can be imported

Also test that associated constants are set correctly

Thanks https://stackoverflow.com/a/25562415/10473080
"""
import importlib
import pkgutil

import openscm_calibration
import openscm_calibration.emcee_plotting


def import_submodules(package_name):
    package = importlib.import_module(package_name)

    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        importlib.import_module(full_name)
        if is_pkg:
            import_submodules(full_name)


try:
    import seaborn  # noqa: F401

    assert openscm_calibration.emcee_plotting.HAS_SEABORN
except ImportError:
    assert not openscm_calibration.emcee_plotting.HAS_SEABORN

try:
    import corner  # noqa: F401

    assert openscm_calibration.emcee_plotting.HAS_CORNER
except ImportError:
    assert not openscm_calibration.emcee_plotting.HAS_CORNER


import_submodules("openscm_calibration")
print(openscm_calibration.__version__)
