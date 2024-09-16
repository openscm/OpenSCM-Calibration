# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## OpenSCM Calibration v0.6.0 (2024-09-16)

### ⚠️  Breaking Changes

- Re-factored to make scmdata an actually optional dependency.
  See the how-to docs for an example of how to get the same behaviour with the new API. ([#29](https://github.com/openscm/OpenSCM-Calibration/pull/29))
- - [`get_autocorrelation_info`][openscm_calibration.emcee_utils.get_autocorrelation_info] now returns a [`AutoCorrelationInfo`][openscm_calibration.emcee_utils.AutoCorrelationInfo] object, rather than a [`dict`][].
  - [`x_and_parameters_to_named_with_units`][openscm_calibration.model_runner.x_and_parameters_to_named_with_units] now expects a [`ParameterOrder`][openscm_calibration.parameter_handling.ParameterOrder] object, rather than a [`list`][] of [`tuple`][].

  ([#31](https://github.com/openscm/OpenSCM-Calibration/pull/31))

### 🆕 Features

- - Added a number of functions to [`emcee_plotting`][openscm_calibration.emcee_plotting]
    and [`emcee_utils`][openscm_calibration.emcee_utils].
    These were extracted from the how-to guides.
  - Added [`parameter_handling`][openscm_calibration.parameter_handling] to clarify parameter handling, particularly units and order preservation.

  ([#31](https://github.com/openscm/OpenSCM-Calibration/pull/31))

### 🎉 Improvements

- Made pandas, scmdata and IPython optional dependencies. ([#29](https://github.com/openscm/OpenSCM-Calibration/pull/29))

### 📚 Improved Documentation

- Updated the how-to guides for calibration.
  These now have one example with a custom data container and one example using [scmdata](https://scmdata.readthedocs.io/en/latest). ([#31](https://github.com/openscm/OpenSCM-Calibration/pull/31))

### 🔧 Trivial/Internal Changes

- [#29](https://github.com/openscm/OpenSCM-Calibration/pull/29), [#30](https://github.com/openscm/OpenSCM-Calibration/pull/30)


## OpenSCM Calibration v0.5.2 (2024-09-13)

### 🔧 Trivial/Internal Changes

- [#28](https://github.com/openscm/OpenSCM-Calibration/pull/28)


## OpenSCM Calibration v0.5.1 (2024-09-13)

### 🔧 Trivial/Internal Changes

- [#27](https://github.com/openscm/OpenSCM-Calibration/pull/27)
