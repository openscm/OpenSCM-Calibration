<!--- --8<-- [start:description] -->
# OpenSCM Calibration

To do:

- move docs to markdown
    - including README
- docs CI
    - install the package too, otherwise doc auto-generation doesn't work
        - also push this to copier

For #26:
- install all optional dependencies for the tests
    - we want the tests to run with maximum coverage
        - also push this to copier

Calibration tools for simple climate models (with potential for applications elsewhere)

**Key info :**
[![Docs](https://readthedocs.org/projects/openscm-calibration/badge/?version=latest)](https://openscm-calibration.readthedocs.io)
[![Main branch: supported Python versions](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fopenscm%2FOpenSCM-Calibration%2Fmain%2Fpyproject.toml)](https://github.com/openscm/OpenSCM-Calibration/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/pypi/l/openscm-calibration?label=licence)](https://github.com/openscm/OpenSCM-Calibration/blob/main/LICENCE)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/openscm-calibration.svg)](https://pypi.org/project/openscm-calibration/)
[![PyPI install](https://github.com/openscm/OpenSCM-Calibration/actions/workflows/install-pypi.yaml/badge.svg?branch=main)](https://github.com/openscm/OpenSCM-Calibration/actions/workflows/install-pypi.yaml)

<!--- If you release on conda, you can use this
**Conda :**
[![Conda](https://img.shields.io/conda/vn/conda-forge/openscm-calibration.svg)](https://anaconda.org/conda-forge/openscm-calibration)
[![Conda platforms](https://img.shields.io/conda/pn/conda-forge/openscm-calibration.svg)](https://anaconda.org/conda-forge/openscm-calibration)
[![Conda install](https://github.com/openscm/OpenSCM-Calibration/actions/workflows/install-conda.yaml/badge.svg?branch=main)](https://github.com/openscm/OpenSCM-Calibration/actions/workflows/install-conda.yaml)
-->

**Tests :**
[![CI](https://github.com/openscm/OpenSCM-Calibration/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/openscm/OpenSCM-Calibration/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/openscm/OpenSCM-Calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/openscm/OpenSCM-Calibration)

**Other info :**
[![Last Commit](https://img.shields.io/github/last-commit/openscm/OpenSCM-Calibration.svg)](https://github.com/openscm/OpenSCM-Calibration/commits/main)
[![Contributors](https://img.shields.io/github/contributors/openscm/OpenSCM-Calibration.svg)](https://github.com/openscm/OpenSCM-Calibration/graphs/contributors)
## Status

<!---

We recommend having a status line in your repo
to tell anyone who stumbles on your repository where you're up to.
Some suggested options:

- prototype: the project is just starting up and the code is all prototype
- development: the project is actively being worked on
- finished: the project has achieved what it wanted
  and is no longer being worked on, we won't reply to any issues
- dormant: the project is no longer worked on
  but we might come back to it,
  if you have questions, feel free to raise an issue
- abandoned: this project is no longer worked on
  and we won't reply to any issues
-->

- development: the project is actively being worked on

<!--- --8<-- [end:description] -->

Full documentation can be found at:
[openscm-calibration.readthedocs.io](https://openscm-calibration.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.

## Installation

<!--- --8<-- [start:installation] -->
### As an application

If you want to use OpenSCM Calibration as an application,
then we recommend using the 'locked' version of the package.
This version pins the version of all dependencies too,
which reduces the chance of installation issues
because of breaking updates to dependencies.

The locked version of OpenSCM Calibration can be installed with

<!--- If you release on conda, you can use this
=== "mamba"
    ```sh
    mamba install -c conda-forge openscm-calibration-locked
    ```

=== "conda"
    ```sh
    conda install -c conda-forge openscm-calibration-locked
    ```

-->
=== "pip"
    ```sh
    pip install openscm-calibration[locked]
    ```

### As a library

If you want to use OpenSCM Calibration as a library,
for example you want to use it
as a dependency in another package/application that you're building,
then we recommend installing the package with the commands below.
This method provides the loosest pins possible of all dependencies.
This gives you, the package/application developer,
as much freedom as possible to set the versions of different packages.
However, the tradeoff with this freedom is that you may install
incompatible versions of OpenSCM Calibration's dependencies
(we cannot test all combinations of dependencies,
particularly ones which haven't been released yet!).
Hence, you may run into installation issues.
If you believe these are because of a problem in OpenSCM Calibration,
please [raise an issue](https://github.com/openscm/OpenSCM-Calibration/issues).

The (non-locked) version of OpenSCM Calibration can be installed with

<!--- If you release on conda, you can use this
=== "mamba"
    ```sh
    mamba install -c conda-forge openscm-calibration
    ```

=== "conda"
    ```sh
    conda install -c conda-forge openscm-calibration
    ```

-->
=== "pip"
    ```sh
    pip install openscm-calibration
    ```

Additional dependencies can be installed using

<!--- If you release on conda, you can use this
=== "mamba"
    If you are installing with mamba, we recommend
    installing the extras by hand because there is no stable
    solution yet (see [conda issue #7502](https://github.com/conda/conda/issues/7502))

=== "conda"
    If you are installing with conda, we recommend
    installing the extras by hand because there is no stable
    solution yet (see [conda issue #7502](https://github.com/conda/conda/issues/7502))

-->
=== "pip"
    ```sh
    # To add plotting dependencies
    pip install openscm-calibration[plots]
    # To add notebook dependencies
    pip install openscm-calibration[notebooks]
    ```

### For developers

For development, we rely on [pdm](https://pdm-project.org/en/latest/)
for all our dependency management.
To get started, you will need to make sure that pdm is installed
([instructions here](https://pdm-project.org/en/latest/#installation),
although we found that installing with [pipx](https://pipx.pypa.io/stable/installation/)
worked perfectly for us).

For all of work, we use our `Makefile`.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone.
In order to create your environment, run `make virtual-environment`.

If there are any issues, the messages from the `Makefile` should guide you through.
If not, please raise an issue in the
[issue tracker](https://github.com/openscm/OpenSCM-Calibration/issues).

For the rest of our developer docs, please see [development][development].

<!--- --8<-- [end:installation] -->

## Original template

This project was generated from this template:
[copier core python repository](https://gitlab.com/znicholls/copier-core-python-repository).
[copier](https://copier.readthedocs.io/en/stable/) is used to manage and
distribute this template.
