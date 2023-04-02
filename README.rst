.. sec-begin-description

OpenSCM-Calibration
===================

Calibration tools for simple climate models (and other things perhaps,
surprise us with your applications).

.. sec-end-description

TO-DO:

- add https://commitizen-tools.github.io/commitizen/
- tests
- CI (checks plus docs)
- notebooks

.. sec-begin-installation

Installation
------------

OpenSCM-Calibration can be installed with conda or pip:

.. code:: bash

    pip install openscm-calibration
    conda install -c conda-forge openscm-calibration

Additional dependencies can be installed using

.. code:: bash

    # To add plotting dependencies
    pip install openscm-calibration[plots]
    # To run the notebooks
    pip install openscm-calibration[notebooks, plots]
    # If you are installing with conda, we recommend
    # doing this by hand as there is no stable solution yet
    # (issue here: https://github.com/conda/conda/issues/7502)

.. sec-end-installation

.. sec-begin-installation-dev

For developers
~~~~~~~~~~~~~~

For development, we rely on `poetry <https://python-poetry.org>`_ for all our
dependency management. To get started, you will need to make sure that poetry
is installed
(https://python-poetry.org/docs/#installing-with-the-official-installer, we
found that pipx and pip worked better to install on a Mac).

For all of work, we use our ``Makefile``.
You can read the instructions out and run the commands by hand if you wish,
but we generally discourage this because it can be error prone and doesn't
update if dependencies change (e.g. the environment is updated).
In order to create your environment, run ``make virtual-environment -B``.

If there are any issues, the messages from the ``Makefile`` should guide you
through. If not, please raise an issue in the
`issue tracker <https://github.com/openscm/OpenSCM-Calibration/issues>`_.

.. sec-end-installation-dev
