# Makefile to help automate key steps

.DEFAULT_GOAL := help

VENV_DIR ?= venv

PYTHON=$(VENV_DIR)/bin/python
SETUP_CFG=setup.cfg
PYPROJECT_TOML=pyproject.toml


# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:  ## print short description of each target
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: docs
docs: $(VENV_DIR)  ## build the docs
	$(MAKE) -C docs html

.PHONY: test
test: $(VENV_DIR)  ## run the tests
	$(VENV_DIR)/bin/pytest tests -r a -vv

virtual-environment:  ## update virtual environment, create a new one if it doesn't already exist
	make $(VENV_DIR)

$(VENV_DIR):
	[ -d $(VENV_DIR) ] || python3 -m venv $(VENV_DIR)

	# Put virutal environments in the project
	$(VENV_DIR)/bin/poetry virtualenvs.in-project
	$(VENV_DIR)/bin/poetry install

	touch $(VENV_DIR)
