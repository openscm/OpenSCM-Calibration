# Makefile to help automate key steps

.DEFAULT_GOAL := help


# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


help:  ## print short description of each target
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: checks
checks:  ## run all the linting checks of the codebase
	@echo "=== black ==="; poetry run black --check src tests docs/source/conf.py scripts || echo "--- black failed ---" >&2; \
		echo "=== ruff ==="; poetry run ruff check src tests scripts || echo "--- ruff failed ---" >&2; \
		echo "=== mypy ==="; poetry run mypy src || echo "--- mypy failed ---" >&2; \
		echo "=== black docs ==="; poetry run blacken-docs docs/source/notebooks/*.md || echo "--- black docs failed ---" >&2; \
		echo "======"

.PHONY: black
black:  ## format the code using black
	poetry run black src tests docs/source/conf.py scripts

.PHONY: ruff-fixes
ruff-fixes:  ## fix the code using ruff
	poetry run ruff src tests scripts --fix

.PHONY: test
test:  ## run the tests
	poetry run pytest -r a --doctest-modules --cov

.PHONY: docs
docs:  ## build the docs
	poetry run sphinx-build -b html docs/source docs/build/html

.PHONY: black-docs
black-docs:  ## format the notebok examples using black
	poetry run blacken-docs docs/source/notebooks/*.md

.PHONY: check-commit-messages
check-commit-messages:  ## check commit messages
	poetry run cz check --rev-range c24e5e..HEAD

virtual-environment:  ## update virtual environment, create a new one if it doesn't already exist
	poetry lock --no-update
	# Put virtual environments in the project
	poetry config virtualenvs.in-project true
	poetry install --all-extras
	poetry run jupyter nbextension enable --py widgetsnbextension
