# Makefile to help automate key steps

.DEFAULT_GOAL := help


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
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: checks
checks:  ## run all the linting checks of the codebase
	@echo "=== flake8 ==="; poetry run flake8 src tests || echo "--- flake8 failed ---" >&2; \
		echo "=== black ==="; poetry run black --check src tests || echo "--- black failed ---" >&2; \
		echo "=== isort ==="; poetry run isort --check-only src tests || echo "--- isort failed ---" >&2; \
		echo "======"

.PHONY: black
black:  ## format the code using black
	poetry run black src tests

.PHONY: isort
isort:  ## format the code using black
	poetry run isort src tests

.PHONY: test
test:  ## run the tests
	poetry run pytest -r a -vv

virtual-environment:  ## update virtual environment, create a new one if it doesn't already exist
	# Put virtual environments in the project
	poetry config virtualenvs.in-project true
	poetry install
