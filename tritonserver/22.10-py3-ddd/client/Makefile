# This Makefile helps in setting up a Python virtual environment,
# installing necessary packages, and managing project setup tasks.

.PHONY: virtualenv setup install help

# Define the directory for the virtual environment.
V_ENV := triton_env
# Define the path to the Python executable within the virtual environment.
VENV_PYTHON := $(V_ENV)/bin/python3

# Target for creating the virtual environment
virtualenv:
	clear
	@echo "Creating virtual environment..."
	# Exit immediately if a command exits with a non-zero status
	set -e; \
	rm -rf $(V_ENV); \
	python3 -m venv $(V_ENV)

# Target for setting up the virtual environment by upgrading pip and setuptools
setup: virtualenv
	@echo "Setting up virtual environment..."
	set -e; \
	$(VENV_PYTHON) -m pip install --upgrade pip setuptools

# Target for installing the triton-python-client package within the virtual environment
install: setup
	@echo "Installing triton-python-client..."
	set -e; \
	$(VENV_PYTHON) -m pip install tritonclient[all] typer scikit-learn torch alibi-detect
	@echo "Run the following command to activate the new python env:"
	@echo "$$ source $(V_ENV)/bin/activate"

# Target to display the Makefile's content as a form of help.
help:
	@echo "Available targets:"
	@echo "  virtualenv: Create a new virtual environment."
	@echo "  setup: Set up the virtual environment by upgrading pip and setuptools."
	@echo "  install: Install triton-python-client within the virtual environment."
	@echo "  help: Display this help message."
	@cat Makefile
