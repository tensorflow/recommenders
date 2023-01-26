#!/bin/bash

set -e   # fail and exit on any command erroring

# Install.
source ./tools/build_scripts/pip_install.sh

# Install test dependencies.
python3.10 -m pip install pytest

# Run tests.
python3.10 -m pytest -v .
