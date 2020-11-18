#!/bin/bash

set -e   # fail and exit on any command erroring

# Install.
source ./tools/build_scripts/pip_install.sh

# Install test dependencies.
pip install pytest

# Run tests.
py.test -v .
