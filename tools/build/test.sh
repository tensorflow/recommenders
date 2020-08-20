#!/bin/bash

set -e   # fail and exit on any command erroring

# Install.
source ./tools/build/pip_install.sh

# Install test dependencies.
pip install pytest

# Run tests.
py.test .
