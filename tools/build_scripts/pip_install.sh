#!/bin/bash

set -x
set -e   # fail and exit on any command erroring

# Need to set these env vars
: "${TF_VERSION:?}"
: "${PY_VERSION:?}"

# Import build functions.
source ./tools/build_scripts/utils.sh

# Set up a virtualenv.
echo "Creating virtualenv..."
create_virtualenv "tfrs_$TF_VERSION_$PY_VERSION" "python$PY_VERSION"

# Install TensorFlow.
echo "Installing TensorFlow..."
install_tf "$TF_VERSION"

# Install TensorFlow Recommenders.
echo "Installing TensorFlow Recommenders..."
pip install -e .[docs]

# Test successful build.
echo "Testing import..."
python -c "import tensorflow_recommenders as tfrs"

echo "Done."
