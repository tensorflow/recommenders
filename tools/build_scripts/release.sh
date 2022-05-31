#!/bin/bash

set -e   # fail and exit on any command erroring

TF_VERSION="2.6.0"

GIT_COMMIT_ID=${1:-""}
[[ -z $GIT_COMMIT_ID ]] && echo "Must provide a commit." && exit 1
SETUP_ARGS=""
if [ "$GIT_COMMIT_ID" = "nightly" ]
then
  echo "Nightly version building currently not implemented."
  exit 1
fi

# Import build functions.
source ./tools/build_scripts/utils.sh

# Set up Python.
echo "Setting up Python..."
setup_python "$PY_VERSION"

# Install PyPI-related packages.
pip install -q --upgrade setuptools pip
pip install -q wheel twine pyopenssl

echo "Checking out commit $GIT_COMMIT_ID..."
git checkout $GIT_COMMIT_ID

echo "Building source distribution..."

# Build the wheels
python setup.py sdist $SETUP_ARGS
python setup.py bdist_wheel $SETUP_ARGS

# Check setup.py.
twine check dist/*

# Install and test the distribution
echo "Running tests..."
pip install dist/*.whl
pip install scann
pip install pytest
py.test -v .

# Publish to PyPI
read -p "Publish? (y/n) " -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Publishing to PyPI."
  twine upload dist/*
else
  echo "Skipping upload."
fi

echo "Done."
