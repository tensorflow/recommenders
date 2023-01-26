#!/bin/bash

set -e   # fail and exit on any command erroring
set -x

TF_VERSION="2.9.0"

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

which python3.10
python3.10 --version

# Install PyPI-related packages.
python3.10 -m pip install -q wheel twine pyopenssl

echo "Checking out commit $GIT_COMMIT_ID..."
git checkout $GIT_COMMIT_ID

echo "Building source distribution..."

# Build the wheels
python3.10 setup.py sdist $SETUP_ARGS
python3.10 setup.py bdist_wheel $SETUP_ARGS

# Check setup.py.
twine check dist/*

# Install and test the distribution
echo "Running tests..."
python3.10 -m pip install dist/*.whl
python3.10 -m pip install scann
python3.10 -m pip install pytest
python3.10 -m pytest -v .

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
