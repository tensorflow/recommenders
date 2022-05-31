#!/bin/bash

function setup_python() {
  local python_version=$1
  pyenv install --skip-existing $python_version
  pyenv global $python_version
  which python
  python --version
  echo "Upgrading pip."
  pip install --upgrade pip
}

function install_tf() {
  local version=$1
  if [[ "$version" == "tf-nightly"  ]]
  then
    pip install -q tf-nightly;
  else
    pip install -q "tensorflow==$version"
  fi
}
