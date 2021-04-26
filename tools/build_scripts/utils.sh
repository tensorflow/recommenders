#!/bin/bash

# Setup virtualenv
# create_virtualenv my_new_env
# or
# create_virtualenv my_new_env python3.6
function create_virtualenv() {
  local env_name=$1
  local env_python=${2:-python3.6}
  mkdir -p ~/virtualenv
  pushd ~/virtualenv
  rm -rf $env_name
  virtualenv --no-pip -p $env_python $env_name
  source $env_name/bin/activate
  # Keep using an old version of pip to work around
  # https://github.com/pypa/pip/blob/6d636902d7712f77abdb4428c290ba9bdbe70d9c/news/9831.bugfix.rst
  python -m ensurepip
  python -m pip install -U pip==21.0.1
  pip install "pip==21.0.1"
  pip --version
  popd
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
