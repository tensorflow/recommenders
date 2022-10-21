#!/bin/bash

function install_tf() {
  local version=$1
  if [[ "$version" == "tf-nightly"  ]]
  then
    pip install -q tf-nightly;
  else
    pip install -q "tensorflow==$version"
  fi
  pip install -q urllib3
}
