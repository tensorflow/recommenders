# Copyright 2021 The TensorFlow Recommenders Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint-as: python3
"""Tools for cleaning and testing notebooks."""

import glob
import os
import subprocess
import tempfile

from typing import Text

import fire
import nbformat


def clean_cell(cell):
  """Cleans a cell."""
  metadata = cell.metadata

  for key in ("pinned", "imported_from", "executionInfo", "outputId"):
    if key in metadata:
      del metadata[key]

  if cell["cell_type"] == "code":
    cell["execution_count"] = 0


def clean_notebook(notebook):
  """Cleans a notebook."""
  colab = notebook["metadata"]["colab"]

  for key in ("defaultview", "views", "last_runtime", "provenance"):
    if key in colab:
      del colab[key]

  for cell in notebook.cells:
    clean_cell(cell)

  return notebook


class NBTool:
  """Tool for checking and cleaning notebooks."""

  def format(self, path):
    """Formats notebooks."""

    for notebook_path in glob.glob(os.path.join(path, "*ipynb")):
      print(f"Formatting {notebook_path}")

      with open(notebook_path, "r") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

      with open(notebook_path, "w") as notebook_file:
        nbformat.write(notebook, notebook_file)

  def clean(self, path: Text):
    """Cleans notebooks."""
    for notebook_path in glob.glob(os.path.join(path, "*ipynb")):
      print(f"Cleaning {notebook_path}")

      with open(notebook_path, "r") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

      notebook = clean_notebook(notebook)

      with open(notebook_path, "w") as notebook_file:
        nbformat.write(notebook, notebook_file)

  def check(self, path: Text):
    """Executes a notebook, checking for execution errors."""

    with tempfile.NamedTemporaryFile(mode="w", delete=True) as fle:
      fname = fle.name
      args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
              "--ExecutePreprocessor.timeout=600",
              "--ExecutePreprocessor.kernel_name=python3",
              "--output", fname, path]

      try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
      except subprocess.CalledProcessError as e:
        raise Exception(
            f"Execution of notebook {path} failed: {e.stdout, e.stderr}.")

  def check_all(self, path: Text):
    """Runs all notebooks under path."""
    for notebook_path in glob.glob(os.path.join(path, "*ipynb")):
      print(f"Executing {notebook_path}")

      self.check(notebook_path)


if __name__ == "__main__":
  fire.Fire(NBTool, name="nbtool")
