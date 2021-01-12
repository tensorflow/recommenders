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

# lint-as: python3
r"""Tool to generate API docs.

# How to run

Install tensorflow_docs if needed:

```
pip install git+https://github.com/tensorflow/docs
```

Run the docs generator:

```shell
python $(pwd)/tensorflow_recommenders/tools/build_api_docs.py
```
"""

from typing import Text

import fire

import tensorflow as tf

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_recommenders as tfrs


GITHUB_CODE_PATH = (
    "https://github.com/tensorflow/recommenders/"
    "blob/main/tensorflow_recommenders/"
)


def _hide_layer_and_module_methods():
  """Hide methods and properties defined in the base classes of Keras layers.

  We hide all methods and properties of the base classes, except:
  - `__init__` is always documented.
  - `call` is always documented, as it can carry important information for
    complex layers.
  """

  module_contents = list(tf.Module.__dict__.items())
  model_contents = list(tf.keras.Model.__dict__.items())
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())

  for name, obj in module_contents + layer_contents + model_contents:
    if name == "__init__":
      # Always document __init__.
      continue

    if name == "call":
      # Always document `call`.
      if hasattr(obj, doc_controls._FOR_SUBCLASS_IMPLEMENTERS):  # pylint: disable=protected-access
        delattr(obj, doc_controls._FOR_SUBCLASS_IMPLEMENTERS)  # pylint: disable=protected-access
      continue

    # Otherwise, exclude from documentation.
    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass


def build_api_docs(output_dir: Text = "/tmp/tensorflow_recommenders/api_docs",
                   code_url_prefix: Text = GITHUB_CODE_PATH,
                   search_hints: bool = True,
                   site_path: Text = "recommenders/api_docs/") -> None:
  """Builds the API docs."""

  _hide_layer_and_module_methods()

  print(f"Writing docs to {output_dir}")

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Recommenders",
      py_modules=[("tfrs", tfrs)],
      code_url_prefix=code_url_prefix,
      search_hints=search_hints,
      site_path=site_path,
      callbacks=[
          public_api.local_definitions_filter,
          public_api.explicit_package_contents_filter
      ])
  doc_generator.build(output_dir=output_dir)


if __name__ == "__main__":
  fire.Fire(build_api_docs, name="build_api_docs")
