# Copyright 2025 The TensorFlow Recommenders Authors.
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

"""TensorFlow Recommenders is a library for building recommender system models.

It helps with the full workflow of building a recommender system: data
preparation, model formulation, training, evaluation, and deployment.

It's built on Keras and aims to have a gentle learning curve while still giving
you the flexibility to build complex models.
"""


# For clear error messaging, check at the earliest opportunity for a
# compatible version of TF/Keras, before any import below fails obscurely.
# pylint: disable=g-statement-before-imports,g-import-not-at-top
def _check_keras_version():
  import tensorflow as tf

  keras_version_fn = getattr(tf.keras, "version", None)
  if keras_version_fn:  # Not present in tf.keras for v2 / before TF 2.16.
    keras_version = keras_version_fn()
    if keras_version.startswith("3."):
      raise ImportError(
          "Package tensorflow_recommenders requires tf.keras to be Keras"
          f" version 2 but got version {keras_version}. "
          "For open-source TensorFlow 2.16 and above, "
          "set the environment variable TF_USE_LEGACY_KERAS=1 to fix. "
          "For more information, see for example "
          "https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/keras_version.md"
      )


_check_keras_version()
del _check_keras_version
# pylint: enable=g-statement-before-imports

__version__ = "v0.7.6"

from tensorflow_recommenders import examples
from tensorflow_recommenders import experimental
# Internal extension library import.
from tensorflow_recommenders import layers
from tensorflow_recommenders import metrics
from tensorflow_recommenders import models
from tensorflow_recommenders import tasks
from tensorflow_recommenders import types


Model = models.Model
