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

This is a public version of the library and hence does not include
internal google stuff.
"""

__version__ = "v0.7.6"

# This version does not include internal tfrs google library.
from tensorflow_recommenders import examples
from tensorflow_recommenders import experimental
from tensorflow_recommenders import layers
from tensorflow_recommenders import metrics
from tensorflow_recommenders import models
from tensorflow_recommenders import tasks
from tensorflow_recommenders import types


Model = models.Model

# Artificially using the libraries in order to be able to use the tfrs_pub
# without these imports if needed and not generate a lint error.
__use_examples = examples
__use_experimental = experimental
__use_layers = layers
__use_metrics = metrics
__use_models = models
__use_tasks = tasks
__use_types = types
