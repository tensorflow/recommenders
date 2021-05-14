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

"""TensorFlow Recommenders is a library for building recommender system models.

It helps with the full workflow of building a recommender system: data
preparation, model formulation, training, evaluation, and deployment.

It's built on Keras and aims to have a gentle learning curve while still giving
you the flexibility to build complex models.
"""

__version__ = "v0.5.1"

from tensorflow_recommenders import examples
from tensorflow_recommenders import experimental
# Internal extension library import.
from tensorflow_recommenders import layers
from tensorflow_recommenders import metrics
from tensorflow_recommenders import models
from tensorflow_recommenders import tasks
from tensorflow_recommenders import types


Model = models.Model
