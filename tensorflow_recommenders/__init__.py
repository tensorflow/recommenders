# Copyright 2020 The TensorFlow Recommenders Authors.
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

"""Tensorflow recommenders."""

__version__ = "v0.1.0"

from tensorflow_recommenders import datasets
from tensorflow_recommenders import examples
from tensorflow_recommenders import layers
from tensorflow_recommenders import losses
from tensorflow_recommenders import metrics
from tensorflow_recommenders import models
from tensorflow_recommenders import tasks

Model = models.Model
