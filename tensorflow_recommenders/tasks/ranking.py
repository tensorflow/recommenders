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

# Lint-as: python3
"""A ranking task."""

from typing import List, Optional, Text

import tensorflow as tf


class Ranking(tf.keras.layers.Layer):
  """A ranking task.

  Recommender systems are often composed of two components:
  - a retrieval model, retrieving O(thousands) candidates from a corpus of
    O(millions) candidates.
  - a ranker model, scoring the candidates retrieved by the retrieval model to
    return a ranked shortlist of a few dozen candidates.

  This task helps with building ranker models. Usually, these will involve
  predicting signals such as clicks, cart additions, likes, ratings, and
  purchases.
  """

  def __init__(
      self,
      loss: Optional[tf.keras.losses.Loss] = None,
      metrics: Optional[List[tf.keras.metrics.Metric]] = None,
      name: Optional[Text] = None) -> None:
    """Initializes the task.

    Args:
      loss: Loss function. Defaults to BinaryCrossentropy.
      metrics: List of Keras metrics to be evaluated.
      name: Optional task name.
    """

    super().__init__(name=name)

    self._loss = (
        loss if loss is not None else tf.keras.losses.BinaryCrossentropy())
    self._ranking_metrics = metrics or []

  def call(self,
           labels: tf.Tensor,
           predictions: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           training: bool = False) -> tf.Tensor:
    """Computes the task loss and metrics.

    Args:
      labels: Tensor of labels.
      predictions: Tensor of predictions.
      sample_weight: Tensor of sample weights.
      training: Indicator whether training or test loss is being computed.

    Returns:
      loss: Tensor of loss values.
    """

    loss = self._loss(
        y_true=labels, y_pred=predictions, sample_weight=sample_weight)

    for metric in self.metrics:
      metric.update_state(
          y_true=labels, y_pred=predictions, sample_weight=sample_weight)

    return loss
