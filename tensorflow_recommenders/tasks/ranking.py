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
"""A ranking task."""

from typing import List, Optional, Text

import tensorflow as tf


from tensorflow_recommenders.tasks import base


class Ranking(tf.keras.layers.Layer, base.Task):
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
      prediction_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
      label_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
      loss_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
      name: Optional[Text] = None) -> None:
    """Initializes the task.

    Args:
      loss: Loss function. Defaults to BinaryCrossentropy.
      metrics: List of Keras metrics to be evaluated.
      prediction_metrics: List of Keras metrics used to summarize the
        predictions.
      label_metrics: List of Keras metrics used to summarize the labels.
      loss_metrics: List of Keras metrics used to summarize the loss.
      name: Optional task name.
    """

    super().__init__(name=name)

    self._loss = (
        loss if loss is not None else tf.keras.losses.BinaryCrossentropy())
    self._ranking_metrics = metrics or []
    self._prediction_metrics = prediction_metrics or []
    self._label_metrics = label_metrics or []
    self._loss_metrics = loss_metrics or []

  def call(self,
           labels: tf.Tensor,
           predictions: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           training: bool = False,
           compute_metrics: bool = True) -> tf.Tensor:
    """Computes the task loss and metrics.

    Args:
      labels: Tensor of labels.
      predictions: Tensor of predictions.
      sample_weight: Tensor of sample weights.
      training: Indicator whether training or test loss is being computed.
      compute_metrics: Whether to compute metrics. Set this to False
        during training for faster training.

    Returns:
      loss: Tensor of loss values.
    """

    loss = self._loss(
        y_true=labels, y_pred=predictions, sample_weight=sample_weight)

    if not compute_metrics:
      return loss

    update_ops = []

    for metric in self._ranking_metrics:
      update_ops.append(metric.update_state(
          y_true=labels, y_pred=predictions, sample_weight=sample_weight))

    for metric in self._prediction_metrics:
      update_ops.append(
          metric.update_state(predictions, sample_weight=sample_weight))

    for metric in self._label_metrics:
      update_ops.append(
          metric.update_state(labels, sample_weight=sample_weight))

    for metric in self._loss_metrics:
      update_ops.append(
          metric.update_state(loss, sample_weight=sample_weight))

    # Custom metrics may not return update ops, unlike built-in
    # Keras metrics.
    update_ops = [x for x in update_ops if x is not None]

    with tf.control_dependencies(update_ops):
      return tf.identity(loss)
