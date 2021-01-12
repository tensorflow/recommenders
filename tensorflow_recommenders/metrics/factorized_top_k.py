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
"""Factorized retrieval top K metrics."""

from typing import List, Optional, Sequence, Text, Union

import tensorflow as tf

from tensorflow_recommenders import layers


class FactorizedTopK(tf.keras.layers.Layer):
  """Computes metrics for across top K candidates surfaced by a retrieval model.

  The default metric is top K categorical accuracy: how often the true candidate
   is in the top K candidates for a given query.
  """

  def __init__(
      self,
      candidates: Union[layers.factorized_top_k.TopK, tf.data.Dataset],
      metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None,
      k: int = 100,
      name: Text = "factorized_top_k",
  ) -> None:
    """Initializes the metric.

    Args:
      candidates: A layer for retrieving top candidates in response
        to a query, or a dataset of candidate embeddings from which
        candidates should be retrieved.
      metrics: The metrics to compute. If not supplied, will compute top-K
        categorical accuracy metrics.
      k: The number of top scoring candidates to retrieve for metric evaluation.
      name: Optional name.
    """

    super().__init__(name=name)

    if metrics is None:
      metrics = [
          tf.keras.metrics.TopKCategoricalAccuracy(
              k=x, name=f"{self.name}/top_{x}_categorical_accuracy")
          for x in [1, 5, 10, 50, 100]
      ]

    if isinstance(candidates, tf.data.Dataset):
      candidates = layers.factorized_top_k.Streaming(k=k).index(candidates)

    self._k = k
    self._candidates = candidates
    self._top_k_metrics = metrics

  def update_state(self, query_embeddings: tf.Tensor,
                   true_candidate_embeddings: tf.Tensor) -> tf.Operation:
    """Updates the metrics.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query embeddings.
      true_candidate_embeddings: [num_queries, embedding_dim] tensor of
        embeddings for candidates that were selected for the query.

    Returns:
      Update op. Only used in graph mode.
    """

    positive_scores = tf.reduce_sum(
        query_embeddings * true_candidate_embeddings, axis=1, keepdims=True)

    top_k_predictions, _ = self._candidates(query_embeddings, k=self._k)

    y_true = tf.concat(
        [tf.ones(tf.shape(positive_scores)),
         tf.zeros_like(top_k_predictions)],
        axis=1)
    y_pred = tf.concat([positive_scores, top_k_predictions], axis=1)

    update_ops = []

    for metric in self._top_k_metrics:
      update_ops.append(metric.update_state(y_true=y_true, y_pred=y_pred))

    return tf.group(update_ops)

  def reset_states(self) -> None:
    """Resets the metrics."""

    for metric in self.metrics:
      metric.reset_states()

  def result(self) -> List[tf.Tensor]:
    """Returns a list of metric results."""

    return [metric.result() for metric in self.metrics]
