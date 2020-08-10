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

# lint-as: python3
"""Corpus metrics."""

from typing import List, Optional, Sequence, Text

import tensorflow as tf

from tensorflow_recommenders import layers


class FactorizedTopK(tf.keras.metrics.Metric):
  """Computes top-K metrics for a factorized retrieval model.

  The metrics are computed across a corpus of candidates in a streaming manner,
  allowing metrics such as precision-at-k and recall-at-k to be computed over
  corpora of millions of candidates.

  The metrics:
  - top K categorical accuracy: how often the true candidate is in the top K
    candidates for a given query.
  """

  def __init__(
      self,
      candidates: tf.data.Dataset,
      metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None,
      k: int = 100,
      name: Text = "factorized_top_k",
  ) -> None:
    """Initializes the metric.

    Args:
      candidates: Dataset of candidate features. Elements of the dataset must be
        candidate embeddings.
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

    self._k = k
    self._candidates = candidates
    self._top_k_metrics = metrics

  def update_state(self, query_embeddings: tf.Tensor,
                   true_candidate_embeddings: tf.Tensor) -> None:
    """Updates the metrics.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query embeddings.
      true_candidate_embeddings: [num_queries, embedding_dim] tensor of
        embeddings for candidates that were selected for the query.
    """

    positive_scores = tf.reduce_sum(
        query_embeddings * true_candidate_embeddings, axis=1, keepdims=True)

    top_k_predictions = layers.corpus.DatasetTopK(
        candidates=self._candidates, k=self._k)(
            query_embeddings)

    y_true = tf.concat(
        [tf.ones(tf.shape(positive_scores)),
         tf.zeros_like(top_k_predictions)],
        axis=1)
    y_pred = tf.concat([positive_scores, top_k_predictions], axis=1)

    for metric in self._top_k_metrics:
      metric.update_state(y_true=y_true, y_pred=y_pred)

  def reset_states(self) -> None:
    """Resets the metrics."""

    for metric in self.metrics:
      metric.reset_states()

  def result(self) -> List[tf.Tensor]:
    """Returns a list of metric results."""

    return [metric.result() for metric in self.metrics]
