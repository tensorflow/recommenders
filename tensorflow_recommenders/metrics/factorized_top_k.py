# Copyright 2022 The TensorFlow Recommenders Authors.
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

from typing import List, Optional, Sequence, Union

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
      ks: Sequence[int] = (1, 5, 10, 50, 100),
      name: str = "factorized_top_k",
  ) -> None:
    """Initializes the metric.

    Args:
      candidates: A layer for retrieving top candidates in response
        to a query, or a dataset of candidate embeddings from which
        candidates should be retrieved.
      ks: A sequence of values of `k` at which to perform retrieval evaluation.
      name: Optional name.
    """

    super().__init__(name=name)

    if isinstance(candidates, tf.data.Dataset):
      candidates = (
          layers.factorized_top_k.Streaming(k=max(ks))
          .index_from_dataset(candidates)
      )

    self._ks = ks
    self._candidates = candidates
    self._top_k_metrics = [
        tf.keras.metrics.Mean(
            name=f"{self.name}/top_{x}_categorical_accuracy"
        ) for x in ks
    ]

  def update_state(
      self,
      query_embeddings: tf.Tensor,
      true_candidate_embeddings: tf.Tensor,
      true_candidate_ids: Optional[tf.Tensor] = None
  ) -> tf.Operation:
    """Updates the metrics.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query embeddings.
      true_candidate_embeddings: [num_queries, embedding_dim] tensor of
        embeddings for candidates that were selected for the query.
      true_candidate_ids: Ids of the true candidates. If supplied, evaluation
        will be id-based: the supplied ids will be matched against the ids of
        the top candidates returned from the retrieval index, which should have
        been constructed with the appropriate identifiers.

        If not supplied, evaluation will be score-based: the score of the true
        candidate will be computed and compared with the scores returned from
        the index for the top candidates.

        Score-based evaluation is useful for when the true candidate is not
        in the retrieval index. Id-based evaluation is useful for when scores
        returned from the index are not directly comparable to scores computed
        by multiplying the candidate and embedding vector. For example, scores
        returned by ScaNN are quantized, and cannot be compared to
        full-precision scores.

    Returns:
      Update op. Only used in graph mode.
    """

    if true_candidate_ids is None and not self._candidates.is_exact():
      raise ValueError(
          f"The candidate generation layer ({self._candidates}) does not return "
          "exact results. To perform evaluation using that layer, you must "
          "supply `true_candidate_ids`, which will be checked against "
          "the candidate ids returned from the candidate generation layer."
      )

    positive_scores = tf.reduce_sum(
        query_embeddings * true_candidate_embeddings, axis=1, keepdims=True)

    top_k_predictions, retrieved_ids = self._candidates(
        query_embeddings, k=max(self._ks))

    update_ops = []

    if true_candidate_ids is not None:
      # We're using ID-based evaluation.
      if len(true_candidate_ids.shape) == 1:
        true_candidate_ids = tf.expand_dims(true_candidate_ids, 1)

      # Deal with ScaNN using `NaN`-padding by converting its
      # `NaN` scores into minimum scores.
      nan_padding = tf.math.is_nan(top_k_predictions)
      top_k_predictions = tf.where(
          nan_padding,
          tf.ones_like(top_k_predictions) * tf.float32.min,
          top_k_predictions
      )

      # Check sortedness.
      is_sorted = (
          top_k_predictions[:, :-1] - top_k_predictions[:, 1:]
      )
      tf.debugging.assert_non_negative(
          is_sorted, message="Top-K predictions must be sorted."
      )

      # Check whether the true candidates were retrieved, accounting
      # for padding.
      ids_match = tf.cast(
          tf.math.logical_and(
              tf.math.equal(true_candidate_ids, retrieved_ids),
              tf.math.logical_not(nan_padding)
          ),
          tf.float32
      )

      for k, metric in zip(self._ks, self._top_k_metrics):
        # By slicing until :k we assume scores are sorted.
        # Clip to only count multiple matches once.
        match_found = tf.clip_by_value(
            tf.reduce_sum(ids_match[:, :k], axis=1, keepdims=True),
            0.0, 1.0
        )
        update_ops.append(metric.update_state(match_found))
    else:
      # Score-based evaluation.
      y_pred = tf.concat([positive_scores, top_k_predictions], axis=1)

      for k, metric in zip(self._ks, self._top_k_metrics):
        targets = tf.zeros(tf.shape(positive_scores)[0], dtype=tf.int32)
        top_k_accuracy = tf.math.in_top_k(
            targets=targets,
            predictions=y_pred,
            k=k
        )
        update_ops.append(metric.update_state(top_k_accuracy))

    return tf.group(update_ops)

  def reset_states(self) -> None:
    """Resets the metrics."""

    for metric in self.metrics:
      metric.reset_states()

  def result(self) -> List[tf.Tensor]:
    """Returns a list of metric results."""

    return [metric.result() for metric in self.metrics]
