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
"""A factorized retrieval task."""

from typing import List, Optional, Text

import tensorflow as tf

from tensorflow_recommenders import layers
from tensorflow_recommenders import losses
from tensorflow_recommenders import metrics


class Retrieval(tf.keras.layers.Layer):
  """A factorized retrieval task.

  Recommender systems are often composed of two components:
  - a retrieval model, retrieving O(thousands) candidates from a corpus of
    O(millions) candidates.
  - a ranker model, scoring the candidates retrieved by the retrieval model to
    return a ranked shortlist of a few dozen candidates.

  This task defines models that facilitate efficient retrieval of candidates
  from large corpora by maintaining a two-tower, factorized structure: separate
  query and candidate representation towers, joined at the top via a lightweight
  scoring function.
  """

  def __init__(
      self,
      loss: Optional[tf.keras.losses.Loss] = None,
      batch_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
      corpus_metrics: Optional[metrics.corpus.FactorizedTopK] = None,
      name: Optional[Text] = None) -> None:
    """Initializes the task.

    Args:
      loss: Loss function. Defaults to `BatchSoftmaxLoss`.
      batch_metrics: List of Keras metrics to be evaluated on each
        batch. These can measure how good the model is at differentiating true
        candidates from negative candidates within a batch. These metrics are
        approximate, but can be computed quickly during training.
      corpus_metrics: Object for evaluating top-K metrics over a
       corpus of candidates. These metrics measure how good the model is at
       picking the true candidate out of all possible candidates in the system,
       and are a much better guide to model quality than batch metrics.
       However, because they range over the entire candidate set, they are
       usually much slower to compute.
      name: Optional task name.
    """

    super().__init__(name=name)

    self._loss = loss if loss is not None else losses.BatchSoftmax()

    self._batch_evaluation_metrics = batch_metrics or []
    self._corpus_evaluation_metrics = corpus_metrics

  def call(self,
           query_embeddings: tf.Tensor,
           candidate_embeddings: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           candidate_sampling_probability: Optional[tf.Tensor] = None,
           candidate_ids: Optional[tf.Tensor] = None,
           evaluate_corpus_metrics: bool = True) -> tf.Tensor:
    """Computes the task loss and metrics.

    The main argument are pairs of query and candidate embeddings: the first row
    of query_embeddings denotes a query for which the candidate from the first
    row of candidate embeddings was selected by the user.

    The task will try to maximize the affinity of these query, candidate pairs
    while minimizing the affinity between the query and candidates belonging
    to other queries in the batch.

    Args:
      query_embeddings: [num_queries, embedding_dim] tensor of query
        representations.
      candidate_embeddings: [num_queries, embedding_dim] tensor of candidate
        representations.
      sample_weight: [num_queries] tensor of sample weights.
      candidate_sampling_probability: Optional tensor of candidate sampling
        probabilities. When given will be be used to correct the logits to
        reflect the sampling probability of negative candidates.
      candidate_ids: Optional tensor containing candidate ids. When given
        enables removing accidental hits of examples used as negatives. An
        accidental hit is defined as an candidate that is used as an in-batch
        negative but has the same id with the positive candidate.
      evaluate_corpus_metrics: If true, corpus metrics will be computed. Because
        evaluating corpus metrics may be slow, consider disabling this
        in training.

    Returns:
      loss: Tensor of loss values.
    """

    scores = tf.linalg.matmul(
        query_embeddings, candidate_embeddings, transpose_b=True)

    num_queries = tf.shape(scores)[0]
    num_candidates = tf.shape(scores)[1]

    labels = tf.eye(num_queries, num_candidates)

    if candidate_sampling_probability is not None:
      scores = layers.loss.SamplingProbablityCorrection()(
          scores, candidate_sampling_probability)

    if candidate_ids is not None:
      scores = layers.loss.RemoveAccidentalHits()(labels, scores, candidate_ids)

    loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)

    for metric in self._batch_evaluation_metrics:
      metric.update_state(
          y_true=labels, y_pred=scores, sample_weight=sample_weight)

    if not self._corpus_evaluation_metrics:
      return loss

    if not evaluate_corpus_metrics:
      return loss

    self._corpus_evaluation_metrics.update_state(query_embeddings,
                                                 candidate_embeddings)

    return loss
