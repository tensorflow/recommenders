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
"""A factorized retrieval task."""

from typing import Optional, Text

import tensorflow as tf

from tensorflow_recommenders import layers
from tensorflow_recommenders import metrics as tfrs_metrics
from tensorflow_recommenders.tasks import base


class Retrieval(tf.keras.layers.Layer, base.Task):
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

  def __init__(self,
               loss: Optional[tf.keras.losses.Loss] = None,
               metrics: Optional[tfrs_metrics.FactorizedTopK] = None,
               temperature: Optional[float] = None,
               num_hard_negatives: Optional[int] = None,
               name: Optional[Text] = None) -> None:
    """Initializes the task.

    Args:
      loss: Loss function. Defaults to
        `tf.keras.losses.CategoricalCrossentropy`.
      metrics: Object for evaluating top-K metrics over a
       corpus of candidates. These metrics measure how good the model is at
       picking the true candidate out of all possible candidates in the system.
       Note, because the metrics range over the entire candidate set, they are
       usually much slower to compute. Consider setting `compute_metrics=False`
       during training to save the time in computing the metrics.
      temperature: Temperature of the softmax.
      num_hard_negatives: If positive, the `num_hard_negatives` negative
        examples with largest logits are kept when computing cross-entropy loss.
        If larger than batch size or non-positive, all the negative examples are
        kept.
      name: Optional task name.
    """

    super().__init__(name=name)

    self._loss = loss if loss is not None else tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    self._factorized_metrics = metrics
    self._temperature = temperature
    self._num_hard_negatives = num_hard_negatives

  @property
  def factorized_metrics(self) -> Optional[tfrs_metrics.FactorizedTopK]:
    """The metrics object used to compute retrieval metrics."""

    return self._factorized_metrics

  @factorized_metrics.setter
  def factorized_metrics(self,
                         value: Optional[tfrs_metrics.FactorizedTopK]) -> None:
    """Sets factorized metrics."""

    self._factorized_metrics = value

  def call(self,
           query_embeddings: tf.Tensor,
           candidate_embeddings: tf.Tensor,
           sample_weight: Optional[tf.Tensor] = None,
           candidate_sampling_probability: Optional[tf.Tensor] = None,
           candidate_ids: Optional[tf.Tensor] = None,
           compute_metrics: bool = True) -> tf.Tensor:
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
      compute_metrics: Whether to compute metrics. Set this to False
        during training for faster training.

    Returns:
      loss: Tensor of loss values.
    """

    if isinstance(tf.distribute.get_strategy(), tf.distribute.TPUStrategy):
      candidate_embeddings = _cross_replica_concat(candidate_embeddings)
      if candidate_sampling_probability is not None:
        candidate_sampling_probability = _cross_replica_concat(
            candidate_sampling_probability
        )

    scores = tf.linalg.matmul(
        query_embeddings, candidate_embeddings, transpose_b=True)

    num_queries = tf.shape(scores)[0]
    num_candidates = tf.shape(scores)[1]

    labels = tf.eye(num_queries, num_candidates)

    if self._temperature is not None:
      scores = scores / self._temperature

    if candidate_sampling_probability is not None:
      scores = layers.loss.SamplingProbabilityCorrection()(
          scores, candidate_sampling_probability)

    if candidate_ids is not None:
      scores = layers.loss.RemoveAccidentalHits()(labels, scores, candidate_ids)

    if self._num_hard_negatives is not None:
      scores, labels = layers.loss.HardNegativeMining(self._num_hard_negatives)(
          scores,
          labels)

    loss = self._loss(y_true=labels, y_pred=scores, sample_weight=sample_weight)

    if not compute_metrics:
      return loss

    if not self._factorized_metrics:
      return loss

    update_op = self._factorized_metrics.update_state(query_embeddings,
                                                      candidate_embeddings)

    with tf.control_dependencies([update_op]):
      return tf.identity(loss)


def _cross_replica_concat(values: tf.Tensor) -> tf.Tensor:
  """Combine tensors, one from each TPU core, into a single concatenated tensor.

  The resulting tensor's elements are in the order of the IDs of the cores that
  contributed them, but offset so that the first element on each core is the one
  contributed by that core. On the ith core, out of N total, it would look like:
  tf.Tensor([
      values from core i,
      values from core i+1,
      ...
      values from core N,
      values from core 1,
      ...
      values from core i-1
  ]).

  Here is an example that is meant to run on 4 TPU cores:

  >>> resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="")
  >>> tf.config.experimental_connect_to_cluster(resolver)
  >>> tf.tpu.experimental.initialize_tpu_system(resolver)
  >>> strategy = tf.distribute.experimental.TPUStrategy(resolver)
  >>> data = np.array([
  ...     [0, 0, 0, 0],
  ...     [1, 1, 1, 1],
  ...     [2, 2, 2, 2],
  ...     [3, 3, 3, 3]
  ... ])
  >>> dataset = tf.data.Dataset.from_tensor_slices(data).repeat().batch(4)
  >>> dataset_iterator = iter(strategy.experimental_distribute_dataset(dataset))
  >>> distributed_values = next(dataset_iterator)
  >>> strategy.run(tf.function(_cross_replica_concat), (distributed_values,))
  PerReplica: {
      0: <tf.Tensor: shape=(4, 4), dtype=int64, numpy=array([
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3]
      ], dtype=int64)>,
      1: <tf.Tensor: shape=(4, 4), dtype=int64, numpy=array([
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3],
          [0, 0, 0, 0]
      ], dtype=int64)>,
      2: <tf.Tensor: shape=(4, 4), dtype=int64, numpy=array([
          [2, 2, 2, 2],
          [3, 3, 3, 3],
          [0, 0, 0, 0],
          [1, 1, 1, 1]
      ], dtype=int64)>,
      3: <tf.Tensor: shape=(4, 4), dtype=int64, numpy=array([
          [3, 3, 3, 3],
          [0, 0, 0, 0],
          [1, 1, 1, 1],
          [2, 2, 2, 2]
      ], dtype=int64)>
  }

  Args:
    values: The current TPU core's contribution to the concatenated tensor.

  Returns:
    A concatenated tensor that is made up of one tensor from each TPU core.
  """
  context = tf.distribute.get_replica_context()
  gathered = context.all_gather(values, axis=0)
  return tf.roll(
      gathered,
      -context.replica_id_in_sync_group * values.shape[0],
      axis=0
  )
