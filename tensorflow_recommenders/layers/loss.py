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
"""Layers related to loss computation."""
from typing import Tuple

import numpy as np
import tensorflow as tf

MAX_FLOAT = np.finfo(np.float32).max / 100.0
MIN_FLOAT = np.finfo(np.float32).min / 100.0


def _gather_elements_along_row(data: tf.Tensor,
                               column_indices: tf.Tensor) -> tf.Tensor:
  """Gathers elements from a 2D tensor given the column indices of each row.

  A more efficient way of gathering elements from 2D tensor than tf.gather_nd().
  First, gets the flat 1D indices to gather from. Then flattens the data to 1D
  and uses tf.gather() to generate 1D output and finnally reshapes the
  output back to 2D.

  Args:
    data: A [N, M] 2D `Tensor`.
    column_indices: A [N, K] 2D `Tensor` denoting for each row, the K column
      indices to gather elements from the data `Tensor`.

  Returns:
    A [N, K] `Tensor` including output elements gathered from data `Tensor`.

  Raises:
    ValueError: if the first dimensions of data and column_indices don't match.
  """
  with tf.control_dependencies(
      [tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]):
    num_row = tf.shape(data)[0]
    num_column = tf.shape(data)[1]
    num_gathered = tf.shape(column_indices)[1]
    row_indices = tf.tile(
        tf.expand_dims(tf.range(num_row), -1),
        [1, num_gathered])
    flat_data = tf.reshape(data, [-1])
    flat_indices = tf.reshape(
        row_indices * num_column + column_indices, [-1])
    return tf.reshape(
        tf.gather(flat_data, flat_indices), [num_row, num_gathered])


class HardNegativeMining(tf.keras.layers.Layer):
  """Transforms logits and labels to return hard negatives."""

  def __init__(self, num_hard_negatives: int) -> None:
    """Initializes the layer.

    Args:
      num_hard_negatives: How many hard negatives to return.
    """

    super(HardNegativeMining, self).__init__()
    self._num_hard_negatives = num_hard_negatives

  def call(self, logits: tf.Tensor,
           labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Filters logits and labels with per-query hard negative mining.

    The result will include logits and labels for num_hard_negatives
    negatives as well as the positive candidate.

    Args:
      logits: [batch_size, number_of_candidates] tensor of logits.
      labels: [batch_size, number_of_candidates] one-hot tensor of labels.

    Returns:
      logits: [batch_size, num_hard_negatives + 1] tensor of logits.
      labels: [batch_size, num_hard_negatives + 1] one-hot tensor of labels.
    """

    # Number of sampled logits, i.e, the number of hard negatives to be
    # sampled (k) + number of true logit (1) per query, capped by batch size.
    num_sampled = tf.minimum(self._num_hard_negatives + 1, tf.shape(logits)[1])
    # To gather indices of top k negative logits per row (query) in
    # logits, true logits need to be excluded. First replace the true
    # logits (corresponding to positive labels) with a large score value
    # and then select the top k + 1 logits from each
    # row so that selected indices include the indices of true logit + top k
    # negative logits. This approach is to avoid using inefficient
    # tf.boolean_mask() when excluding true logits.

    # For each query, get the indices of the logits which have the highest
    # k + 1 logit values, including the highest k negative logits and one true
    # logit.
    _, col_indices = tf.nn.top_k(
        logits + labels * MAX_FLOAT, k=num_sampled, sorted=False)

    # Gather sampled logits and corresponding labels.
    logits = _gather_elements_along_row(logits, col_indices)
    labels = _gather_elements_along_row(labels, col_indices)

    return logits, labels


class RemoveAccidentalHits(tf.keras.layers.Layer):
  """Zeroes the logits of accidental negatives."""

  def call(self, labels: tf.Tensor, logits: tf.Tensor,
           candidate_ids: tf.Tensor) -> tf.Tensor:
    """Zeros selected logits.

    For each row in the batch, zeros the logits of negative candidates that have
    the same id as the positive candidate in that row.

    Args:
      labels: [batch_size, num_candidates] one-hot labels tensor.
      logits: [batch_size, num_candidates] logits tensor.
      candidate_ids: [num_candidates] candidate identifiers tensor

    Returns:
      logits: Modified logits.
    """
    # A more principled way is to implement softmax_cross_entropy_with_logits
    # with a input mask. Here we approximate so by letting accidental hits
    # have extremely small logits (MIN_FLOAT) for ease-of-implementation.

    candidate_ids = tf.expand_dims(candidate_ids, 1)

    positive_indices = tf.math.argmax(labels, axis=1)
    positive_candidate_ids = tf.gather(candidate_ids, positive_indices)

    duplicate = tf.cast(
        tf.equal(positive_candidate_ids, tf.transpose(candidate_ids)),
        labels.dtype
    )
    duplicate = duplicate - labels

    return logits + duplicate * MIN_FLOAT


class SamplingProbablityCorrection(tf.keras.layers.Layer):
  """Sampling probability correction."""

  def __call__(self, logits: tf.Tensor,
               candidate_sampling_probability: tf.Tensor) -> tf.Tensor:
    """Corrects the input logits to account for candidate sampling probability."""

    return logits - tf.math.log(candidate_sampling_probability)
