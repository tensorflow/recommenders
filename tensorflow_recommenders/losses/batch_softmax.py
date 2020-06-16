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
"""Loss functions."""

from typing import Text, Optional

import tensorflow as tf

from tensorflow_recommenders.layers import loss


class BatchSoftmax(object):
  """Retrieval softmax loss."""

  def __init__(self,
               temperature: Optional[float] = None,
               num_hard_negatives: Optional[int] = None,
               reduction: Text = tf.keras.losses.Reduction.SUM) -> None:
    """Initializes the loss.

    Args:
      temperature: Temperature t used to rescale logits, i.e., s(i,j) / t.
      num_hard_negatives: If positive, the `num_hard_negatives` negative
        examples with largest logits are kept when computing cross-entropy loss.
        If larger than batch size or non-positive, all the negative examples are
        kept.
      reduction: Reduction applied to the loss. Valid values are `SUM` and
        `SUM_OVER_BATCH_SIZE`. Defaults is `SUM`.

    Raises:
      ValueError if an unsupported loss reduction is supplied.
    """

    self._temperature = temperature
    self._num_hard_negatives = num_hard_negatives
    self._reduction = reduction

    if self._reduction not in (tf.keras.losses.Reduction.SUM,
                               tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
      raise ValueError(f"Unsupported loss reduction: {self._reduction}.")

  def __call__(self,
               y_true: tf.Tensor,
               y_pred: tf.Tensor,
               sample_weight: Optional[tf.Tensor] = None,
               candidate_sampling_probability: Optional[tf.Tensor] = None,
               candidate_ids: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Computes the loss.

    Args:
      y_true: [num_queries] integer tensor of label indices.
      y_pred: [num_queries, num_candidates] tensor of logits.
      sample_weight: [batch_size] tensor of sample weights.
      candidate_sampling_probability: Optional tensor of candidate sampling
        probabilities. When given will be be used to correct the logits to
        reflect the sampling probability of negative candidates.
      candidate_ids: Optional tensor containing candidate ids. When given
        enables removing accidental hits of examples used as negatives. An
        accidental hit is defined as an candidate that is used as an in-batch
        negative but has the same id with the positive candidate.

    Returns:
      Loss scalar.
    """

    labels = y_true
    logits = y_pred

    if self._temperature is not None:
      logits = logits / self._temperature

    if candidate_sampling_probability is not None:
      logits = loss.SamplingProbablityCorrection()(
          logits, candidate_sampling_probability)

    if candidate_ids is not None:
      logits = loss.RemoveAccidentalHits()(labels, logits, candidate_ids)

    if self._num_hard_negatives is not None:
      logits, labels = loss.HardNegativeMining(self._num_hard_negatives)(logits,
                                                                         labels)

    losses = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)

    if sample_weight is None:
      sample_weight = tf.ones_like(losses)

    sum_of_loss = tf.reduce_sum(losses * tf.reshape(sample_weight, [-1]))

    if self._reduction == tf.keras.losses.Reduction.SUM:
      return sum_of_loss
    else:
      return sum_of_loss / tf.cast(tf.shape(losses)[0], sum_of_loss.dtype)
