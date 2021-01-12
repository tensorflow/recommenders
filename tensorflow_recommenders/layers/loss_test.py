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
"""Tests for loss layers."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers import loss


class LossTest(tf.test.TestCase, parameterized.TestCase):
  """Loss layers tests."""

  @parameterized.parameters(42, 123, 8391, 12390, 1230)
  def test_hard_negative_mining(self, random_seed):
    """Test hard negative mining."""

    num_hard_negatives = 3
    shape = (2, 20)
    rng = np.random.RandomState(random_seed)

    logits = rng.uniform(size=shape).astype(np.float32)
    labels = rng.permutation(np.eye(*shape).T).T.astype(np.float32)

    out_logits, out_labels = loss.HardNegativeMining(num_hard_negatives)(logits,
                                                                         labels)
    out_logits, out_labels = out_logits.numpy(), out_labels.numpy()

    self.assertEqual(out_logits.shape[-1], num_hard_negatives + 1)

    # Logits for positives are always returned.
    self.assertAllClose((out_logits * out_labels).sum(axis=1),
                        (logits * labels).sum(axis=1))

    # Set the logits for the labels to be highest to ignore
    # the effect of labels.
    logits = logits + labels * 1000.0

    out_logits, out_labels = loss.HardNegativeMining(num_hard_negatives)(logits,
                                                                         labels)
    out_logits, out_labels = out_logits.numpy(), out_labels.numpy()

    # Highest K logits are always returned.
    self.assertAllClose(
        np.sort(logits, axis=1)[:, -num_hard_negatives - 1:],
        np.sort(out_logits))

  @parameterized.parameters(42, 123, 8391, 12390, 1230)
  def test_remove_accidental_hits(self, random_seed):

    shape = (2, 4)
    rng = np.random.RandomState(random_seed)

    logits = rng.uniform(size=shape).astype(np.float32)
    labels = rng.permutation(np.eye(*shape).T).T.astype(np.float32)
    candidate_ids = rng.randint(0, 3, size=shape[-1])

    out_logits = loss.RemoveAccidentalHits()(
        labels, logits, candidate_ids).numpy()

    # Logits of labels are unchanged.
    self.assertAllClose((out_logits * labels).sum(axis=1),
                        (logits * labels).sum(axis=1))

    for row_idx in range(shape[0]):

      row_positive_idx = np.argmax(labels[row_idx])
      positive_candidate_id = candidate_ids[row_positive_idx]

      for col_idx in range(shape[1]):

        same_candidate_as_positive = (
            positive_candidate_id == candidate_ids[col_idx])
        is_positive = col_idx == row_positive_idx

        if same_candidate_as_positive and not is_positive:
          # We zeroed the logits.
          self.assertAllClose(out_logits[row_idx, col_idx],
                              logits[row_idx, col_idx] + loss.MIN_FLOAT)
        else:
          # We left the logits unchanged.
          self.assertAllClose(out_logits[row_idx, col_idx], logits[row_idx,
                                                                   col_idx])


if __name__ == "__main__":
  tf.test.main()
