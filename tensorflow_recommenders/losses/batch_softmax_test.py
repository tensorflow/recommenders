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
"""Tests for loss BatchSoftmax."""

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.losses import batch_softmax


class LossTest(tf.test.TestCase):

  def test_batch_softmax(self):
    """Checks whether batch_softmax works."""
    predictions = np.array(((6.0, 3.0), (9.0, 5.0)))
    labels = np.identity(2)
    expected = (-tf.math.log(tf.math.sigmoid(3.0)) -
                tf.math.log(1 - tf.math.sigmoid(4.0)))

    actual = batch_softmax.BatchSoftmax()(labels, predictions)

    self.assertAllClose(
        expected.numpy(), actual.numpy(), rtol=1e-2, atol=1e-2)

  def test_batch_softmax_with_temp_and_example_weights(self):
    predictions = np.array(((6.0, 3.0), (9.0, 5.0)))
    labels = np.identity(2)
    example_weights = np.array((1.0, 2.0))
    expected = (-tf.math.log(tf.math.sigmoid(1.5)) -
                2.0 * tf.math.log(1 - tf.math.sigmoid(2.0)))
    actual = batch_softmax.BatchSoftmax(temperature=2.0)(
        labels, predictions, example_weights)
    self.assertAllClose(
        expected.numpy(), actual.numpy(), rtol=1e-2, atol=1e-2)

  def test_batch_softmax_with_temp_and_example_weights_and_reduction(self):
    predictions = np.array(((6.0, 3.0), (9.0, 5.0)))
    labels = np.identity(2)
    example_weights = np.array((1.0, 2.0))
    expected = (-tf.math.log(tf.math.sigmoid(1.5)) -
                2.0 * tf.math.log(1 - tf.math.sigmoid(2.0)))
    actual = batch_softmax.BatchSoftmax(
        temperature=2.0,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)(
            labels, predictions, example_weights)
    self.assertAllClose(
        expected.numpy() / 2, actual.numpy(), rtol=1e-2, atol=1e-2)

  def test_batch_softmax_with_sampling_probability_correction(self):
    predictions = np.array(((6.0, 3.0), (9.0, 5.0)))
    labels = np.identity(2)
    candidate_sampling_probability = np.array((1.0, 0.5))
    expected = (-tf.math.log(1.0 / (1.0 + 0.5 * tf.math.exp(-3.0))) -
                1.0 * tf.math.log(0.5 / (tf.math.exp(4.0) + 0.5)))
    actual = batch_softmax.BatchSoftmax()(
        labels,
        predictions,
        candidate_sampling_probability=candidate_sampling_probability)

    self.assertAllClose(
        expected, self.evaluate(actual), rtol=1e-2, atol=1e-2)

  def test_batch_softmax_with_remove_accidental_hits(self):
    predictions = np.array(((6.0, 6.0, 3.0), (6.0, 6.0, 5.0)))
    labels = np.eye(2, 3)
    expected = (-tf.math.log(1.0 / (1.0 + tf.math.exp(-3.0))) -
                tf.math.log(1.0 / (1.0 + tf.math.exp(-1.0))))
    actual = batch_softmax.BatchSoftmax()(
        labels, predictions, candidate_ids=np.array(["a", "a", "b"]))
    self.assertAllClose(
        expected, self.evaluate(actual), rtol=1e-2, atol=1e-2)

  def test_batch_softmax_with_hard_negative_mining(self):
    labels = np.eye(2, 3)
    weights = np.array([1.0, 2.0], dtype=np.float32)
    predictions = np.array([[4.0, 2.0, 2.0], [2.0, 0.0, 3.0]], dtype=np.float32)
    actual = batch_softmax.BatchSoftmax(
        temperature=2.0, num_hard_negatives=1)(labels, predictions, weights)
    loss_1 = 0 - tf.math.log(
        tf.math.exp(2.0) / (tf.math.exp(2.0) + tf.math.exp(1.0)))
    loss_2 = 0 - tf.math.log(1.0 / (1.0 + tf.math.exp(1.5)))
    expected = loss_1 + 2.0 * loss_2
    self.assertAllClose(
        expected.numpy(), actual.numpy(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
  tf.test.main()
