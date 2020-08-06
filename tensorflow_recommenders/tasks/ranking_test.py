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
"""Tests ranking tasks."""

import math

import tensorflow as tf

from tensorflow_recommenders.tasks import ranking


class RankingTest(tf.test.TestCase):

  def test_task(self):

    task = ranking.Ranking(
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])

    predictions = tf.constant([[1], [0.3]], dtype=tf.float32)
    labels = tf.constant([[1], [1]], dtype=tf.float32)

    # Standard log loss formula.
    expected_loss = -(math.log(1) + math.log(0.3)) / 2.0
    expected_metrics = {"accuracy": 0.5}

    loss = task(predictions=predictions, labels=labels)
    metrics = {
        metric.name: metric.result().numpy() for metric in task.metrics
    }

    self.assertIsNotNone(loss)
    self.assertAllClose(expected_loss, loss)
    self.assertAllClose(expected_metrics, metrics)


if __name__ == "__main__":
  tf.test.main()
