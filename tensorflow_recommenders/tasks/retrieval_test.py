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
"""Tests retrieval tasks."""

import numpy as np
import tensorflow as tf

from tensorflow_recommenders import metrics
from tensorflow_recommenders.tasks import retrieval


def _sigmoid(x):
  return 1. / (1 + np.exp(-x))


class RetrievalTest(tf.test.TestCase):

  def test_task(self):

    query = tf.constant([[1, 2, 3], [2, 3, 4]], dtype=tf.float32)
    candidate = tf.constant([[1, 1, 1], [1, 1, 0]], dtype=tf.float32)
    candidate_dataset = tf.data.Dataset.from_tensor_slices(
        np.array([[0, 0, 0]] * 20, dtype=np.float32))

    task = retrieval.Retrieval(
        metrics=metrics.FactorizedTopK(
            candidates=candidate_dataset.batch(16),
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(
                    k=5, name="factorized_categorical_accuracy_at_5")
            ]))

    # All_pair_scores: [[6, 3], [9, 5]].
    # Normalized logits: [[3, 0], [4, 0]].
    expected_loss = -np.log(_sigmoid(3.0)) - np.log(1 - _sigmoid(4.0))
    expected_metrics = {
        "factorized_categorical_accuracy_at_5": 1.0,
    }

    loss = task(query_embeddings=query, candidate_embeddings=candidate)
    metrics_ = {
        metric.name: metric.result().numpy() for metric in task.metrics
    }

    self.assertIsNotNone(loss)
    self.assertAllClose(expected_loss, loss)
    self.assertAllClose(expected_metrics, metrics_)

  def test_task_graph(self):

    with tf.Graph().as_default():
      with tf.compat.v1.Session() as sess:
        query = tf.constant([[1, 2, 3], [2, 3, 4]], dtype=tf.float32)
        candidate = tf.constant([[1, 1, 1], [1, 1, 0]], dtype=tf.float32)
        candidate_dataset = tf.data.Dataset.from_tensor_slices(
            np.array([[0, 0, 0]] * 20, dtype=np.float32))

        task = retrieval.Retrieval(
            metrics=metrics.FactorizedTopK(
                candidates=candidate_dataset.batch(16),
                metrics=[
                    tf.keras.metrics.TopKCategoricalAccuracy(
                        k=5, name="factorized_categorical_accuracy_at_5")
                ]))

        expected_metrics = {
            "factorized_categorical_accuracy_at_5": 1.0,
        }

        loss = task(query_embeddings=query, candidate_embeddings=candidate)

        sess.run([var.initializer for var in task.variables])
        for metric in task.metrics:
          sess.run([var.initializer for var in metric.variables])
        sess.run(loss)

        metrics_ = {
            metric.name: sess.run(metric.result()) for metric in task.metrics
        }

        self.assertAllClose(expected_metrics, metrics_)


if __name__ == "__main__":
  tf.test.main()
