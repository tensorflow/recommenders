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
"""Tests base model."""

import numpy as np
import tensorflow as tf

from tensorflow_recommenders import metrics
from tensorflow_recommenders import models
from tensorflow_recommenders import tasks


class ModelTest(tf.test.TestCase):

  def test_ranking_model(self):
    """Tests a simple ranking model."""

    class Model(models.Model):

      def __init__(self):
        super().__init__()
        self._dense = tf.keras.layers.Dense(1)
        self.task = tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])

      def call(self, inputs):
        return self._dense(inputs)

      def compute_loss(self, inputs, training=False):
        features, labels = inputs

        predictions = self(features)

        return self.task(predictions=predictions, labels=labels)

    data = tf.data.Dataset.from_tensor_slices(
        (np.random.normal(size=(10, 3)), np.ones(10)))

    model = Model()
    model.compile()
    model.fit(data.batch(2))
    metrics_ = model.evaluate(data.batch(2), return_dict=True)

    self.assertIn("loss", metrics_)
    self.assertIn("accuracy", metrics_)

  def test_factorized_model(self):
    """Tests a simple factorized retrieval model."""

    class Model(models.Model):

      def __init__(self, candidate_dataset):
        super().__init__()

        self.query_model = tf.keras.layers.Dense(16)
        self.candidate_model = tf.keras.layers.Dense(16)

        self.task = tasks.Retrieval(
            metrics=metrics.FactorizedTopK(
                candidates=candidate_dataset.map(self.candidate_model),
                k=5,
                metrics=[
                    tf.keras.metrics.TopKCategoricalAccuracy(
                        k=5, name="factorized_categorical_accuracy_at_5")
                ]))

      def compute_loss(self, inputs, training=False):
        query_features, candidate_features = inputs

        query_embeddings = self.query_model(query_features)
        candidate_embeddings = self.candidate_model(candidate_features)

        return self.task(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings)

    candidate_dataset = tf.data.Dataset.from_tensor_slices(
        np.random.normal(size=(10, 3)))
    data = tf.data.Dataset.from_tensor_slices((
        np.random.normal(size=(10, 3)),
        np.random.normal(size=(10, 3)),
    ))

    model = Model(candidate_dataset.batch(10))
    model.compile()
    model.fit(data.batch(2))
    metrics_ = model.evaluate(data.batch(2), return_dict=True)

    self.assertIn("loss", metrics_)
    self.assertIn("factorized_categorical_accuracy_at_5", metrics_)

  def test_multiask_model(self):
    """Test a joint ranking-retrieval model."""

    class Model(models.Model):

      def __init__(self, candidate_dataset):
        super().__init__()

        self.query_model = tf.keras.layers.Dense(16)
        self.candidate_model = tf.keras.layers.Dense(16)
        self.ctr_model = tf.keras.layers.Dense(1, activation="sigmoid")

        self.retrieval_task = tasks.Retrieval(
            metrics=metrics.FactorizedTopK(
                candidates=candidate_dataset.map(self.candidate_model),
                k=5,
                metrics=[
                    tf.keras.metrics.TopKCategoricalAccuracy(
                        k=5, name="factorized_categorical_accuracy_at_5")
                ]))
        self.ctr_task = tasks.Ranking(
            metrics=[tf.keras.metrics.AUC(name="ctr_auc")])

      def compute_loss(self, inputs, training):
        query_features, candidate_features, clicks = inputs

        query_embeddings = self.query_model(query_features)
        candidate_embeddings = self.candidate_model(candidate_features)

        pctr = self.ctr_model(
            tf.concat([query_features, candidate_features], axis=1))

        retrieval_loss = self.retrieval_task(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings)
        ctr_loss = self.ctr_task(predictions=pctr, labels=clicks)

        return retrieval_loss + ctr_loss

    candidate_dataset = tf.data.Dataset.from_tensor_slices(
        np.random.normal(size=(10, 3)))
    data = tf.data.Dataset.from_tensor_slices((
        np.random.normal(size=(10, 3)),
        np.random.normal(size=(10, 3)),
        np.random.randint(0, 2, size=10),
    ))

    model = Model(candidate_dataset.batch(10))
    model.compile()
    model.fit(data.batch(2))
    metrics_ = model.evaluate(data.batch(2), return_dict=True)

    self.assertIn("loss", metrics_)
    self.assertIn("factorized_categorical_accuracy_at_5", metrics_)
    self.assertIn("ctr_auc", metrics_)

  def test_regularization_losses(self):

    class Model(models.Model):

      def __init__(self):
        super().__init__()
        self._dense = tf.keras.layers.Dense(1)
        self.task = tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])

      def call(self, inputs):
        self.add_loss(1000.0)
        return self._dense(inputs)

      def compute_loss(self, inputs, training=False):
        features, labels = inputs

        predictions = self(features)

        return self.task(predictions=predictions, labels=labels)

    data = tf.data.Dataset.from_tensor_slices(
        (np.random.normal(size=(10, 3)), np.ones(10)))

    model = Model()
    model.compile()
    model.fit(data.batch(2))
    metrics_ = model.evaluate(data.batch(2), return_dict=True)

    self.assertIn("loss", metrics_)
    self.assertIn("accuracy", metrics_)

    self.assertEqual(metrics_["regularization_loss"], 1000.0)


if __name__ == "__main__":
  tf.test.main()
