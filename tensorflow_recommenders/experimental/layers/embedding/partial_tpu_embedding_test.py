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

"""Unit tests for data_pipeline."""

import math
from typing import Dict, List
from absl.testing import parameterized

import tensorflow as tf

import tensorflow_recommenders as tfrs


def _get_tpu_embedding_feature_config(
    vocab_sizes: List[int],
    embedding_dims: List[int]
) -> Dict[str, tf.tpu.experimental.embedding.FeatureConfig]:
  """Returns TPU embedding feature config.

  Args:
    vocab_sizes: List of sizes of categories/id's in the table.
    embedding_dims: Embedding dimensions.
  Returns:
    A dictionary of feature_name, FeatureConfig pairs.
  """
  assert len(vocab_sizes) == len(embedding_dims)
  feature_config = {}

  for i, vocab_size in enumerate(vocab_sizes):
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dims[i],
        combiner="mean",
        initializer=tf.initializers.TruncatedNormal(
            mean=0.0, stddev=1 / math.sqrt(embedding_dims[i])
        ),
        name=f"table_{i}"
    )
    feature_config[str(i)] = tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config)

  return feature_config


class PartialTPUEmbeddingTest(parameterized.TestCase, tf.test.TestCase):

  def test_embedding_layer(self):
    feature_config = _get_tpu_embedding_feature_config(
        vocab_sizes=[5, 20, 8, 9, 15], embedding_dims=[2, 4, 6, 8, 10])

    embedding_layer = tfrs.experimental.layers.embedding.PartialTPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.keras.optimizers.Adam(),
        size_threshold=10)

    tpu_embedding_tables = embedding_layer.tpu_embedding.embedding_tables
    keras_embedding_layers = embedding_layer.keras_embedding_layers

    self.assertLen(tpu_embedding_tables, 2)
    self.assertLen(keras_embedding_layers, 3)

    for tbl_config, weight in tpu_embedding_tables.items():
      print(tbl_config, weight)
      if "1" in tbl_config.name:
        self.assertEqual(tbl_config.vocabulary_size, 20)
        self.assertEqual(tbl_config.dim, 4)
      else:
        self.assertEqual(tbl_config.vocabulary_size, 15)
        self.assertEqual(tbl_config.dim, 10)

    self.assertEqual(keras_embedding_layers["0"].input_dim, 5)
    self.assertEqual(keras_embedding_layers["0"].output_dim, 2)
    self.assertEqual(keras_embedding_layers["2"].input_dim, 8)
    self.assertEqual(keras_embedding_layers["2"].output_dim, 6)
    self.assertEqual(keras_embedding_layers["3"].input_dim, 9)
    self.assertEqual(keras_embedding_layers["3"].output_dim, 8)

    output = embedding_layer({"0": 4, "1": 10, "2": 6, "3": 8, "4": 0})
    for key, val in output.items():
      self.assertEqual(val.shape, feature_config[key].table.dim)

  def test_all_keras_embedding(self):
    feature_config = _get_tpu_embedding_feature_config(
        vocab_sizes=[5, 20, 8, 9, 15], embedding_dims=[2, 4, 6, 8, 10])

    embedding_layer = tfrs.experimental.layers.embedding.PartialTPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.keras.optimizers.Adam(),
        size_threshold=None)

    self.assertIsNone(embedding_layer.tpu_embedding)
    keras_embedding_layers = embedding_layer.keras_embedding_layers

    self.assertLen(keras_embedding_layers, 5)

    output = embedding_layer({"0": 4, "1": 10, "2": 6, "3": 8, "4": 0})
    for key, val in output.items():
      self.assertEqual(val.shape, feature_config[key].table.dim)

  def test_all_tpu_embedding(self):
    feature_config = _get_tpu_embedding_feature_config(
        vocab_sizes=[5, 20, 8, 9, 15], embedding_dims=[2, 4, 6, 8, 10])
    embedding_layer = tfrs.experimental.layers.embedding.PartialTPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.keras.optimizers.Adam(),
        size_threshold=0)

    self.assertLen(embedding_layer.tpu_embedding.embedding_tables, 5)

    output = embedding_layer({"0": 4, "1": 10, "2": 6, "3": 8, "4": 0})
    for key, val in output.items():
      self.assertEqual(val.shape, feature_config[key].table.dim)


if __name__ == "__main__":
  tf.test.main()
