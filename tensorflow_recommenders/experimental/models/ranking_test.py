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

# pylint: disable=g-long-lambda
"""Tests for tensorflow_recommenders.experimental.models.ranking_model."""

import itertools
import math

from typing import List, Dict

from absl.testing import parameterized

import tensorflow as tf

import tensorflow_recommenders as tfrs


def _get_tpu_embedding_feature_config(
    vocab_sizes: List[int],
    embedding_dim: int,
    table_name_prefix: str = "embedding_table"
) -> Dict[str, tf.tpu.experimental.embedding.FeatureConfig]:
  """Returns TPU embedding feature config.

  Args:
    vocab_sizes: List of sizes of categories/id's in the table.
    embedding_dim: Embedding dimension.
    table_name_prefix: A prefix for embedding tables.
  Returns:
    A dictionary of feature_name, FeatureConfig pairs.
  """
  feature_config = {}

  for i, vocab_size in enumerate(vocab_sizes):
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dim,
        combiner="mean",
        initializer=tf.initializers.TruncatedNormal(
            mean=0.0, stddev=1 / math.sqrt(embedding_dim)
        ),
        name=f"{table_name_prefix}_{i}"
    )
    feature_config[str(i)] = tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config)

  return feature_config


def _generate_synthetic_data(num_dense: int,
                             vocab_sizes: List[int],
                             dataset_size: int,
                             batch_size: int,
                             generate_weights: bool = False) -> tf.data.Dataset:
  dense_tensor = tf.random.uniform(
      shape=(dataset_size, num_dense), maxval=1.0, dtype=tf.float32)

  sparse_tensors = []
  for size in vocab_sizes:
    sparse_tensors.append(
        tf.random.uniform(
            shape=(dataset_size,), maxval=int(size), dtype=tf.int32))

  sparse_tensor_elements = {
      str(i): sparse_tensors[i] for i in range(len(sparse_tensors))
  }

  # The mean is in [0, 1] interval.
  dense_tensor_mean = tf.math.reduce_mean(dense_tensor, axis=1)

  sparse_tensors = tf.stack(sparse_tensors, axis=-1)
  sparse_tensors_mean = tf.math.reduce_sum(sparse_tensors, axis=1)
  # The mean is in [0, 1] interval.
  sparse_tensors_mean = tf.cast(sparse_tensors_mean, dtype=tf.float32)
  sparse_tensors_mean /= sum(vocab_sizes)
  # The label is in [0, 1] interval.
  label_tensor = (dense_tensor_mean + sparse_tensors_mean) / 2.0
  # Use the threshold 0.5 to convert to 0/1 labels.
  label_tensor = tf.cast(label_tensor + 0.5, tf.int32)

  if generate_weights:
    weights = tf.random.uniform(shape=(dataset_size, 1))

    input_elem = (
        {"dense_features": dense_tensor,
         "sparse_features": sparse_tensor_elements},
        label_tensor,
        weights
    )
  else:
    input_elem = (
        {"dense_features": dense_tensor,
         "sparse_features": sparse_tensor_elements},
        label_tensor,
    )

  dataset = tf.data.Dataset.from_tensor_slices(input_elem)

  return dataset.batch(batch_size, drop_remainder=True)


class RankingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          # Feature interaction layers.
          (
              tfrs.layers.feature_interaction.DotInteraction,
              lambda: tf.keras.Sequential([
                  tf.keras.layers.Concatenate(),
                  tfrs.layers.feature_interaction.Cross()
              ]),
          ),
          # Bottom stack.
          (lambda: None, lambda: tfrs.layers.blocks.MLP(units=[40, 16])),
          # Top stack.
          (lambda: None, lambda: tfrs.layers.blocks.MLP(
              units=[40, 20, 1], final_activation="sigmoid")),
          # Use weights.
          (True, False),
          # Size threshold.
          (None, -1, 20)))
  def test_ranking_model(self,
                         feature_interaction_layer,
                         bottom_stack,
                         top_stack,
                         use_weights=False,
                         size_threshold=10):
    """Tests a ranking model."""
    vocabulary_sizes = [30, 3, 26]

    embedding_feature_config = _get_tpu_embedding_feature_config(
        vocab_sizes=vocabulary_sizes, embedding_dim=16)
    optimizer = tf.keras.optimizers.Adam()

    model = tfrs.experimental.models.Ranking(
        embedding_layer=tfrs.experimental.layers.embedding.PartialTPUEmbedding(
            feature_config=embedding_feature_config,
            optimizer=optimizer,
            size_threshold=size_threshold),
        bottom_stack=bottom_stack(),
        feature_interaction=feature_interaction_layer(),
        top_stack=top_stack())
    model.compile(optimizer=optimizer, steps_per_execution=5)

    dataset = _generate_synthetic_data(
        num_dense=8,
        vocab_sizes=vocabulary_sizes,
        dataset_size=64,
        batch_size=16,
        generate_weights=use_weights)

    model.fit(
        dataset.repeat(), validation_data=dataset, epochs=1, steps_per_epoch=5)

    metrics_ = model.evaluate(dataset, return_dict=True)

    self.assertIn("loss", metrics_)
    self.assertIn("accuracy", metrics_)


if __name__ == "__main__":
  tf.test.main()
