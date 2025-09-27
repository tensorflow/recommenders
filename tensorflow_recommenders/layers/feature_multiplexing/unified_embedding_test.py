# Copyright 2025 The TensorFlow Recommenders Authors.
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

"""Tests for UnifiedEmbedding class."""
import os
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers.feature_multiplexing import unified_embedding

UnifiedEmbedding = unified_embedding.UnifiedEmbedding
UnifiedEmbeddingConfig = unified_embedding.UnifiedEmbeddingConfig


class UnifiedEmbeddingTest(tf.test.TestCase):

  def setUp(self):

    super().setUp()

    self.dataset_size = 10
    rng = np.random.default_rng(seed=42)
    self.vocabs = {
        "genre": ["romance", "drama", "fantasy", "action", "comedy", "horror"],
        "year": [str(n) for n in range(1950, 2023)],
        "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "history": [f"Movie {n}" for n in range(1000)],
        "label": [0, 1],
    }
    # Construct a synthetic dataset for tests.
    n = self.dataset_size
    dense_hist_len = 4
    sparse_hist_len = (1, 10)
    self.dataset = {
        "genre": rng.choice(self.vocabs["genre"], size=n),
        "year": rng.choice(self.vocabs["year"], size=n),
        "city": rng.choice(self.vocabs["genre"], size=n),
        "num_watched": 100*(1.0 - rng.power(4, size=n)).astype(int),
        "history": rng.choice(self.vocabs["history"], size=[n, dense_hist_len]),
        "label": rng.choice(self.vocabs["label"], size=n),
    }
    for feature_name in self.dataset:
      self.dataset[feature_name] = tf.constant(self.dataset[feature_name])
    ragged_lens = rng.integers(sparse_hist_len[0], sparse_hist_len[1], size=n)
    ragged_history = [rng.choice(self.vocabs["history"], size=history_len)
                      for history_len in ragged_lens]
    self.dataset["history_varlen"] = tf.ragged.constant(ragged_history)

    self.default_optimizer = tf.tpu.experimental.embedding.SGD()

  def test_single_feature(self):
    config = UnifiedEmbeddingConfig(
        buckets_per_table=10,
        dim_per_table=8,
        num_tables=1,
        name="single_ue_table",
    )
    config.add_feature("genre", 2)
    ue_layer = UnifiedEmbedding(config, self.default_optimizer)
    ue_output = ue_layer(self.dataset)[0]
    self.assertAllEqual(ue_output.shape, [self.dataset_size, 16])

  def test_multiple_features(self):
    config = UnifiedEmbeddingConfig(
        buckets_per_table=10,
        dim_per_table=8,
        num_tables=3,
        name="multiple_ue_table",
    )
    config.add_feature("genre", 1)
    config.add_feature("year", 2)
    config.add_feature("city", 3)
    ue_layer = UnifiedEmbedding(config, self.default_optimizer)
    ue_outputs = ue_layer(self.dataset)
    correct_sizes = [
        [self.dataset_size, 8],
        [self.dataset_size, 16],
        [self.dataset_size, 24]
    ]
    for correct_size, embed in zip(correct_sizes, ue_outputs):
      self.assertAllEqual(embed.shape, correct_size)

  def test_dense_multivalent(self):
    # This should not call the "combiner" of the TPUEmbedding.
    config = UnifiedEmbeddingConfig(
        buckets_per_table=10,
        dim_per_table=8,
        num_tables=3,
        name="dense_multivalent_ue_table",
    )
    config.add_feature("history", 3)
    correct_size = list(self.dataset["history"].shape)
    correct_size.append(24)
    ue_layer = UnifiedEmbedding(config, self.default_optimizer)
    ue_output = ue_layer(self.dataset)[0]
    self.assertAllEqual(ue_output.shape, correct_size)

  def test_sparse_multivalent(self):
    # This does call the "combiner" of the TPUEmbedding.
    config = UnifiedEmbeddingConfig(
        buckets_per_table=10,
        dim_per_table=8,
        num_tables=3,
        name="sparse_multivalent_ue_table",
    )
    config.add_feature("history_varlen", 3)
    correct_size = [self.dataset_size, 24]
    ue_layer = UnifiedEmbedding(config, self.default_optimizer)
    ue_output = ue_layer(self.dataset)[0]
    self.assertAllEqual(ue_output.shape, correct_size)

  def test_feature_output_order(self):
    config = UnifiedEmbeddingConfig(
        buckets_per_table=10,
        dim_per_table=8,
        num_tables=3,
        name="reordered_ue_table",
    )
    config.add_feature("year", 2)
    config.add_feature("genre", 1)
    config.add_feature("city", 3)
    ue_layer = UnifiedEmbedding(config, self.default_optimizer)
    ue_output = ue_layer(self.dataset)
    correct_sizes = [
        [self.dataset_size, 16],
        [self.dataset_size, 8],
        [self.dataset_size, 24]
    ]
    for correct_size, embed in zip(correct_sizes, ue_output):
      self.assertAllEqual(embed.shape, correct_size)

  def test_save_model(self):
    config = UnifiedEmbeddingConfig(
        buckets_per_table=10,
        dim_per_table=8,
        num_tables=4,
        name="ue_table",
    )
    config.add_feature("year", 1)
    config.add_feature("city", 3)
    config.add_feature("genre", 2)

    model = tf.keras.Sequential([
        UnifiedEmbedding(config, self.default_optimizer),
        tf.keras.layers.Concatenate(axis=-1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    prediction = model.predict(self.dataset)
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "ue_model")
      model.save(path)
      loaded_model = tf.keras.models.load_model(path)
      loaded_prediction = loaded_model.predict(self.dataset)
      self.assertAllClose(prediction, loaded_prediction)

if __name__ == "__main__":
  tf.test.main()
