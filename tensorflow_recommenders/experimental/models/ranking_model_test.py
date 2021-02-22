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

"""Tests for tensorflow_recommenders.experimental.models.ranking_model."""

from typing import List

import tensorflow as tf

from tensorflow_recommenders.experimental.models.ranking_model import DotInteraction
from tensorflow_recommenders.experimental.models.ranking_model import MlpBlock
from tensorflow_recommenders.experimental.models.ranking_model import RankingModel


def _generate_synthetic_data(num_dense: int,
                             vocab_sizes: List[int],
                             dataset_size: int,
                             is_training: bool,
                             batch_size: int) -> tf.data.Dataset:
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
  # Using the threshold 0.5 to convert to 0/1 labels.
  label_tensor = tf.cast(label_tensor + 0.5, tf.int32)

  input_elem = {
      "dense_features": dense_tensor,
      "sparse_features": sparse_tensor_elements
  }, label_tensor

  dataset = tf.data.Dataset.from_tensor_slices(input_elem)
  if is_training:
    dataset = dataset.repeat()

  return dataset.batch(batch_size, drop_remainder=True)


class RankingModelTest(tf.test.TestCase):

  def test_ranking_model(self):
    """Tests a ranking model."""
    optimizer = tf.keras.optimizers.Adam()

    vocab_sizes = [100, 26]

    model = RankingModel(
        vocab_sizes=vocab_sizes,
        embedding_dim=20,
        bottom_stack=MlpBlock(units_list=[100, 20], out_activation="relu"),
        feature_interaction=DotInteraction(),
        top_stack=MlpBlock(units_list=[40, 20, 1], out_activation="sigmoid"),
        emb_optimizer=optimizer,
    )
    model.compile(optimizer,
                  steps_per_execution=5)

    train_dataset = _generate_synthetic_data(
        num_dense=8,
        vocab_sizes=vocab_sizes,
        dataset_size=1024,
        is_training=True,
        batch_size=16)

    eval_dataset = _generate_synthetic_data(
        num_dense=8,
        vocab_sizes=vocab_sizes,
        dataset_size=256,
        is_training=False,
        batch_size=16)

    model.fit(train_dataset,
              epochs=2,
              steps_per_epoch=100)

    metrics_ = model.evaluate(eval_dataset, return_dict=True)

    self.assertIn("loss", metrics_)
    self.assertIn("accuracy", metrics_)


if __name__ == "__main__":
  tf.test.main()
