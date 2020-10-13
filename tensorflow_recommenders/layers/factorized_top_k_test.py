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
"""Tests for factorized top K layers."""
import itertools

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers import factorized_top_k


class FactorizedTopKTestBase(tf.test.TestCase, parameterized.TestCase):

  def run_top_k_test(self, layer_class, k, batch_size, num_queries,
                     num_candidates, random_seed, indices_dtype):

    layer = layer_class(k=k)

    rng = np.random.RandomState(random_seed)
    candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)

    if indices_dtype is not None:
      candidate_indices = rng.normal(
          size=(num_candidates)).astype(indices_dtype)
    else:
      candidate_indices = np.arange(num_candidates).astype(np.int64)

    scores = np.dot(query, candidates.T)
    expected_top_scores = -np.sort(-scores, axis=1)[:, :k]
    indices = np.argsort(-scores, axis=1)[:, :k]
    expected_top_indices = candidate_indices[indices]

    candidates = tf.data.Dataset.from_tensor_slices(candidates).batch(
        batch_size)

    if indices_dtype is not None:
      identifiers = tf.data.Dataset.from_tensor_slices(candidate_indices).batch(
          batch_size)
    else:
      identifiers = None

    # Call twice to ensure the results are repeatable.
    for _ in range(2):
      top_scores, top_indices = (layer.index(candidates, identifiers)(query))

    self.assertAllEqual(top_scores.shape, expected_top_scores.shape)
    self.assertAllEqual(top_indices.shape, expected_top_indices.shape)
    self.assertAllClose(top_scores, expected_top_scores)

    self.assertAllEqual(top_indices.numpy().astype(indices_dtype),
                        expected_top_indices)

  @parameterized.parameters(
      itertools.product(
          (5, 10),
          (3, 16),
          (3, 15, 16),
          # A batch size of 3 ensures the batches are smaller than k.
          (1024, 3),
          (42, 123, 256),
          (np.int64, np.str, None),
      ))
  def test_streaming(self, *args, **kwargs):
    self.run_top_k_test(factorized_top_k.Streaming, *args, **kwargs)

  @parameterized.parameters(
      itertools.product((5, 10), (3, 16), (3, 15, 16), (1024, 128),
                        (42, 123, 256), (np.int64, np.str, None)))
  def test_brute_force(self, *args, **kwargs):
    self.run_top_k_test(factorized_top_k.BruteForce, *args, **kwargs)


if __name__ == "__main__":
  tf.test.main()
