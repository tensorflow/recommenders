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
"""Tests for corpus layers."""
import itertools

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers import corpus


class CorpusTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(itertools.product(
      (5, 10),
      (3, 16),
      (3, 15, 16),
      # A batch size of 3 ensures the batches are smaller than k.
      (1024, 3),
      (42, 123, 256),
      (np.int64, np.str)
  ))
  def test_indexed_top_k(self, k, batch_size, num_queries,
                         num_candidates, random_seed, indices_dtype):

    rng = np.random.RandomState(random_seed)
    candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)
    candidate_indices = rng.normal(size=(num_candidates)).astype(indices_dtype)

    scores = np.dot(query, candidates.T)
    expected_top_scores = -np.sort(-scores, axis=1)[:, :k]
    indices = np.argsort(-scores, axis=1)[:, :k]
    expected_top_indices = candidate_indices[indices]

    candidate_dataset = tf.data.Dataset.from_tensor_slices(
        (candidate_indices, candidates)).batch(batch_size)

    top_scores, top_indices = corpus.DatasetIndexedTopK(
        candidates=candidate_dataset, k=k)(query)

    self.assertAllEqual(top_scores.shape, expected_top_scores.shape)
    self.assertAllEqual(top_indices.shape, expected_top_indices.shape)
    self.assertAllClose(top_scores, expected_top_scores)

    self.assertAllEqual(top_indices.numpy().astype(indices_dtype),
                        expected_top_indices)

  @parameterized.parameters(itertools.product(
      (5, 10),
      (3, 16),
      (2, 5),
      # A batch size of 3 ensures the batches are smaller than k.
      (1024, 3),
      (42, 123, 256),
  ))
  def test_top_k(self, k, batch_size, num_queries, num_candidates, random_seed):

    rng = np.random.RandomState(random_seed)
    candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)
    scores = np.dot(query, candidates.T)
    expected_top_scores = -np.sort(-scores, axis=1)[:, :k]

    candidate_dataset = (tf.data.Dataset.from_tensor_slices(candidates)
                         .batch(batch_size))

    top_scores = corpus.DatasetTopK(
        candidates=candidate_dataset, k=k)(query)

    self.assertAllEqual(top_scores.shape, expected_top_scores.shape)
    self.assertAllClose(top_scores, expected_top_scores)


if __name__ == "__main__":
  tf.test.main()
