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
"""Tests for ANN layers."""
import os
import tempfile

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers import ann


class AnnTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(np.str, np.float32, np.float64, np.int32, np.int64)
  def test_brute_force(self, identifier_dtype):

    num_candidates, num_queries = (1000, 4)

    rng = np.random.RandomState(42)
    candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)
    candidate_names = np.arange(num_candidates).astype(identifier_dtype)

    index = ann.BruteForce(query_model=lambda x: x)
    index.index(candidates, candidate_names)

    for _ in range(100):
      pre_serialization_results = index(query[:2])

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "query_model")
      index.save(path)
      loaded = tf.keras.models.load_model(path)

    for _ in range(100):
      post_serialization_results = loaded(tf.constant(query[:2]))

    self.assertAllEqual(post_serialization_results, pre_serialization_results)

  def test_brute_force_dataset_arg_no_identifiers(self):

    num_candidates = 100
    candidates = tf.data.Dataset.from_tensor_slices(
        np.random.normal(size=(num_candidates, 4)).astype(np.float32))

    index = ann.BruteForce()
    index.index(candidates.batch(100))

  def test_brute_force_dataset_arg_with_identifiers(self):

    num_candidates = 100
    candidates = tf.data.Dataset.from_tensor_slices(
        np.random.normal(size=(num_candidates, 4)).astype(np.float32))
    identifiers = tf.data.Dataset.from_tensor_slices(np.arange(num_candidates))

    index = ann.BruteForce()
    index.index(candidates.batch(100), identifiers)


if __name__ == "__main__":
  tf.test.main()
