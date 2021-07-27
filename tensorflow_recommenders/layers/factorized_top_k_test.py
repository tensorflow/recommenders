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
"""Tests for factorized top K layers."""

import itertools
import os

from typing import Any, Dict, Iterator

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers import factorized_top_k


def test_cases(
    k=(5, 10),
    batch_size=(3, 16),
    num_queries=(3, 15, 16),
    num_candidates=(1024, 128),
    indices_dtype=(np.str, None),
    use_exclusions=(True, False)) -> Iterator[Dict[str, Any]]:
  """Generates test cases.

  Generates all possible combinations of input arguments as test cases.

  Args:
    k: The number of candidates to retrieve.
    batch_size: The query batch size.
    num_queries: Number of queries.
    num_candidates: Number of candidates.
    indices_dtype: The type of indices.
    use_exclusions: Whether to test exclusions.

  Yields:
    Keyword argument dicts.
  """

  keys = ("k", "batch_size", "num_queries", "num_candidates", "indices_dtype",
          "use_exclusions")

  for values in itertools.product(k, batch_size, num_queries, num_candidates,
                                  indices_dtype, use_exclusions):
    yield dict(zip(keys, values))


class FactorizedTopKTestBase(tf.test.TestCase, parameterized.TestCase):

  def run_save_and_restore_test(self, layer, query, num):
    for _ in range(num):
      pre_serialization_results = layer(query)

    path = os.path.join(self.get_temp_dir(), "layer")
    layer.save(
        path, options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))
    restored = tf.keras.models.load_model(path)

    for _ in range(num):
      post_serialization_results = restored(tf.constant(query))

    self.assertAllEqual(post_serialization_results, pre_serialization_results)

  def run_top_k_test(self,
                     layer_class,
                     k,
                     batch_size,
                     num_queries,
                     num_candidates,
                     indices_dtype,
                     use_exclusions,
                     random_seed=42,
                     check_export=True):

    layer = layer_class(k=k)

    rng = np.random.RandomState(random_seed)
    candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)

    candidate_indices = np.arange(num_candidates).astype(
        indices_dtype if indices_dtype is not None else np.int32)

    exclude = rng.randint(0, num_candidates, size=(num_queries, 5))

    scores = np.dot(query, candidates.T)

    # Set scores of candidates chosen for exclusion to a low value.
    adjusted_scores = scores.copy()
    if use_exclusions:
      exclude_identifiers = candidate_indices[exclude]
      for row_idx, row in enumerate(exclude):
        for col_idx in set(row):
          adjusted_scores[row_idx, col_idx] -= 1000.0
    else:
      exclude_identifiers = None

    # Get indices based on adjusted scores, but retain actual scores.
    indices = np.argsort(-adjusted_scores, axis=1)[:, :k]
    expected_top_scores = np.take_along_axis(scores, indices, 1)
    expected_top_indices = candidate_indices[indices]

    candidates = tf.data.Dataset.from_tensor_slices(candidates).batch(
        batch_size)

    if indices_dtype is not None:
      identifiers = tf.data.Dataset.from_tensor_slices(candidate_indices).batch(
          batch_size)
      candidates = tf.data.Dataset.zip((identifiers, candidates))

    # Call twice to ensure the results are repeatable.
    for _ in range(2):
      if use_exclusions:
        layer.index_from_dataset(candidates)
        top_scores, top_indices = layer.query_with_exclusions(
            query, exclude_identifiers)
      else:
        layer.index_from_dataset(candidates)
        top_scores, top_indices = layer(query)

    self.assertAllEqual(top_scores.shape, expected_top_scores.shape)
    self.assertAllEqual(top_indices.shape, expected_top_indices.shape)
    self.assertAllClose(top_scores, expected_top_scores)

    self.assertAllEqual(top_indices.numpy().astype(indices_dtype),
                        expected_top_indices)

    if not check_export:
      return

    # Save and restore to check export.
    path = os.path.join(self.get_temp_dir(), "layer")
    layer.save(
        path, options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]))
    restored = tf.keras.models.load_model(path)

    if use_exclusions:
      _, restored_top_indices = restored.query_with_exclusions(
          query, exclude_identifiers)
    else:
      _, restored_top_indices = restored(query)

    self.assertAllEqual(restored_top_indices.numpy().astype(indices_dtype),
                        expected_top_indices)

  @parameterized.parameters(test_cases())
  def test_streaming(self, *args, **kwargs):
    self.run_top_k_test(
        factorized_top_k.Streaming, *args, check_export=False, **kwargs)

  @parameterized.parameters(test_cases())
  def test_brute_force(self, *args, **kwargs):
    self.run_top_k_test(factorized_top_k.BruteForce, *args, **kwargs)

  @parameterized.parameters(np.str, np.float32, np.float64, np.int32, np.int64)
  def test_scann(self, identifier_dtype):

    num_candidates, num_queries = (1000, 4)

    rng = np.random.RandomState(42)
    candidates = rng.normal(size=(num_candidates, 4)).astype(np.float32)
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)
    candidate_names = np.arange(num_candidates).astype(identifier_dtype)

    scann = factorized_top_k.ScaNN()
    scann.index(candidates, candidate_names)

    self.run_save_and_restore_test(scann, query, 100)

  def test_scann_dataset_arg_no_identifiers(self):

    num_candidates, num_queries = (100, 4)

    rng = np.random.RandomState(42)
    candidates = tf.data.Dataset.from_tensor_slices(
        rng.normal(size=(num_candidates, 4)).astype(np.float32))
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)

    scann = factorized_top_k.ScaNN()
    scann.index_from_dataset(candidates.batch(100))

    self.run_save_and_restore_test(scann, query, 100)

  def test_scann_dataset_arg_with_identifiers(self):

    num_candidates, num_queries = (100, 4)

    rng = np.random.RandomState(42)
    candidates = tf.data.Dataset.from_tensor_slices(
        rng.normal(size=(num_candidates, 4)).astype(np.float32))
    query = rng.normal(size=(num_queries, 4)).astype(np.float32)
    identifiers = tf.data.Dataset.from_tensor_slices(np.arange(num_candidates))

    index = factorized_top_k.ScaNN()
    index.index_from_dataset(identifiers.zip(candidates).batch(100))

    self.run_save_and_restore_test(index, query, 100)

  @parameterized.parameters(factorized_top_k.ScaNN, factorized_top_k.BruteForce)
  def test_raise_on_incorrect_input_shape(
      self, layer_class: factorized_top_k.TopK):

    num_candidates = 100
    candidates = tf.data.Dataset.from_tensor_slices(
        np.random.normal(size=(num_candidates, 4)).astype(np.float32))
    identifiers = tf.data.Dataset.from_tensor_slices(
        np.arange(num_candidates - 1))

    with self.assertRaises(ValueError):
      index = layer_class()
      index.index_from_dataset(
          tf.data.Dataset.zip((identifiers.batch(20), candidates.batch(100)))
      )

  @parameterized.parameters(test_cases())
  def test_scann_top_k(self, k, batch_size, num_queries, num_candidates,
                       indices_dtype, use_exclusions):

    def scann(k):
      """Returns brute-force-like ScaNN for testing."""
      return factorized_top_k.ScaNN(
          k=k,
          num_leaves=1,
          num_leaves_to_search=1,
          num_reordering_candidates=num_candidates)

    self.run_top_k_test(scann, k, batch_size, num_queries, num_candidates,
                        indices_dtype, use_exclusions)


if __name__ == "__main__":
  tf.test.main()
