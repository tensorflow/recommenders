# Copyright 2022 The TensorFlow Recommenders Authors.
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
"""Tests factorized top K metrics."""

import itertools

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders import layers
from tensorflow_recommenders import metrics


class FactorizedTopKTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          (layers.factorized_top_k.Streaming,
           layers.factorized_top_k.BruteForce,
           None),
          (True, False)
      )
  )
  def test_factorized_top_k(self, top_k_layer, use_candidate_ids):

    rng = np.random.RandomState(42)

    num_candidates, num_queries, embedding_dim = (100, 10, 4)

    candidate_ids = np.arange(0, num_candidates).astype(str)
    candidates = rng.normal(size=(num_candidates,
                                  embedding_dim)).astype(np.float32)

    query = rng.normal(size=(num_queries, embedding_dim)).astype(np.float32)

    true_candidate_indexes = rng.randint(0, num_candidates, size=num_queries)
    true_candidate_embeddings = candidates[true_candidate_indexes]
    true_candidate_ids = candidate_ids[true_candidate_indexes]

    candidate_scores = query @ candidates.T

    ks = [1, 5, 10, 50]

    candidates = tf.data.Dataset.from_tensor_slices(
        (candidate_ids, candidates)).batch(32)

    if top_k_layer is not None:
      candidates = top_k_layer().index_from_dataset(candidates)

    metric = metrics.FactorizedTopK(
        candidates=candidates,
        ks=ks
    )
    metric.update_state(
        query_embeddings=query,
        true_candidate_embeddings=true_candidate_embeddings,
        true_candidate_ids=true_candidate_ids if use_candidate_ids else None
    )

    for k, metric_value in zip(ks, metric.result()):
      in_top_k = tf.math.in_top_k(
          targets=true_candidate_indexes,
          predictions=candidate_scores,
          k=k)

      self.assertAllClose(metric_value, in_top_k.numpy().mean())

  @parameterized.parameters(
      layers.factorized_top_k.Streaming,
      layers.factorized_top_k.BruteForce,
      layers.factorized_top_k.ScaNN
  )
  def test_id_based_evaluation(self, layer):

    rng = np.random.default_rng(42)

    k = 100
    num_candidates, num_queries, embedding_dim = (1280, 128, 128)
    candidates = rng.normal(size=(num_candidates,
                                  embedding_dim)).astype(np.float32)
    queries = rng.normal(size=(num_queries, embedding_dim)).astype(np.float32)
    true_candidate_indices = rng.integers(
        0, num_candidates, size=num_queries).astype(np.int32)

    index = layer(k=k).index_from_dataset(
        tf.data.Dataset.from_tensor_slices(candidates).batch(32))

    metric = metrics.FactorizedTopK(
        candidates=index,
        ks=[k]
    )

    in_top_k = 0

    for query, true_candidate_idx in zip(queries, true_candidate_indices):

      metric.update_state(
          query.reshape(1, -1),
          candidates[true_candidate_idx].reshape(1, -1),
          np.array([true_candidate_idx])
      )

      top_scores, top_indices = index(query.reshape(1, -1))
      top_scores, top_indices = top_scores.numpy()[0], top_indices.numpy()[0]

      if true_candidate_idx in top_indices.tolist():
        in_top_k += 1

    expected_metric = in_top_k / num_queries

    self.assertEqual(metric.result()[0], expected_metric)


if __name__ == "__main__":
  tf.test.main()
