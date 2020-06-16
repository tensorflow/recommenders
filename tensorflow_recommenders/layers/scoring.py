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
"""Scoring layers."""

import tensorflow as tf


class DotProductScorer(tf.keras.layers.Layer):
  """Computes predictions of a factorized model.

  The predictions are computed by taking the dot product of query and candidate
  embeddings.
  """

  def call(self, query_embeddings: tf.Tensor,
           candidate_embeddings: tf.Tensor) -> tf.Tensor:
    """Computes predictions.

    Args:
      query_embeddings: [query_batch_size, embedding_dim] embedding tensor.
      candidate_embeddings: [candidate_batch_size, embedding_dim] embedding
        tensor.

    Returns:
      scores: [query_batch_size, candidate_batch_size] tensor of scores.
    """

    return tf.matmul(query_embeddings, candidate_embeddings, transpose_b=True)
