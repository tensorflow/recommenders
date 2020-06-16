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

# Lint as: python3
"""Layers operating on corpus datasets.."""

import contextlib

from typing import Tuple

import numpy as np
import tensorflow as tf


_FLOAT_MIN = np.finfo("float32").min


@contextlib.contextmanager
def _wrap_batch_too_small_error(k: int):
  """Context manager that provides a more helpful error message."""

  try:
    yield
  except tf.errors.InvalidArgumentError as e:
    error_message = str(e)
    if "input must have at least k columns" in error_message:
      raise ValueError(
          "Tried to retrieve k={k} top items, but the candidate "
          "dataset batch size is too small. This may be because "
          "your candidate batch size is too small or the last "
          "batch of your dataset is too small. "
          "To resolve this, increase your batch size, set the "
          "drop_remainder argument to True when batching your "
          "candidates, or set the handle_incomplete_batches "
          "argument to True in the DatasetTopK constructor. "
          .format(k=k)
      )


def _pad_scores_to_k(x: tf.Tensor, k: int) -> tf.Tensor:
  """Pad scores to K columns to ensure we can get K elements."""

  num_columns = tf.shape(x)[1]
  num_padding_entries = tf.clip_by_value(k - num_columns, 0, k)
  return tf.pad(x, [[0, 0], [0, num_padding_entries]],
                constant_values=_FLOAT_MIN)


def _pad_indices_to_k(x: tf.Tensor, k: int) -> tf.Tensor:
  """Pad indices to K elements."""
  num_rows = tf.shape(x)[0]
  num_padding_entries = tf.clip_by_value(k - num_rows, 0, k)
  return tf.pad(x, [[0, num_padding_entries]])


class DatasetTopK(tf.keras.layers.Layer):
  """Layer for retrieving K highest scoring candidates from a large dataset.

  Used to efficiently retrieve top K query-candidate scores from a dataset.
  This is particularly useful for evaluating retrieval models when we want to
  compare the score assigned to the true positive relative to all other items.

  If you also want to retrieve identifiers of top scoring candidates, use
  DatasetIndexedTopK.
  """

  def __init__(self,
               candidates: tf.data.Dataset,
               k: int = 10,
               handle_incomplete_batches: bool = True) -> None:
    """Initializes the layer.

    Args:
      candidates: Dataset of candidates. Elements have to be batch of
        [candidate_batch_size, embedding dimension] candidate embeddings.
      k: Number of top scores to retrieve.
      handle_incomplete_batches: When True, candidate batches smaller than k
        will be correctly handled at the price of some performance. As an
        alternative, consider using the drop_remainer option when batching
        the candidate dataset.
    """

    super(DatasetTopK, self).__init__()
    self._candidates = candidates
    self._k = k
    self._handle_incomplete_batches = handle_incomplete_batches

  def call(self, query_embeddings: tf.Tensor) -> tf.Tensor:
    """Computes K highest scores for a given user representation.

    Args:
      query_embeddings: [query_batch_size, embedding_dim] tensor of query
        embeddings.

    Returns:
      [query_batch_size, k] tensor of top scores for each query.
    """

    def top_scores(candidate_batch: tf.Tensor) -> tf.Tensor:
      """Computes top scores for a batch of candidates."""
      scores = tf.matmul(
          query_embeddings,
          candidate_batch,
          transpose_b=True)

      if self._handle_incomplete_batches:
        scores = _pad_scores_to_k(scores, self._k)

      scores, _ = tf.math.top_k(scores, k=self._k)

      return scores

    def top_k(state_scores: tf.Tensor, x_scores: tf.Tensor) -> tf.Tensor:
      """Reduction function.

      Returns top K scores from a combination of existing top K scores and new
      candidate scores.

      Args:
        state_scores: [query_batch_size, k] tensor of highest scores so far.
        x_scores: [query_batch_size, k] tensor of new scores.

      Returns:
        [query_batch_size, k] tensor of highest scores from state and x.
      """
      joined_scores = tf.concat([state_scores, x_scores], axis=1)
      scores, _ = tf.math.top_k(joined_scores, k=self._k)

      return scores

    def remove_sentinel_values(scores: tf.Tensor) -> tf.Tensor:
      """Removes all columns with the marker _FLOAT_MIN value."""
      is_not_sentinel = tf.logical_not(tf.math.is_inf(
          tf.math.reduce_sum(top_scores, axis=0)
      ))

      return tf.boolean_mask(scores, is_not_sentinel, axis=1)

    # Initialize the state.
    initial_state = tf.zeros((tf.shape(query_embeddings)[0], 0),
                             dtype=tf.float32)

    with _wrap_batch_too_small_error(self._k):
      top_scores = (
          self._candidates
          # Compute scores over all candidates, and select top k in each batch.
          # Each element is a ([query_batch_size, k] tensor,
          # [query_batch_size, k] tensor) of scores and indices (where query_
          # batch_size is the leading dimension of the input query embeddings).
          .map(top_scores)
          # Reduce into a single tuple of output tensors by keeping a running
          # tally of top k scores.
          .reduce(initial_state, top_k)
      )

    return remove_sentinel_values(top_scores)


class DatasetIndexedTopK(tf.keras.layers.Layer):
  """Retrieves K highest scoring items and their ids from a large dataset.

  Used to efficiently retrieve top K query-candidate scores from a dataset,
  along with the top scoring candidates' identifiers.

  If you do not need candidate identifiers, use DatasetTopK.
  """

  def __init__(self,
               candidates: tf.data.Dataset,
               k: int = 10,
               handle_incomplete_batches: bool = True) -> None:
    """Initializes the layer.

    Args:
      candidates: Dataset of candidates. Elements are be tuples of
        [candidate_batch_size] tensor of candidate indices, and [batch_size,
        embedding_dim] tensor of candidate embeddings. The indices can be used
        to identify the highest scoring elements; for example, for example, if
        item ids are supplied as the index tensor, the ids of the highest
        scoring elements will be returned along with their scores.
      k: Number of top scores to retrieve.
      handle_incomplete_batches: When True, candidate batches smaller than k
        will be correctly handled at the price of some performance. As an
        alternative, consider using the drop_remainer option when batching
        the candidate dataset.

    Raises:
      ValueError if candidate elements are not tuples.
    """

    super(DatasetIndexedTopK, self).__init__()

    if (not isinstance(candidates.element_spec, tuple)
        or len(candidates.element_spec) != 2):
      raise ValueError(
          "Dataset elements must be tuples of (index tensor, "
          "embedding tensor)."
      )
    self._candidates = candidates
    self._k = k
    self._handle_incomplete_batches = handle_incomplete_batches

  def call(self, query_embeddings: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes K highest scores and candidate indices for a given query.

    Args:
      query_embeddings: [query_batch_size, embedding_dim] tensor of query
        embeddings.

    Returns:
      Tuple of [query_batch_size, k] tensor of top scores for each query and
      [query_batch_size, k] tensor of indices for highest scoring candidates.
    """

    def top_scores(candidate_index: tf.Tensor,
                   candidate_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Computes top scores and indices for a batch of candidates."""
      scores = tf.matmul(
          query_embeddings,
          candidate_batch,
          transpose_b=True)

      if self._handle_incomplete_batches:
        scores = _pad_scores_to_k(scores, self._k)
        candidate_index = _pad_indices_to_k(candidate_index, self._k)

      scores, indices = tf.math.top_k(scores, k=self._k)

      return scores, tf.gather(candidate_index, indices)

    def top_k(state: Tuple[tf.Tensor, tf.Tensor],
              x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
      """Reduction function.

      Returns top K scores from a combination of existing top K scores and new
      candidate scores, as well as their corresponding indices.

      Args:
        state: tuple of [query_batch_size, k] tensor of highest scores so far
          and [query_batch_size, k] tensor of indices of highest scoring
          elements.
        x: tuple of [query_batch_size, k] tensor of new scores and
          [query_batch_size, k] tensor of new indices.

      Returns:
        Tuple of [query_batch_size, k] tensors of highest scores and indices
          from state and x.
      """
      state_scores, state_indices = state
      x_scores, x_indices = x

      joined_scores = tf.concat([state_scores, x_scores], axis=1)
      joined_indices = tf.concat([state_indices, x_indices], axis=1)

      scores, indices = tf.math.top_k(joined_scores, k=self._k)

      return scores, tf.gather(joined_indices, indices, batch_dims=1)

    def remove_sentinel_values(
        scores: tf.Tensor,
        indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Removes all columns with the marker _FLOAT_MIN value."""
      is_not_sentinel = tf.logical_not(tf.math.is_inf(
          tf.math.reduce_sum(scores, axis=0)
      ))

      return (
          tf.boolean_mask(scores, is_not_sentinel, axis=1),
          tf.boolean_mask(indices, is_not_sentinel, axis=1)
      )

    # Initialize the state with dummy scores and candidate indices.
    index_dtype = self._candidates.element_spec[0].dtype
    initial_state = (
        tf.zeros((tf.shape(query_embeddings)[0], 0),
                 dtype=tf.float32),
        tf.zeros((tf.shape(query_embeddings)[0], 0),
                 dtype=index_dtype)
    )

    with _wrap_batch_too_small_error(self._k):
      results = (
          self._candidates
          # Compute scores over all candidates, and select top k in each batch.
          # Each element is a ([query_batch_size, k] tensor,
          # [query_batch_size, k] tensor) of scores and indices (where query_
          # batch_size is the leading dimension of the input query embeddings).
          .map(top_scores)
          # Reduce into a single tuple of output tensors by keeping a running
          # tally of top k scores and indices.
          .reduce(initial_state, top_k)
      )

    return remove_sentinel_values(*results)
