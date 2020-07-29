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
"""Approximate retrieval layers."""

import abc

from typing import Dict, Optional, Text, Tuple, Union

import tensorflow as tf


class ANN(abc.ABC):
  """Interface for ANN layers.

  Implementers must provide the following two methods:

  1. `index`: takes a tensor of candidate embeddings and creates the retrieval
    index.
  2. `call`: takes a tensor of queries and returns top K candidates for those
    queries.
  """

  @abc.abstractmethod
  def index(
      self,
      candidates: Union[tf.Tensor, tf.data.Dataset],
      identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]] = None) -> None:
    """Builds the retrieval index.

    Args:
      candidates: Matrix (or dataset) of candidate embeddings.
      identifiers: Optional tensor (or dataset) of candidate identifiers. If
        given these will be return to identify top candidates when performing
        searches. If not given, indices into the candidates tensor will be
        given instead.
    """

    raise NotImplementedError()

  @abc.abstractmethod
  def call(
      self, queries: Union[tf.Tensor, Dict[Text, tf.Tensor]]
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
    """

    raise NotImplementedError()


class BruteForce(ANN, tf.keras.Model):
  """Brute force retrieval."""

  def __init__(
      self,
      query_model: Optional[tf.keras.Model] = None,
      name: Optional[Text] = None):
    """Initializes the layer.

    Args:
      query_model: Optional Keras model for representing queries. If provided,
        will be used to transform raw features into query embeddings when
        querying the layer. If not provided, the layer will expect to be given
        query embeddings as inputs.
      name: Name of the layer.
    """

    super().__init__(name=name)

    self.query_model = query_model

  def index(
      self,
      candidates: Union[tf.Tensor, tf.data.Dataset],
      identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]] = None) -> None:
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix (or dataset) of candidate embeddings.
      identifiers: Optional tensor (or dataset) of candidate identifiers. If
        given these will be return to identify top candidates when performing
        searches. If not given, indices into the candidates tensor will be
        given instead.
    """

    if identifiers is None:
      identifiers = tf.range(candidates.shape[0])

    if isinstance(candidates, tf.data.Dataset):
      candidates = tf.concat(list(candidates), axis=0)  # pytype: disable=wrong-arg-types

    if isinstance(identifiers, tf.data.Dataset):
      identifiers = tf.concat(list(identifiers), axis=0)  # pytype: disable=wrong-arg-types

    if tf.rank(candidates) != 2:
      raise ValueError(
          f"The candidates tensor must be 2D (got {candidates.shape}).")

    self._identifiers = self.add_weight(
        name="identifiers",
        dtype=identifiers.dtype,
        shape=identifiers.shape,
        initializer=tf.keras.initializers.Constant(value=""),
        trainable=False)
    self._candidates = self.add_weight(
        name="candidates",
        dtype=candidates.dtype,
        shape=candidates.shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=False)

    self._identifiers.assign(identifiers)
    self._candidates.assign(candidates)

  def call(
      self, queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      num_candidates: int = 10
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      num_candidates: The number of candidates to retrieve.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
    """

    if self._candidates is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    if self.query_model is not None:
      queries = self.query_model(queries)

    scores = tf.linalg.matmul(queries, self._candidates, transpose_b=True)

    values, indices = tf.math.top_k(scores, k=num_candidates)

    return values, tf.gather(self._identifiers, indices)
