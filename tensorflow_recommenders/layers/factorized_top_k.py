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

# Lint as: python3
# pylint: disable=g-import-not-at-top
"""Layers for retrieving top K recommendations from factorized retrieval models."""

import abc
import contextlib
from typing import Dict, Optional, Text, Tuple, Union
import uuid

import tensorflow as tf

try:
  # ScaNN is an optional dependency, and might not be present.
  from scann import scann_ops
  _HAVE_SCANN = True
except ImportError:
  _HAVE_SCANN = False


@contextlib.contextmanager
def _wrap_batch_too_small_error(k: int):
  """Context manager that provides a more helpful error message."""

  try:
    yield
  except tf.errors.InvalidArgumentError as e:
    error_message = str(e)
    if "input must have at least k columns" in error_message:
      raise ValueError("Tried to retrieve k={k} top items, but the candidate "
                       "dataset batch size is too small. This may be because "
                       "your candidate batch size is too small or the last "
                       "batch of your dataset is too small. "
                       "To resolve this, increase your batch size, set the "
                       "drop_remainder argument to True when batching your "
                       "candidates, or set the handle_incomplete_batches "
                       "argument to True in the constructor. ".format(k=k))
    else:
      raise


def _take_along_axis(arr: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
  """Partial TF implementation of numpy.take_along_axis.

  See
  https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html
  for details.

  Args:
    arr: 2D matrix of source values.
    indices: 2D matrix of indices.

  Returns:
    2D matrix of values selected from the input.
  """

  row_indices = tf.tile(
      tf.expand_dims(tf.range(tf.shape(indices)[0]), 1),
      [1, tf.shape(indices)[1]])
  gather_indices = tf.concat(
      [tf.reshape(row_indices, (-1, 1)),
       tf.reshape(indices, (-1, 1))], axis=1)

  return tf.reshape(tf.gather_nd(arr, gather_indices), tf.shape(indices))


def _exclude(scores: tf.Tensor, identifiers: tf.Tensor, exclude: tf.Tensor,
             k: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Removes a subset of candidates from top K candidates.

  For each row of inputs excludes those candidates whose identifiers match
  any of the identifiers present in the exclude matrix for that row.

  Args:
    scores: 2D matrix of candidate scores.
    identifiers: 2D matrix of candidate identifiers.
    exclude: 2D matrix of identifiers to exclude.
    k: Number of candidates to return.

  Returns:
    Tuple of (scores, indices) of candidates after exclusions.
  """

  idents = tf.expand_dims(identifiers, -1)
  exclude = tf.expand_dims(exclude, 1)

  isin = tf.math.reduce_any(tf.math.equal(idents, exclude), -1)

  # Set the scores of the excluded candidates to a very low value.
  adjusted_scores = (scores - tf.cast(isin, tf.float32) * 1.0e5)

  k = tf.math.minimum(k, tf.shape(scores)[1])

  _, indices = tf.math.top_k(adjusted_scores, k=k)

  return _take_along_axis(scores,
                          indices), _take_along_axis(identifiers, indices)


class TopK(tf.keras.Model, abc.ABC):
  """Interface for top K layers.

  Implementers must provide the following two methods:

  1. `index`: takes a tensor of candidate embeddings and creates the retrieval
    index.
  2. `call`: takes a tensor of queries and returns top K candidates for those
    queries.
  """

  def __init__(self, k: int, **kwargs) -> None:
    """Initializes the base class."""

    super().__init__(**kwargs)
    self._k = k

  @abc.abstractmethod
  def index(
      self,
      candidates: Union[tf.Tensor, tf.data.Dataset],
      identifiers: Optional[Union[tf.Tensor,
                                  tf.data.Dataset]] = None) -> "TopK":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix (or dataset) of candidate embeddings.
      identifiers: Optional tensor (or dataset) of candidate identifiers. If
        given, these will be used to as identifiers of top candidates returned
        when performing searches. If not given, indices into the candidates
        tensor will be given instead.

    Returns:
      Self.
    """

    raise NotImplementedError()

  @abc.abstractmethod
  def call(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      k: The number of candidates to retrieve. If not supplied, defaults to the
        `k` value supplied in the constructor.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
    """

    raise NotImplementedError()

  @tf.function
  def query_with_exclusions(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      exclusions: tf.Tensor,
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      exclusions: `[query_batch_size, num_to_exclude]` tensor of identifiers to
        be excluded from the top-k calculation. This is most commonly used to
        exclude previously seen candidates from retrieval. For example, if a
        user has already seen items with ids "42" and "43", you could set
        exclude to `[["42", "43"]]`.
      k: The number of candidates to retrieve. Defaults to constructor `k`
        parameter if not supplied.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
      ValueError if `queries` is not a tensor (after being passed through
        the query model).
    """

    # Ideally, `exclusions` would simply be an optional parameter to
    # `call`. However, Keras is unable to handle `call` signatures
    # that have more than one Tensor input parameter. The alternative
    # is to either pack all inputs into the first positional argument
    # (via tuples or dicts), or else have a separate method. We opt
    # for the second solution here. The ergonomics in either case aren't
    # great, but having two methods is simpler to explain.
    # See https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/
    # python/keras/engine/base_layer.py#L942 for details of why Keras
    # puts us in this predicament.

    k = k if k is not None else self._k

    adjusted_k = k + exclusions.shape[1]
    x, y = self(queries=queries, k=adjusted_k)
    return _exclude(x, y, exclude=exclusions, k=k)

  def _reset_tf_function_cache(self):
    """Resets the tf.function cache.

    We need to invalidate the compiled tf.function cache here. We just
    dropped some variables and created new ones. The concrete function is
    still referring to the old ones - and because it only holds weak
    references, this does not prevent the old variables being garbage
    collected. The end result is that it references dead objects.
    To resolve this, we throw away the existing tf.function object and
    create a new one.
    """

    if hasattr(self.query_with_exclusions, "python_function"):
      self.query_with_exclusions = tf.function(
          self.query_with_exclusions.python_function)


class Streaming(TopK):
  """Retrieves K highest scoring items and their ids from a large dataset.

  Used to efficiently retrieve top K query-candidate scores from a dataset,
  along with the top scoring candidates' identifiers.
  """

  def __init__(self,
               query_model: Optional[tf.keras.Model] = None,
               k: int = 10,
               handle_incomplete_batches: bool = True,
               num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
               sorted_order: bool = True) -> None:
    """Initializes the layer.

    Args:
      query_model: Optional Keras model for representing queries. If provided,
        will be used to transform raw features into query embeddings when
        querying the layer. If not provided, the layer will expect to be given
        query embeddings as inputs.
      k: Number of top scores to retrieve.
      handle_incomplete_batches: When True, candidate batches smaller than k
        will be correctly handled at the price of some performance. As an
        alternative, consider using the drop_remainer option when batching the
        candidate dataset.
      num_parallel_calls: Degree of parallelism when computing scores. Defaults
        to autotuning.
      sorted_order: If the resulting scores should be returned in sorted order.
        setting this to False may result in a small increase in performance.

    Raises:
      ValueError if candidate elements are not tuples.
    """

    super().__init__(k=k)

    self.query_model = query_model
    self._candidates = None
    self._identifiers = None
    self._handle_incomplete_batches = handle_incomplete_batches
    self._num_parallel_calls = num_parallel_calls
    self._sorted = sorted_order

    self._counter = self.add_weight("counter", dtype=tf.int32, trainable=False)

  def index(self,
            candidates: tf.data.Dataset,
            identifiers: Optional[tf.data.Dataset] = None) -> "Streaming":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix (or dataset) of candidate embeddings.
      identifiers: Optional tensor (or dataset) of candidate identifiers. If
        given, these will be used to as identifiers of top candidates returned
        when performing searches. If not given, indices into the candidates
        tensor will be given instead.

    Returns:
      Self.
    """

    self._candidates = candidates
    self._identifiers = identifiers

    return self

  def call(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Computes K highest scores and candidate indices for a given query.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      k: Number of elements to retrieve. If not set, will default to the k set
        in the constructor.

    Returns:
      Tuple of [query_batch_size, k] tensor of top scores for each query and
      [query_batch_size, k] tensor of indices for highest scoring candidates.

    Raises:
      ValueError if `index` has not been called.
    """

    k = k if k is not None else self._k

    if self._candidates is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    if self.query_model is not None:
      queries = self.query_model(queries)

    # Reset the element counter.
    self._counter.assign(0)

    def top_scores(candidate_index: tf.Tensor,
                   candidate_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Computes top scores and indices for a batch of candidates."""
      scores = tf.matmul(queries, candidate_batch, transpose_b=True)

      if self._handle_incomplete_batches:
        k_ = tf.math.minimum(k, tf.shape(scores)[1])
      else:
        k_ = k

      scores, indices = tf.math.top_k(scores, k=k_, sorted=self._sorted)

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

      if self._handle_incomplete_batches:
        k_ = tf.math.minimum(k, tf.shape(joined_scores)[1])
      else:
        k_ = k

      scores, indices = tf.math.top_k(joined_scores, k=k_, sorted=self._sorted)

      return scores, tf.gather(joined_indices, indices, batch_dims=1)

    # Initialize the state with dummy scores and candidate indices.
    index_dtype = self._identifiers.element_spec.dtype if self._identifiers is not None else tf.int32
    initial_state = (tf.zeros((tf.shape(queries)[0], 0), dtype=tf.float32),
                     tf.zeros((tf.shape(queries)[0], 0), dtype=index_dtype))

    def enumerate_rows(batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Enumerates rows in each batch using a total element counter."""

      starting_counter = self._counter.read_value()
      end_counter = self._counter.assign_add(tf.shape(batch)[0])

      return tf.range(starting_counter, end_counter), batch

    if self._identifiers is not None:
      dataset = tf.data.Dataset.zip((self._identifiers, self._candidates))
    else:
      dataset = self._candidates.map(enumerate_rows)

    with _wrap_batch_too_small_error(k):
      results = (
          dataset
          # Compute scores over all candidates, and select top k in each batch.
          # Each element is a ([query_batch_size, k] tensor,
          # [query_batch_size, k] tensor) of scores and indices (where query_
          # batch_size is the leading dimension of the input query embeddings).
          .map(top_scores, num_parallel_calls=self._num_parallel_calls)
          # Reduce into a single tuple of output tensors by keeping a running
          # tally of top k scores and indices.
          .reduce(initial_state, top_k))

    return results


class BruteForce(TopK):
  """Brute force retrieval."""

  def __init__(self,
               query_model: Optional[tf.keras.Model] = None,
               k: int = 10,
               name: Optional[Text] = None):
    """Initializes the layer.

    Args:
      query_model: Optional Keras model for representing queries. If provided,
        will be used to transform raw features into query embeddings when
        querying the layer. If not provided, the layer will expect to be given
        query embeddings as inputs.
      k: Default k.
      name: Name of the layer.
    """

    super().__init__(k=k, name=name)

    self.query_model = query_model

  def index(
      self,
      candidates: Union[tf.Tensor, tf.data.Dataset],
      identifiers: Optional[Union[tf.Tensor, tf.data.Dataset]] = None
  ) -> "BruteForce":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix (or dataset) of candidate embeddings.
      identifiers: Optional tensor (or dataset) of candidate identifiers. If
        given, these will be used to as identifiers of top candidates returned
        when performing searches. If not given, indices into the candidates
        tensor will be given instead.

    Raises:
      ValueError on incorrectly shaped inputs.

    Returns:
      Self.
    """

    if isinstance(candidates, tf.data.Dataset):
      candidates = tf.concat(list(candidates), axis=0)  # pytype: disable=wrong-arg-types

    if identifiers is None:
      identifiers = tf.range(candidates.shape[0])

    if isinstance(identifiers, tf.data.Dataset):
      identifiers = tf.concat(list(identifiers), axis=0)  # pytype: disable=wrong-arg-types

    if tf.rank(candidates) != 2:
      raise ValueError(
          f"The candidates tensor must be 2D (got {candidates.shape}).")

    if candidates.shape[0] != identifiers.shape[0]:
      raise ValueError(
          "The candidates and identifiers tensors must have the same number of rows "
          f"(got {candidates.shape[0]} candidates rows and {identifiers.shape[0]} "
          "identifier rows). "
      )

    # We need any value that has the correct dtype.
    identifiers_initial_value = tf.zeros((), dtype=identifiers.dtype)

    self._identifiers = self.add_weight(
        name="identifiers",
        dtype=identifiers.dtype,
        shape=identifiers.shape,
        initializer=tf.keras.initializers.Constant(
            value=identifiers_initial_value),
        trainable=False)
    self._candidates = self.add_weight(
        name="candidates",
        dtype=candidates.dtype,
        shape=candidates.shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=False)

    self._identifiers.assign(identifiers)
    self._candidates.assign(candidates)

    self._reset_tf_function_cache()

    return self

  def call(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      k: The number of candidates to retrieve. If not supplied, defaults to the
        k value supplied in the constructor.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
    """

    k = k if k is not None else self._k

    if self._candidates is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    if self.query_model is not None:
      queries = self.query_model(queries)

    scores = tf.linalg.matmul(queries, self._candidates, transpose_b=True)

    values, indices = tf.math.top_k(scores, k=k)

    return values, tf.gather(self._identifiers, indices)


class ScaNN(TopK):
  """ScaNN approximate retrieval index for a factorized retrieval model.

  This layer uses the state-of-the-art
  [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
  library to retrieve the best candidates for a given query.

  To understand how to use this layer effectively, have a look at the efficient
  retrieval
  [tutorial](https://www.tensorflow.org/recommenders/examples/efficient_serving).

  To deploy this layer in TensorFlow Serving you can use our customized
  TensorFlow Serving Docker container, available on
  [Docker Hub](https://hub.docker.com/r/google/tf-serving-scann). You can also
  build the image yourself from the
  [Dockerfile](https://github.com/google-research/google-research/tree/master/scann/tf_serving).
  """

  def __init__(self,
               query_model: Optional[tf.keras.Model] = None,
               k: int = 10,
               distance_measure: Text = "dot_product",
               num_leaves: int = 100,
               num_leaves_to_search: int = 10,
               dimensions_per_block: int = 2,
               num_reordering_candidates: Optional[int] = None,
               parallelize_batch_searches: bool = True,
               name: Optional[Text] = None):
    """Initializes the layer.

    Args:
      query_model: Optional Keras model for representing queries. If provided,
        will be used to transform raw features into query embeddings when
        querying the layer. If not provided, the layer will expect to be given
        query embeddings as inputs.
      k: Default number of results to retrieve. Can be overridden in `call`.
      distance_measure: Distance metric to use.
      num_leaves: Number of leaves.
      num_leaves_to_search: Number of leaves to search.
      dimensions_per_block: Controls the dataset compression ratio. A higher
        number results in greater compression, leading to faster scoring but
        less accuracy and more memory usage.
      num_reordering_candidates: If set, the index will perform a final
        refinement pass on `num_reordering_candidates` candidates after
        retrieving an initial set of neighbours. This helps improve accuracy,
        but requires the original representations to be kept, and so will
        increase the final model size."
      parallelize_batch_searches: Whether batch querying should be done in
        parallel.
      name: Name of the layer.

    Raises:
      ImportError: if the scann library is not installed.
    """

    super().__init__(k=k, name=name)

    if not _HAVE_SCANN:
      raise ImportError(
          "The scann library is not present. Please install it using "
          "`pip install scann` to use the ScaNN layer.")

    self.query_model = query_model
    self._k = k
    self._parallelize_batch_searches = parallelize_batch_searches
    self._num_reordering_candidates = num_reordering_candidates

    def build_searcher(candidates):
      builder = scann_ops.builder(
          db=candidates,
          num_neighbors=self._k,
          distance_measure=distance_measure)

      builder = builder.tree(
          num_leaves=num_leaves, num_leaves_to_search=num_leaves_to_search)
      builder = builder.score_ah(dimensions_per_block=dimensions_per_block)

      if self._num_reordering_candidates is not None:
        builder = builder.reorder(self._num_reordering_candidates)

      # Set a unique name to prevent unintentional sharing between
      # ScaNN instances.
      return builder.build(shared_name=f"{self.name}/{uuid.uuid4()}")

    self._build_searcher = build_searcher
    self._serialized_searcher = None

  def index(
      self,
      candidates: Union[tf.Tensor, tf.data.Dataset],
      identifiers: Optional[Union[tf.Tensor,
                                  tf.data.Dataset]] = None) -> "ScaNN":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix (or dataset) of candidate embeddings.
      identifiers: Optional tensor (or dataset) of candidate identifiers. If
        given, these will be used to as identifiers of top candidates returned
        when performing searches. If not given, indices into the candidates
        tensor will be given instead.

    Raises:
      ValueError on incorrectly shaped inputs.

    Returns:
      Self.
    """

    if isinstance(candidates, tf.data.Dataset):
      candidates = tf.concat(list(candidates), axis=0)  # pytype: disable=wrong-arg-types

    if identifiers is None:
      identifiers = tf.range(candidates.shape[0])

    if isinstance(identifiers, tf.data.Dataset):
      identifiers = tf.concat(list(identifiers), axis=0)  # pytype: disable=wrong-arg-type

    if len(candidates.shape) != 2:
      raise ValueError(
          f"The candidates tensor must be 2D (got {candidates.shape}).")

    if candidates.shape[0] != identifiers.shape[0]:
      raise ValueError(
          "The candidates and identifiers tensors must have the same number of rows "
          f"(got {candidates.shape[0]} candidates rows and {identifiers.shape[0]} "
          "identifier rows). "
      )

    self._serialized_searcher = self._build_searcher(
        candidates).serialize_to_module()

    # We need any value that has the correct dtype.
    identifiers_initial_value = tf.zeros((), dtype=identifiers.dtype)

    self._identifiers = self.add_weight(
        name="identifiers",
        dtype=identifiers.dtype,
        shape=identifiers.shape,
        initializer=tf.keras.initializers.Constant(
            value=identifiers_initial_value),
        trainable=False)
    self._candidates = self.add_weight(
        name="candidates",
        dtype=candidates.dtype,
        shape=candidates.shape,
        initializer=tf.keras.initializers.Zeros(),
        trainable=False)

    self._identifiers.assign(identifiers)
    self._candidates.assign(candidates)

    self._reset_tf_function_cache()

    return self

  def call(self,
           queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
           k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """Query the index.

    Args:
      queries: Query features. If `query_model` was provided in the constructor,
        these can be raw query features that will be processed by the query
        model before performing retrieval. If `query_model` was not provided,
        these should be pre-computed query embeddings.
      k: The number of candidates to retrieve. Defaults to constructor `k`
        parameter if not supplied.

    Returns:
      Tuple of (top candidate scores, top candidate identifiers).

    Raises:
      ValueError if `index` has not been called.
      ValueError if `queries` is not a tensor (after being passed through
        the query model) or is not rank 2.
    """

    k = k if k is not None else self._k

    if self._serialized_searcher is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    searcher = scann_ops.searcher_from_module(self._serialized_searcher,
                                             self._candidates)

    if self.query_model is not None:
      queries = self.query_model(queries)

    if not isinstance(queries, tf.Tensor):
      raise ValueError(f"Queries must be a tensor, got {type(queries)}.")

    if len(queries.shape) == 2:
      if self._parallelize_batch_searches:
        result = searcher.search_batched_parallel(
            queries, final_num_neighbors=k)
      else:
        result = searcher.search_batched(queries, final_num_neighbors=k)
      indices = result.indices
      distances = result.distances
    elif len(queries.shape) == 1:
      result = searcher.search(queries, final_num_neighbors=k)
      indices = result.index
      distances = result.distance
    else:
      raise ValueError(
          f"Queries must be of rank 2 or 1, got {len(queries.shape)}.")

    return distances, tf.gather(self._identifiers, indices)
