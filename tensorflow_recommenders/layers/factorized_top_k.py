# Copyright 2024 The TensorFlow Recommenders Authors.
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


def _check_candidates_with_identifiers(candidates: tf.data.Dataset) -> None:
  """Checks preconditions the dataset used for indexing."""

  spec = candidates.element_spec

  if isinstance(spec, tuple):
    if len(spec) != 2:
      raise ValueError(
          "The dataset must yield candidate embeddings or "
          "tuples of (candidate identifiers, candidate embeddings). "
          f"Got {spec} instead."
      )

    identifiers_spec, candidates_spec = spec

    if candidates_spec.shape[0] != identifiers_spec.shape[0]:
      raise ValueError(
          "Candidates and identifiers have to have the same batch dimension. "
          f"Got {candidates_spec.shape[0]} and {identifiers_spec.shape[0]}."
      )


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
      candidates: tf.Tensor,
      identifiers: Optional[tf.Tensor] = None) -> "TopK":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Matrix of candidate embeddings.
      identifiers: Optional tensor of candidate identifiers. If
        given, these will be used as identifiers of top candidates returned
        when performing searches. If not given, indices into the candidates
        tensor will be returned instead.

    Returns:
      Self.
    """

    raise NotImplementedError()

  def index_from_dataset(
      self,
      candidates: tf.data.Dataset
  ) -> "TopK":
    """Builds the retrieval index.

    When called multiple times the existing index will be dropped and a new one
    created.

    Args:
      candidates: Dataset of candidate embeddings or (candidate identifier,
        candidate embedding) pairs. If the dataset returns tuples,
        the identifiers will be used as identifiers of top candidates
        returned when performing searches. If not given, indices into the
        candidates dataset will be given instead.

    Returns:
      Self.

    Raises:
      ValueError if the dataset does not have the correct structure.
    """

    _check_candidates_with_identifiers(candidates)

    spec = candidates.element_spec

    if isinstance(spec, tuple):
      identifiers_and_candidates = list(candidates)
      candidates = tf.concat(
          [embeddings for _, embeddings in identifiers_and_candidates],
          axis=0
      )
      identifiers = tf.concat(
          [identifiers for identifiers, _ in identifiers_and_candidates],
          axis=0
      )
    else:
      candidates = tf.concat(list(candidates), axis=0)
      identifiers = None

    return self.index(candidates, identifiers)

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

  @abc.abstractmethod
  def is_exact(self) -> bool:
    """Indicates whether the results returned by the layer are exact.

    Some layers may return approximate scores: for example, the ScaNN layer
    may return approximate results.

    Returns:
      True if the layer returns exact results, and False otherwise.
    """

    raise NotImplementedError()

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

  def _compute_score(self, queries: tf.Tensor,
                     candidates: tf.Tensor) -> tf.Tensor:
    """Computes the standard dot product score from queries and candidates.

    Args:
      queries: Tensor of queries for which the candidates are to be retrieved.
      candidates: Tensor of candidate embeddings.

    Returns:
      The dot product of queries and candidates.
    """

    return tf.matmul(queries, candidates, transpose_b=True)


class Streaming(TopK):
  """Retrieves K highest scoring items and their ids from a large dataset.

  Used to efficiently retrieve top K query-candidate scores from a dataset,
  along with the top scoring candidates' identifiers.
  """

  def __init__(self,
               query_model: Optional[tf.keras.Model] = None,
               k: int = 10,
               handle_incomplete_batches: bool = True,
               num_parallel_calls: int = tf.data.AUTOTUNE,
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
    self._handle_incomplete_batches = handle_incomplete_batches
    self._num_parallel_calls = num_parallel_calls
    self._sorted = sorted_order

    self._counter = self.add_weight(name="counter", dtype=tf.int32, trainable=False)

  def index_from_dataset(
      self,
      candidates: tf.data.Dataset
  ) -> "TopK":

    _check_candidates_with_identifiers(candidates)

    self._candidates = candidates

    return self

  def index(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self,
      candidates: tf.data.Dataset,
      identifiers: Optional[tf.data.Dataset] = None) -> "Streaming":
    """Not implemented. Please call `index_from_dataset` instead."""

    raise NotImplementedError(
        "The streaming top k class only accepts datasets. "
        "Please call `index_from_dataset` instead."
    )

  def call(
      self,
      queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
      k: Optional[int] = None,
  ) -> Tuple[tf.Tensor, tf.Tensor]:

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

      scores = self._compute_score(queries, candidate_batch)

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

    def enumerate_rows(batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Enumerates rows in each batch using a total element counter."""

      starting_counter = self._counter.read_value()
      end_counter = self._counter.assign_add(tf.shape(batch)[0])

      return tf.range(starting_counter, end_counter), batch

    if not isinstance(self._candidates.element_spec, tuple):
      # We don't have identifiers.
      candidates = self._candidates.map(enumerate_rows)
      index_dtype = tf.int32
    else:
      candidates = self._candidates
      index_dtype = self._candidates.element_spec[0].dtype

    # Initialize the state with dummy scores and candidate indices.
    initial_state = (tf.zeros((tf.shape(queries)[0], 0), dtype=tf.float32),
                     tf.zeros((tf.shape(queries)[0], 0), dtype=index_dtype))

    with _wrap_batch_too_small_error(k):
      results = (
          candidates
          # Compute scores over all candidates, and select top k in each batch.
          # Each element is a ([query_batch_size, k] tensor,
          # [query_batch_size, k] tensor) of scores and indices (where query_
          # batch_size is the leading dimension of the input query embeddings).
          .map(top_scores, num_parallel_calls=self._num_parallel_calls)
          # Reduce into a single tuple of output tensors by keeping a running
          # tally of top k scores and indices.
          .reduce(initial_state, top_k))

    return results

  def is_exact(self) -> bool:
    return True


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
    self._candidates = None

  def index(
      self,
      candidates: tf.Tensor,
      identifiers: Optional[tf.Tensor] = None
  ) -> "BruteForce":

    if identifiers is None:
      identifiers = tf.range(candidates.shape[0])

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

    k = k if k is not None else self._k

    if self._candidates is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    if self.query_model is not None:
      queries = self.query_model(queries)

    scores = self._compute_score(queries, self._candidates)

    values, indices = tf.math.top_k(scores, k=k)

    return values, tf.gather(self._identifiers, indices)

  def is_exact(self) -> bool:
    return True


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
               training_iterations: int = 12,
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
      training_iterations: Number of training iterations when performing tree
        building.
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
    self._training_iterations = training_iterations
    self._identifiers = None

    def build_searcher(candidates):
      builder = scann_ops.builder(
          db=candidates,
          num_neighbors=self._k,
          distance_measure=distance_measure)

      builder = builder.tree(
          num_leaves=num_leaves,
          num_leaves_to_search=num_leaves_to_search,
          training_iterations=self._training_iterations,
      )
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
      candidates: tf.Tensor,
      identifiers: Optional[tf.Tensor] = None) -> "ScaNN":

    if len(candidates.shape) != 2:
      raise ValueError(
          f"The candidates tensor must be 2D (got {candidates.shape}).")

    if identifiers is not None and candidates.shape[0] != identifiers.shape[0]:
      raise ValueError(
          "The candidates and identifiers tensors must have the same number of rows "
          f"(got {candidates.shape[0]} candidates rows and {identifiers.shape[0]} "
          "identifier rows). "
      )

    self._serialized_searcher = self._build_searcher(
        candidates).serialize_to_module()

    if identifiers is not None:
      # We need any value that has the correct dtype.
      identifiers_initial_value = tf.zeros((), dtype=identifiers.dtype)
      self._identifiers = self.add_weight(
          name="identifiers",
          dtype=identifiers.dtype,
          shape=identifiers.shape,
          initializer=tf.keras.initializers.Constant(
              value=identifiers_initial_value),
          trainable=False)
      self._identifiers.assign(identifiers)

    self._reset_tf_function_cache()

    return self

  def call(self,
           queries: Union[tf.Tensor, Dict[Text, tf.Tensor]],
           k: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:

    k = k if k is not None else self._k

    if self._serialized_searcher is None:
      raise ValueError("The `index` method must be called first to "
                       "create the retrieval index.")

    searcher = scann_ops.searcher_from_module(self._serialized_searcher)

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

    if self._identifiers is None:
      return distances, indices

    return distances, tf.gather(self._identifiers, indices)

  def is_exact(self) -> bool:
    return False
