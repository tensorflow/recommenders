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

"""A pre-built ranking model."""


import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from tensorflow_recommenders import models
from tensorflow_recommenders import tasks
from tensorflow_recommenders.layers import embedding


class DotInteraction(tf.keras.layers.Layer):
  """Dot interaction layer.

  See theory in the DLRM paper: https://arxiv.org/pdf/1906.00091.pdf,
  section 2.1.3. Sparse activations and dense activations are combined.
  Dot interaction is applied to a batch of input Tensors [e1,...,e_k] of the
  same dimension and the output is a batch of Tensors with all distinct pairwise
  dot products of the form dot(e_i, e_j) for i <= j if self self_interaction is
  True, otherwise dot(e_i, e_j) i < j.
  TODO(agagik): Move all layers to their own module.

  Attributes:
    self_interaction: Boolean indicating if features should self-interact.
      If it is True, then the diagonal enteries of the interaction matric are
      also taken.
    name: String name of the layer.
  """

  def __init__(self,
               self_interaction: bool = False,
               name: Optional[str] = None,
               **kwargs) -> None:
    self._self_interaction = self_interaction
    super().__init__(name=name, **kwargs)

  def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
    """Performs the interaction operation on the tensors in the list.

    The tensors represent dense and sparse features.
    Pre-condition: The tensors should all have the same shape.

    Args:
      inputs: List of features with shape [batch_size, feature_dim].

    Returns:
      activations: Tensor representing interacted features.
    """
    batch_size = tf.shape(inputs[0])[0]
    # concat_features shape: B,num_features,feature_width
    try:
      concat_features = tf.stack(inputs, axis=1)
    except ValueError as e:
      raise ValueError(f"Input tensors` dimensions must be equal, original"
                       f"error message: {e}")

    # Interact features, select lower-triangular portion, and re-shape.
    xactions = tf.matmul(concat_features, concat_features, transpose_b=True)
    ones = tf.ones_like(xactions)
    feature_dim = xactions.shape[-1]
    if self._self_interaction:
      # Selecting lower-triangular portion including the diagonal.
      lower_tri_mask = tf.linalg.band_part(ones, -1, 0)
      out_dim = feature_dim * (feature_dim + 1) // 2
    else:
      # Selecting lower-triangular portion not included the diagonal.
      upper_tri_mask = tf.linalg.band_part(ones, 0, -1)
      lower_tri_mask = ones - upper_tri_mask
      out_dim = feature_dim * (feature_dim - 1) // 2
    activations = tf.boolean_mask(xactions, lower_tri_mask)
    activations = tf.reshape(activations, (batch_size, out_dim))
    return activations


class MlpBlock(tf.keras.layers.Layer):
  """Constructs a sequential multi-layer perceptron (MLP) block."""

  def __init__(self,
               units: List[int],
               use_bias: bool = True,
               activation: Union[Callable[[tf.Tensor], tf.Tensor], str,
                                 None] = "relu",
               out_activation: Union[Callable[[tf.Tensor], tf.Tensor], str,
                                     None] = None,
               **kwargs) -> None:
    """Initializes the MLP layer.

    Args:
      units: Sequential list of layer sizes.
      use_bias: Whether to include a bias term.
      activation: Type of activation to use on all except the last layer.
      out_activation: Type of activation to use on last layer.
      **kwargs: Extra args passed to the Keras Layer base class.
    """

    super().__init__(**kwargs)

    self._layers = []

    for num_units in units[:-1]:
      self._layers.append(
          tf.keras.layers.Dense(
              num_units, activation=activation, use_bias=use_bias))
    self._layers.append(
        tf.keras.layers.Dense(
            units[-1], activation=out_activation, use_bias=use_bias))

  def call(self, x: tf.Tensor) -> tf.Tensor:
    for layer in self._layers:
      x = layer(x)

    return x


def _get_tpu_embedding_feature_config(
    vocab_sizes: List[int],
    embedding_dim: int,
    table_name_prefix: str = "embedding_table"):
  """Returns TPU embedding feature config.

  TODO(agagik) move to a separate module.

  Args:
    vocab_sizes: List of sizes of categories/id's in the table.
    embedding_dim: Embedding dimension.
    table_name_prefix: a prefix for embedding tables.
  """
  feature_config = {}

  for i, vocab_size in enumerate(vocab_sizes):
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dim,
        combiner="mean",
        initializer=tf.initializers.TruncatedNormal(
            mean=0.0, stddev=1 / math.sqrt(embedding_dim)),
        name=table_name_prefix + "_%s" %i)
    feature_config[str(i)] = tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config)

  return feature_config


class RankingModel(models.Model):
  """Keras model definition for the Ranking model.

  For DLRM model DotInteraction is used.

  Attributes:
    vocab_sizes: List of ints, vocab sizes of the sparse features.
    embedding_dim: Integer, the size of the embedding dimension.
    emb_optimizer: Optimizer to use for TPU embeddings. If it is None, Adam is
      used.
    bottom_stack: The `bottom_stack` layer is applied to dense features before
      feature interaction. If it is None, MLP with layer sizes [256, 64,
      embedding_dim] is used.
    feature_interaction: Feature interaction layer is applied to the
      `bottom_stack` output and sparse feature embeddings. If it is None
      DotInteraction layer is used.
    top_stack: The `top_stack` layer is applied to the `feature_interaction`
      output. The output of top_stack should be in the range [0, 1]. If it is
      None MLP with layer sizes [512, 256, 1] is used.
    task: The task the model should optimize for. Defaults to a
      `tfrs.tasks.Ranking` task with a binary cross-entropy loss, suitable
      for tasks like click prediction.
  """

  def __init__(
      self,
      vocab_sizes: List[int],
      embedding_dim: int = 16,
      emb_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
      bottom_stack: Optional[tf.keras.layers.Layer] = None,
      feature_interaction: Optional[tf.keras.layers.Layer] = None,
      top_stack: Optional[tf.keras.layers.Layer] = None,
      task: Optional[tasks.Task] = None) -> None:

    super().__init__()

    emb_feature_config = _get_tpu_embedding_feature_config(
        vocab_sizes=vocab_sizes,
        embedding_dim=embedding_dim)

    if not emb_optimizer:
      emb_optimizer = tf.keras.optimizers.Adam()

    self._tpu_embeddings = embedding.TPUEmbedding(
        emb_feature_config, emb_optimizer)

    self._bottom_stack = bottom_stack if bottom_stack else MlpBlock(
        units=[256, 64, embedding_dim], out_activation="relu")
    self._top_stack = top_stack if top_stack else MlpBlock(
        units=[512, 256, 1], out_activation="sigmoid")
    self._feature_interaction = (feature_interaction if feature_interaction
                                 else DotInteraction())

    if task is not None:
      self._task = task
    else:
      self._task = tasks.Ranking(
          loss=tf.keras.losses.BinaryCrossentropy(
              reduction=tf.keras.losses.Reduction.NONE
          ),
          metrics=[
              tf.keras.metrics.AUC(name="auc"),
              tf.keras.metrics.BinaryAccuracy(name="accuracy"),
          ],
          prediction_metrics=[
              tf.keras.metrics.Mean("prediction_mean"),
          ],
          label_metrics=[
              tf.keras.metrics.Mean("label_mean")
          ]
      )

  def compute_loss(self,
                   inputs: Tuple[Dict[str, tf.Tensor], tf.Tensor],
                   training: bool = False) -> tf.Tensor:
    """Defines the loss function.

    Args:
      inputs: A data structure of tensors of the following format:
        ({"dense_features": dense_tensor,
        "sparse_features": sparse_tensors}, label_tensor)
      training: Whether the model is in training mode.

    Returns:
      Loss tensor.
    """

    features, labels = inputs
    outputs = self(features, training=training)

    loss = self._task(labels, outputs)
    loss = tf.reduce_mean(loss)
    # Scales loss as the default gradients allreduce performs sum inside the
    # optimizer.
    scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync

    return scaled_loss

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Execute forward and backward pass, return loss.

    Args:
      inputs: Model function inputs (features and labels).

    Returns:
      loss: Scalar tensor.
    """
    dense_features = inputs["dense_features"]
    sparse_features = inputs["sparse_features"]

    sparse_embeddings = self._tpu_embeddings(sparse_features)
    # Combining a dictionary to a vector and squeezing dimension from
    # (batch_size, 1, emb) to (batch_size, emb)
    sparse_embeddings = tf.nest.flatten(sparse_embeddings)

    sparse_embedding_vecs = [
        tf.squeeze(sparse_embedding) for sparse_embedding in sparse_embeddings
    ]
    dense_embedding_vec = self._bottom_stack(dense_features)

    interaction_args = sparse_embedding_vecs + [dense_embedding_vec]
    interaction_output = self._feature_interaction(interaction_args)
    feature_interaction_output = tf.concat(
        [dense_embedding_vec, interaction_output], axis=1)

    prediction = self._top_stack(feature_interaction_output)

    return tf.reshape(prediction, [-1])

  @property
  def embedding_trainable_variables(self) -> Sequence[tf.Variable]:
    """Returns trainable variables from embedding tables.

    When training a recommendation model with embedding tables, sometimes it's
    preferable to use separate optimizers/learning rates for embedding
    variables and dense variables.
    `tfrs.experimental.optimizers.CompositeOptimizer` can be used to apply
    different optimizer to embedding variables and the remaining variables.
    """
    return self._tpu_embeddings.trainable_variables

  @property
  def dense_trainable_variables(self) -> Sequence[tf.Variable]:
    """Returns all trainable variables that are not embeddings."""
    dense_vars = []
    for layer in self.layers:
      if layer != self._tpu_embeddings:
        dense_vars.extend(layer.trainable_variables)
    return dense_vars
