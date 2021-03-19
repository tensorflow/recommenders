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


from typing import cast, Callable, Dict, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

from tensorflow_recommenders import models
from tensorflow_recommenders import tasks
from tensorflow_recommenders.layers import feature_interaction as feature_interaction_lib


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


class RankingModel(models.Model):
  """Keras model definition for the Ranking model.

  Layers of Ranking model can be customized as needed. For example we can use
  feature_interaction = tfrs.layers.feature_interaction.DotInteraction()
  for DLRM model and use feature_interaction = tf.keras.Sequential(
  [tf.keras.layers.Concatenate(), tfrs.layers.feature_interaction.Cross()])
  for DCN network.

  Attributes:
    embedding_layer: The embedding layer is applied to the categorical features.
      It expects a string-to-tensor (or SparseTensor/RaggedTensor) dictionary as
      an input, and outputs a dictionary of string-to-tensor of feature_name,
      embedded_value pairs.
      {feature_name_i: tensor_i} -> {feature_name_i: emb(tensor_i)}.
    bottom_stack: The `bottom_stack` layer is applied to dense features before
      feature interaction. If it is None, MLP with layer sizes [256, 64, 16] is
      used. For DLRM model, the output of bottom_stack should be of shape
      (batch_size, embedding dimension).
    feature_interaction: Feature interaction layer is applied to the
      `bottom_stack` output and sparse feature embeddings. If it is None,
      DotInteraction layer is used.
    top_stack: The `top_stack` layer is applied to the `feature_interaction`
      output. The output of top_stack should be in the range [0, 1]. If it is
      None, MLP with layer sizes [512, 256, 1] is used.
    task: The task which the model should optimize for. Defaults to a
      `tfrs.tasks.Ranking` task with a binary cross-entropy loss, suitable
      for tasks like click prediction.
  """

  def __init__(
      self,
      embedding_layer: tf.keras.layers.Layer,
      bottom_stack: Optional[tf.keras.layers.Layer] = None,
      feature_interaction: Optional[tf.keras.layers.Layer] = None,
      top_stack: Optional[tf.keras.layers.Layer] = None,
      task: Optional[tasks.Task] = None) -> None:

    super().__init__()

    self._embedding_layer = embedding_layer
    self._bottom_stack = bottom_stack if bottom_stack else MlpBlock(
        units=[256, 64, 16], out_activation="relu")
    self._top_stack = top_stack if top_stack else MlpBlock(
        units=[512, 256, 1], out_activation="sigmoid")
    self._feature_interaction = (feature_interaction if feature_interaction
                                 else feature_interaction_lib.DotInteraction())

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
                   inputs: Union[
                       # Tuple of (features, labels).
                       Tuple[
                           Dict[str, tf.Tensor],
                           tf.Tensor
                       ],
                       # Tuple of (features, labels, sample weights).
                       Tuple[
                           Dict[str, tf.Tensor],
                           tf.Tensor,
                           Optional[tf.Tensor]
                       ]
                   ],
                   training: bool = False) -> tf.Tensor:
    """Computes the loss and metrics of the model.

    Args:
      inputs: A data structure of tensors of the following format:
        ({"dense_features": dense_tensor,
          "sparse_features": sparse_tensors},
          label_tensor), or
        ({"dense_features": dense_tensor,
          "sparse_features": sparse_tensors},
          label_tensor,
          sample_weight tensor).
      training: Whether the model is in training mode.

    Returns:
      Loss tensor.

    Raises:
      ValueError if the the shape of the inputs is invalid.
    """

    # We need to work around a bug in mypy - tuple narrowing
    # based on length checks doesn't work.
    # See https://github.com/python/mypy/issues/1178 for details.
    if len(inputs) == 2:
      inputs = cast(
          Tuple[
              Dict[str, tf.Tensor],
              tf.Tensor
          ],
          inputs
      )
      features, labels = inputs
      sample_weight = None
    elif len(inputs) == 3:
      inputs = cast(
          Tuple[
              Dict[str, tf.Tensor],
              tf.Tensor,
              Optional[tf.Tensor],
          ],
          inputs
      )
      features, labels, sample_weight = inputs
    else:
      raise ValueError(
          "Inputs should either be a tuple of (features, labels), "
          "or a tuple of (features, labels, sample weights). "
          "Got a length {len(inputs)} tuple instead: {inputs}."
      )

    outputs = self(features, training=training)

    loss = self._task(labels, outputs, sample_weight=sample_weight)
    loss = tf.reduce_mean(loss)
    # Scales loss as the default gradients allreduce performs sum inside the
    # optimizer.
    return loss / tf.distribute.get_strategy().num_replicas_in_sync

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    """Executes forward and backward pass, returns loss.

    Args:
      inputs: Model function inputs (features and labels).

    Returns:
      loss: Scalar tensor.
    """
    dense_features = inputs["dense_features"]
    sparse_features = inputs["sparse_features"]

    sparse_embeddings = self._embedding_layer(sparse_features)
    # Combine a dictionary to a vector and squeeze dimension from
    # (batch_size, 1, emb) to (batch_size, emb).
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
    return self._embedding_layer.trainable_variables

  @property
  def dense_trainable_variables(self) -> Sequence[tf.Variable]:
    """Returns all trainable variables that are not embeddings."""
    dense_vars = []
    for layer in self.layers:
      if layer != self._embedding_layer:
        dense_vars.extend(layer.trainable_variables)
    return dense_vars
