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

"""Embedding layer for the Ranking model."""

from typing import Dict, Optional, Union

import tensorflow as tf

from tensorflow_recommenders.layers.embedding.tpu_embedding_layer import TPUEmbedding

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class PartialTPUEmbedding(tf.keras.layers.Layer):
  """Partial TPU Embedding layer.

  This layer is composed  of `tfrs.layers.embedding.TPUEmbedding` and
  `tf.keras.layers.Embedding` embedding layers. When training on TPUs, it is
  preferable to use TPU Embedding layers for large tables (as they are sharded
  accross TPU cores) and Keras embedding layer for small tables.
  For tables with vocab sizes less than `size_threshold` a Keras embedding
  layer will be used, above that threshold a TPU embedding layer will be used.

  This layer will be applied on a dictionary of feature_name, categorical_tensor
  pairs and return a dictionary of string-to-tensor of feature_name,
  embedded_value pairs.
  """

  def __init__(self,
               feature_config,
               optimizer: tf.keras.optimizers.Optimizer,
               pipeline_execution_with_tensor_core: bool = False,
               batch_size: Optional[int] = None,
               size_threshold: Optional[int] = 10_000) -> None:
    """Initializes the embedding layer.

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An optimizer used for TPU embeddings.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old with potential correctness drawbacks). Set to True
        for improved performance.
      batch_size: If set, this will be used as the global batch size and
        overrides the autodetection of the batch size from the layer's input.
        This is necesarry if all inputs to the layer's call are SparseTensors.
      size_threshold: A threshold for table sizes below which a Keras embedding
        layer is used, and above which a TPU embedding layer is used.
        Set `size_threshold=0` to use TPU embedding for all tables and
        `size_threshold=None` to use only Keras embeddings.
    """
    super().__init__()

    tpu_feature_config = {}
    table_to_keras_emb = {}
    self._keras_embedding_layers = {}

    for name, embedding_feature_config in feature_config.items():
      table_config = embedding_feature_config.table
      if size_threshold is not None and table_config.vocabulary_size > size_threshold:
         # TPUEmbedding layer.
        tpu_feature_config[name] = embedding_feature_config
        continue

      # Keras layer.
      # Multiple features can reuse the same table.
      if table_config not in table_to_keras_emb:
        table_to_keras_emb[table_config] = tf.keras.layers.Embedding(
            input_dim=table_config.vocabulary_size,
            output_dim=table_config.dim)
      self._keras_embedding_layers[name] = table_to_keras_emb[table_config]

    self._tpu_embedding = TPUEmbedding(
        tpu_feature_config, optimizer) if tpu_feature_config else None

  def call(self, inputs: Dict[str, Tensor]) -> Dict[str, tf.Tensor]:
    """Computes the output of the embedding layer.

    It expects a string-to-tensor (or SparseTensor/RaggedTensor) dict as input,
    and outputs a dictionary of string-to-tensor of feature_name, embedded_value
    pairs. Not that SparseTensor/RaggedTensor are only supported for
    TPUEmbedding and are not supported for Keras embeddings.

    Args:
      inputs: A string-to-tensor (or SparseTensor/RaggedTensor) dictionary.

    Returns:
      output: A dictionary of string-to-tensor of feature_name, embedded_value
    pairs.

    Raises:
      ValueError if non tf.Tensor is passed to a Keras embedding layer.
    """
    keras_emb_inputs = {
        key: val for key, val in inputs.items()
        if key in self._keras_embedding_layers
    }
    tpu_emb_inputs = {
        key: val for key, val in inputs.items()
        if key not in self._keras_embedding_layers
    }

    output = {}
    for key, val in keras_emb_inputs.items():
      if not isinstance(val, tf.Tensor):
        raise ValueError("Only tf.Tensor input is supported for Keras embedding"
                         f" layers, but got: {type(val)}")

      output[key] = self._keras_embedding_layers[key](val)

    if self._tpu_embedding:
      tpu_emb_output_dict = self._tpu_embedding(tpu_emb_inputs)  # pylint: disable=[not-callable]
      output.update(tpu_emb_output_dict)
    return output

  @property
  def tpu_embedding(self) -> Optional[TPUEmbedding]:
    """Returns TPUEmbedding or `None` if only Keras embeddings are used."""
    return self._tpu_embedding

  @property
  def keras_embedding_layers(self) -> Dict[str, tf.keras.layers.Embedding]:
    """Returns a dictionary mapping feature names to Keras embedding layers."""
    return self._keras_embedding_layers
