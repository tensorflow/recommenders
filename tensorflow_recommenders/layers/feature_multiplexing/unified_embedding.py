# Copyright 2025 The TensorFlow Recommenders Authors.
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

"""Unified Embedding Module.

This module implements the feature multiplexing framework introduced in
"Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML
Systems" by Coleman et al. using a regular embeding layer as the backend
embedding table format. The intended usage is to construct a
UnifiedEmbeddingConfig object that describes the number and size of the features
to be embedded together in a shared table, and then construct a UnifiedEmbedding
layer from the config.

Example:
```python
total_buckets = 100  # Number of embedding buckets, across 
num_tables = 3  # Number of backend tables to split the buckets across.
table_dimension = 16  # Dimension of each unified embedding chunk.

# Construct the configuration object.
embed_config = UnifiedEmbeddingConfig(
    buckets_per_table=total_buckets,
    dim_per_table=table_dimension,
    num_tables=num_tables,
    name="unified_table",
)

# Add some features to the config, with different table sizes.
embed_config.add_feature("movie_genre", 2)  # 2 chunks = 32 dimensions.
embed_config.add_feature("movie_id", 3)  # 3 chunks = 48 dimensions.
embed_config.add_feature("user_zip_code", 1)  # 1 chunk = 16 dimensions.

# Construct the embedding layer, which takes a feature dict as input and
# returns a list of embeddings, one for each feature in the config.
embed_layer = UnifiedEmbedding(embed_config, embed_optimizer)
```
"""

from typing import Any, Dict, Union

import tensorflow as tf
from tensorflow_recommenders.layers.embedding import tpu_embedding_layer


FeatureConfig = tf.tpu.experimental.embedding.FeatureConfig
Hashing = tf.keras.layers.Hashing
TableConfig = tf.tpu.experimental.embedding.TableConfig
TPUEmbedding = tpu_embedding_layer.TPUEmbedding
ValidTPUOptimizer = Union[
    tf.tpu.experimental.embedding.SGD,
    tf.tpu.experimental.embedding.Adagrad,
    tf.tpu.experimental.embedding.Adam,
    tf.tpu.experimental.embedding.FTRL,
]


class UnifiedEmbeddingConfig:
  """Unified Embedding Config."""

  def __init__(
      self,
      buckets_per_table: int,
      dim_per_table: int,
      num_tables: int,
      name: str,
      **kwargs,
  ):
    self._buckets_per_table = buckets_per_table
    self._dim_per_table = dim_per_table
    self._num_tables = num_tables
    self._current_table = 0
    self._num_features = 0
    self._name = name
    self._table_configs = [
        TableConfig(
            vocabulary_size=self._buckets_per_table,
            dim=self._dim_per_table,
            name=f"{self._name}_{i}",
            **kwargs,
        )
        for i in range(self._num_tables)
    ]
    # Store TPU embedding configs for each feature component (sub-feature).
    self._embed_configs = {}
    self._hashing_configs = {}

  def add_feature(self, name: str, num_chunks: int, **kwargs):
    """Add a categorical feature to the unified embedding config.

    Arguments:
      name: Feature name, used to feed inputs from a feature dict and to track
        the sub-components of the embedding.
      num_chunks: Integer number of chunks to use for the embedding. The final
        dimension will be num_chunks * dim_per_table.
      **kwargs: Arguments to pass through to the underlying FeatureConfig.
    """
    chunk_embed_configs = {}
    chunk_hashing_configs = {}
    for chunk_id in range(num_chunks):
      chunk_name = f"{self._name}_{name}_lookup_{chunk_id}"
      chunk_embed_config = FeatureConfig(
          table=self._table_configs[self._current_table],
          name=chunk_name,
          **kwargs,
      )
      chunk_embed_configs[chunk_name] = chunk_embed_config
      chunk_hashing_configs[chunk_name] = {
          "num_bins": self._buckets_per_table,
          "salt": [self._num_features, chunk_id],
      }
      self._current_table += 1
      self._current_table %= self._num_tables
    self._num_features += 1
    self._embed_configs[name] = chunk_embed_configs
    self._hashing_configs[name] = chunk_hashing_configs

  @property
  def embedding_config(self):
    return self._embed_configs

  @property
  def hashing_config(self):
    return self._hashing_configs


@tf.keras.utils.register_keras_serializable()
class UnifiedEmbedding(tf.keras.layers.Layer):
  """Post-processing layer to concatenate unified embedding components."""

  def __init__(
      self,
      config: UnifiedEmbeddingConfig,
      optimizer: ValidTPUOptimizer,
      **kwargs,
  ):
    super().__init__(**kwargs)
    if config.embedding_config:
      # Init is called with a blank config during serialization/deserialization.
      self._embedding_layer = TPUEmbedding(
          feature_config=config.embedding_config,
          optimizer=optimizer)

    self._hash_config = config.hashing_config
    self._hashing_layers = {}
    for name in self._hash_config:
      self._hashing_layers[name] = {}
      for component_name, component_params in self._hash_config[name].items():
        self._hashing_layers[name][component_name] = Hashing(**component_params)

  def get_config(self):
    config = super().get_config()
    config.update({
        "embed_layer": tf.keras.saving.serialize_keras_object(
            self._embedding_layer),
        "hash_config": self._hash_config,
    })
    return config

  @classmethod
  def from_config(cls, config: Dict[str, Any]) -> "UnifiedEmbedding":
    # The only parameters we need to re-construct the layer are hashing_config,
    # to rebuild the Hashing layers, and the serialized embed_layer. For the
    # other arguments to the initializer, we use empty "dummy" values.
    ue_config = UnifiedEmbeddingConfig(0, 0, 0, "")
    ue_config.hashing_config = config.pop("hashing_config")
    ue_config.hashing_config = {}
    embed_layer = tf.keras.saving.deserialize_keras_object(
        config.pop("embed_layer"))
    config["config"] = ue_config
    config["optimizer"] = None  # Optimizer is stored by the embed_layer.
    ue_layer = cls(**config)
    ue_layer._embedding_layer = embed_layer
    return ue_layer

  def call(self, features: Dict[str, tf.Tensor]):
    """Hash inputs, lookup embedding components, and concatenate the results.

    Args:
      features: Input feature values as a {feature name: Tensor} dictionary.
        The dictionary keys must contain all of the feature names in the
        UnifiedEmbeddingConfig. The dictionary may also contain other features,
        but these will be ignored in the output.

    Returns:
      A list of embeddings, sorted according to the order in which the features
        were added to the UnifiedEmbeddingConfig.
    """
    # 1. Hash the features using different hash layers.
    hashed_features = {}
    for name, hashing_layers in self._hashing_layers.items():
      hashed_features[name] = {}
      feature = features[name]
      for component_name, hashing_layer in hashing_layers.items():
        hashed_features[name][component_name] = hashing_layer(feature)
    # 2. Embed the features using the embedding layer.
    embed_features = self._embedding_layer(hashed_features)
    # 3. Concatenate the sub-components of each feature (in order).
    output_features = []
    for name in embed_features.keys():
      components = embed_features[name]
      component_values = [components[k] for k in sorted(components.keys())]
      embedding = tf.concat(component_values, axis=-1)
      output_features.append(embedding)
    return output_features
