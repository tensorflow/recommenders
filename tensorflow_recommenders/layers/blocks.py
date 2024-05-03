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

"""Convenience blocks for building models."""

from typing import List, Optional

import tensorflow as tf

from tensorflow_recommenders import types


class MLP(tf.keras.layers.Layer):
  """Sequential multi-layer perceptron (MLP) block."""

  def __init__(
      self,
      units: List[int],
      use_bias: bool = True,
      activation: Optional[types.Activation] = "relu",
      final_activation: Optional[types.Activation] = None,
      **kwargs) -> None:
    """Initializes the MLP layer.

    Args:
      units: Sequential list of layer sizes.
      use_bias: Whether to include a bias term.
      activation: Type of activation to use on all except the last layer.
      final_activation: Type of activation to use on last layer.
      **kwargs: Extra args passed to the Keras Layer base class.
    """

    super().__init__(**kwargs)

    self._sublayers = []
    self._units = units
    self.use_bias = use_bias
    self.activation = activation
    self.final_activation = final_activation

    for num_units in units[:-1]:
      self._sublayers.append(
          tf.keras.layers.Dense(
              num_units, activation=activation, use_bias=use_bias))
    self._sublayers.append(
        tf.keras.layers.Dense(
            units[-1], activation=final_activation, use_bias=use_bias))



  # def get_uniform_initializer(self, bottom_dim):
  #   limit = tf.math.sqrt(1.0 / bottom_dim)
  #   return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

  def build(self, input_shape):
    # The first layer's bottom_dim comes from the input shape
    bottom_dim = input_shape[1]
    for _, num_units in enumerate(self._units[:-1]):
      self._sublayers.append(
          tf.keras.layers.Dense(
              num_units,
              activation=self.activation,
              use_bias=self.use_bias,
              kernel_initializer=tf.keras.initializers.HeUniform(),
              bias_initializer=tf.keras.initializers.RandomUniform(
                  minval=-tf.math.sqrt(1.0 / bottom_dim),
                  maxval=tf.math.sqrt(1.0 / bottom_dim),
                  seed=0
              ),
          )
      )
      bottom_dim = num_units  # Update bottom_dim for the next layer

    # Add the final layer
    self._sublayers.append(
        tf.keras.layers.Dense(
            self._units[-1],
            activation=self.final_activation,
            use_bias=self.use_bias,
            kernel_initializer=tf.keras.initializers.HeUniform(),
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-tf.math.sqrt(1.0 / bottom_dim),
                maxval=tf.math.sqrt(1.0 / bottom_dim),
                seed=0
            ),
        )
    )
    super().build(input_shape)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Performs the forward computation of the block."""
    for layer in self._sublayers:
      x = layer(x)

    return x
