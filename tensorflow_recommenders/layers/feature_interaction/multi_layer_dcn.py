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

# Copyright 2023 The TensorFlow Recommenders Authors.
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

"""Implements `Cross` Layer, the cross layer in Deep & Cross Network (DCN)."""

from typing import Union, Text, Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class MultiLayerDCN(tf.keras.layers.Layer):
  """Cross Layer in Deep & Cross Network to learn explicit feature interactions.

  A layer that creates explicit and bounded-degree feature interactions
  efficiently. The `call` method accepts `inputs` as a tuple of size 2
  tensors. The first input `x0` is the base layer that contains the original
  features (usually the embedding layer); the second input `xi` is the output
  of the previous `Cross` layer in the stack, i.e., the i-th `Cross`
  layer. For the first `Cross` layer in the stack, x0 = xi.
  The output is x_{i+1} = x0 .* (W * xi + bias + diag_scale * xi) + xi,
  where .* designates elementwise multiplication, W could be a full-rank
  matrix, or a low-rank matrix U*V to reduce the computational cost, and
  diag_scale increases the diagonal of W to improve training stability (
  especially for the low-rank case).
  References:
      1. [R. Wang et al.](https://arxiv.org/pdf/2008.13535.pdf)
        See Eq. (1) for full-rank and Eq. (2) for low-rank version.
      2. [R. Wang et al.](https://arxiv.org/pdf/1708.05123.pdf)
  Example:
      ```python
      # after embedding layer in a functional model:
      input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
      x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6)
      x1 = MultiLayerDCN()(x0)
      x2 = MultiLayerDCN()(x0)
      logits = tf.keras.layers.Dense(units=10)(x2)
      model = tf.keras.Model(input, logits)
      ```
  Attributes:
      projection_dim: project dimension to reduce the computational cost. a
        low-rank matrix W = U*V will be used, where U is of size `input_dim` by
        `projection_dim` and V is of size `projection_dim` by `input_dim`.
        `projection_dim` need to be smaller than `input_dim`/2 to improve the
        model efficiency. In practice, we've observed that `projection_dim` =
        input_dim/4 consistently preserved the accuracy of a full-rank version.
      num_layers: the number of stacked DCN layers
      use_bias: whether to add a bias term for this layer. If set to False, no
        bias term will be used.
      kernel_initializer: Initializer to use on the kernel matrix.
      bias_initializer: Initializer to use on the bias vector.
      kernel_regularizer: Regularizer to use on the kernel matrix.
      bias_regularizer: Regularizer to use on bias vector.

  Input shape: A tuple of 2 (batch_size, `input_dim`) dimensional inputs.
  Output shape: A single (batch_size, `input_dim`) dimensional output.
  """

  def __init__(
      self,
      projection_dim: Optional[int] = 1,
      num_layers: Optional[int] = 3,
      use_bias: bool = True,
      kernel_initializer: Union[
          Text, tf.keras.initializers.Initializer] = "truncated_normal",
      bias_initializer: Union[Text,
                              tf.keras.initializers.Initializer] = "zeros",
      kernel_regularizer: Union[Text, None,
                                tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Union[Text, None,
                              tf.keras.regularizers.Regularizer] = None,
      **kwargs):

    super(MultiLayerDCN, self).__init__(**kwargs)

    self._projection_dim = projection_dim
    self._num_layers = num_layers
    self._use_bias = use_bias
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._input_dim = None

    self._supports_masking = True

  def build(self, input_shape):
    last_dim = input_shape[-1]
    self._dense_u_kernels, self._dense_v_kernels = [], []

    for _ in range(self._num_layers):
      self._dense_u_kernels.append(tf.keras.layers.Dense(
          self._projection_dim,
          kernel_initializer=_clone_initializer(self._kernel_initializer),
          kernel_regularizer=self._kernel_regularizer,
          use_bias=False,
          dtype=self.dtype,
      ))
      self._dense_v_kernels.append(tf.keras.layers.Dense(
          last_dim,
          kernel_initializer=_clone_initializer(self._kernel_initializer),
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          use_bias=self._use_bias,
          dtype=self.dtype,
      ))

    self.built = True

  def call(self, x0: tf.Tensor) -> tf.Tensor:
    """Computes the multi layer DCN feature cross.

    Args:
      x0: The input tensor
    Returns:
     Tensor of crosses.
    """
    if not self.built:
      self.build(x0.shape)

    xl = x0

    for i in range(self._num_layers):
      prod_output = self._dense_v_kernels[i](self._dense_u_kernels[i](xl))
      xl = x0 * prod_output + xl

    return xl

  def get_config(self):
    config = {
        "projection_dim":
            self._projection_dim,
        "num_layers":
            self._num_layers,
        "use_bias":
            self._use_bias,
        "kernel_initializer":
            tf.keras.initializers.serialize(self._kernel_initializer),
        "bias_initializer":
            tf.keras.initializers.serialize(self._bias_initializer),
        "kernel_regularizer":
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        "bias_regularizer":
            tf.keras.regularizers.serialize(self._bias_regularizer),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _clone_initializer(initializer):
  return initializer.__class__.from_config(initializer.get_config())
