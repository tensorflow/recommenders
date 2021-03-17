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

"""Implements `Dot Interaction` Layer of DLRM model."""

from typing import List, Optional

import tensorflow as tf


class DotInteraction(tf.keras.layers.Layer):
  """Dot interaction layer.

  See theory in the DLRM paper: https://arxiv.org/pdf/1906.00091.pdf,
  section 2.1.3. Sparse activations and dense activations are combined.
  Dot interaction is applied to a batch of input Tensors [e1,...,e_k] of the
  same dimension and the output is a batch of Tensors with all distinct pairwise
  dot products of the form dot(e_i, e_j) for i <= j if self self_interaction is
  True, otherwise dot(e_i, e_j) i < j.

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

    The tensors represent as transformed dense features and embedded categorical
    features.
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
    except (ValueError, tf.errors.InvalidArgumentError) as e:
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

