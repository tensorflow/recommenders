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

"""Clippy Adagrad optimizer implementation."""
from typing import Any, Dict, Sequence, Tuple, Union

import tensorflow as tf


def shrink_by_references(tensor: tf.Tensor, references: Sequence[tf.Tensor],
                         relative_factors: Sequence[float],
                         absolute_factor: float) -> Tuple[tf.Tensor, tf.Tensor]:
  """Shrinks a tensor such that it is element-wise smaller than a reference.

  Scales the given tensor such that for all index i
    |tensor_i| * scale <=
        sum_j |reference_i| * relative_factor_j + absolute_factor,
  where scale is the maximal scalar such that 0 < scale <= 1.

  Args:
    tensor: A Tensor to shrink.
    references: A sequence of Tensors in a shape broadcastable to `tensor`.
      Provides reference values for each coordinate in `tensor` for the
      shrinking calculation.
    relative_factors: A sequence of non-negative floats, used with
      absolute_factor to obtain the per-element shrinking values.
    absolute_factor: A non-negative float, used with relative_factors to obtain
      the per-element shrinking values.

  Returns:
    A tuple containing the scaled tensor (a tensor of the same shape as the
    given tensor) and a scalar scaling factor in [0, 1]. When absolute_factor is
    positive, the scaling factor will also be guaranteed to be positive.

  Raises:
    ValueError: If one of relative_factors is negative, absolute_factor is
    non-positive or the lengths of references and relative_factors lists are not
    equal.
  """
  if any(relative_factor < 0 for relative_factor in relative_factors):
    raise ValueError("relative_factors must all be non-negative.")
  if absolute_factor < 0:
    raise ValueError("absolute_factor must be non-negative.")
  if len(references) != len(relative_factors):
    raise ValueError(
        "references and relative_factors must have the same length. "
        f"Instead they are {len(references)} and {len(relative_factors)}.")
  max_delta = sum(
      (tf.math.abs(reference) * relative_factor
       for reference, relative_factor in zip(references, relative_factors)),
      start=absolute_factor)
  # We are looking for the largest constant 0 <= scale <= 1 such that
  # scale * tf.math.abs(tensor[i]) <= max_delta[i], for all i. Note that both
  # tensor[i] and max_delta[i] may be zeros. If max_delta is zero, then scale
  # must be zero, and if tensor is zero, scale is arbitrary.
  per_element_scale = tf.where(
      tensor == 0., 1., tf.math.divide_no_nan(max_delta, tf.math.abs(tensor)))
  scale = tf.minimum(1., tf.reduce_min(per_element_scale))
  return tensor * scale, scale


@tf.keras.utils.register_keras_serializable()
class ClippyAdagrad(tf.keras.optimizers.Optimizer):
  r"""An Adagrad variant with adaptive clipping.

  The adaptive clipping mechanism multiplies the learning rate for each model
  parameter w by a factor in (0, 1] that ensures that at each iteration w is
  never changed by more than:
    |w| * variable_relative_threshold
        + accumulator_relative_threshold / sqrt(accum) + absolute_threshold,
  where `accum` is the respective Adagrad accumulator.

  Reference: https://arxiv.org/pdf/2302.09178.pdf.

  Attributes:
    iterations: The number of training steps this optimizer has run.
    learning_rate: The learning rate constant or schedule.
    clipping_factors: When the argument `export_clipping_factors` is set to True
      will contain a list of the scaling factors used to clip each variable in
      the model. Otherwise, contains an empty list.
  """

  def __init__(
      self,
      learning_rate: Union[
          float, tf.keras.optimizers.schedules.LearningRateSchedule] = 0.001,
      initial_accumulator_value: float = 0.1,
      variable_relative_threshold: float = 0.1,
      accumulator_relative_threshold: float = 0.0,
      absolute_threshold: float = 1e-7,
      epsilon: float = 1e-7,
      export_clipping_factors: bool = False,
      clip_accumulator_update: bool = False,
      use_standard_accumulator_update: bool = False,
      name: str = "ClippyAdagrad",
      **kwargs,
  ) -> None:
    """Initializes the ClippyAdagrad optimizer.

    Args:
      learning_rate: Initial value for the learning rate: either a floating
        point value, or a `tf.keras.optimizers.schedules.LearningRateSchedule`
        instance. Defaults to 0.001. Note that `Adagrad` tends to benefit from
        higher initial learning rate values compared to other optimizers.
        To match the exact form in the original paper, use 1.0.
      initial_accumulator_value: A non-negative floating point value.
        Starting value for the Adagrad accumulators.
      variable_relative_threshold: A non-negative floating point value. The
        relative threshold factor for the adaptive clipping, relatively to the
        updated variable.
      accumulator_relative_threshold: A non-negative floating point value. The
        clipping threshold factor relatively to the inverse square root of the
        Adagrad accumulators. Default to 0.0 but a non-negative value
        (e.g., 1e-3) allows tightening the clipping threshold in later training.
      absolute_threshold: A non-negative floating point value. The absolute
        clipping threshold constant.
      epsilon: Small floating point value used to maintain numerical stability.
      export_clipping_factors: When set to True, will add an attribute to the
        optimizer, called `clipping_factors`, a list containing the scaling
        factors used to clip each variable in the model. Otherwise,
        the `clipping_factors` attribute is an empty list.
      clip_accumulator_update: When set to True, will also apply clipping on the
        Adagrad accumulator update. This may help improve convergence speed in
        cases where the gradient contains outliers. This cannot be set to True
        when use_standard_accumulator_update is set to True.
      use_standard_accumulator_update: When set to True, will update the
        accumulator before calculating the Adagrad step, as in the classical
        Adagrad method. This cannot be set to True when clip_accumulator_update
        is set to True.
      name: String. The name to use for momentum accumulator weights created by
        the optimizer.
      **kwargs: Other arguments. See `tf.keras.optimizers.Optimizer` docs.

    Raises:
      ValueError: If both `clip_accumulator_update` and
        `use_standard_accumulator_update` are set to True.
    """
    if clip_accumulator_update and use_standard_accumulator_update:
      raise ValueError(
          "clip_accumulator_update and use_standard_accumulator_update cannot "
          "both be set to True.")
    super().__init__(name=name, **kwargs)
    self._learning_rate = self._build_learning_rate(learning_rate)
    self.initial_accumulator_value = initial_accumulator_value
    self.variable_relative_threshold = variable_relative_threshold
    self.accumulator_relative_threshold = accumulator_relative_threshold
    self.absolute_threshold = absolute_threshold
    self.epsilon = epsilon
    self.export_clipping_factors = export_clipping_factors
    self.clip_accumulator_update = clip_accumulator_update
    self.use_standard_accumulator_update = use_standard_accumulator_update
    self.clipping_factors = []

  def build(self, var_list: Sequence[tf.Variable]) -> None:
    super().build(var_list)
    if hasattr(self, "_built") and self._built:  # pylint: disable=access-member-before-definition
      return
    self._built = True
    self._accumulators = []
    self.clipping_factors = []
    initializer = tf.keras.initializers.Constant(self.initial_accumulator_value)
    for var in var_list:
      self._accumulators.append(
          self.add_variable_from_reference(
              var,
              "accumulator",
              initial_value=initializer(shape=var.shape, dtype=var.dtype),
          ))
      if self.export_clipping_factors:
        self.clipping_factors.append(
            self.add_variable_from_reference(
                var,
                "clipping_factor",
                shape=(),
            ))

  def update_step(self, grad: Union[tf.Tensor, tf.IndexedSlices],
                  variable: tf.Variable) -> None:
    """Update step given gradient and the associated model variable."""

    lr = tf.cast(self.learning_rate, variable.dtype)
    epsilon = tf.cast(self.epsilon, variable.dtype)
    var_key = self._var_key(variable)
    accumulator = self._accumulators[self._index_dict[var_key]]

    if self.use_standard_accumulator_update:
      if isinstance(grad, tf.IndexedSlices):
        accumulator.scatter_add(
            tf.IndexedSlices(tf.math.square(grad.values), grad.indices))
      else:
        accumulator.assign_add(tf.math.square(grad))

    if isinstance(grad, tf.IndexedSlices):
      grad_values = grad.values
      accumulator_values = tf.gather(accumulator, indices=grad.indices)
      variable_values = tf.gather(variable, indices=grad.indices)
    else:
      grad_values = grad
      accumulator_values = accumulator
      variable_values = variable

    # Note that unlike the standard Adagrad implementation, ClippyAdagrad
    # supports using accumulator value _before_ adding the current gradient to
    # it (by setting `use_standard_accumulator_update=False`). This allows us to
    # update the accumulator using the clipped gradient value, which is not
    # currently known. Also, mathematically, this makes accumulator independent
    # of the current step, which is ususally considered better practice.
    precondition = tf.math.rsqrt(accumulator_values + epsilon)
    delta = lr * grad_values * precondition
    clipped_delta, clipping_factor = shrink_by_references(
        delta,
        references=[variable_values, precondition],
        relative_factors=[
            self.variable_relative_threshold,
            self.accumulator_relative_threshold
        ],
        absolute_factor=self.absolute_threshold)
    if self.export_clipping_factors:
      self.clipping_factors[self._index_dict[var_key]].assign(clipping_factor)

    if not self.use_standard_accumulator_update:
      # Delayed accumulator update. This allows clipping accumulator update.
      if self.clip_accumulator_update:
        # Clip the accumulator update: this acts like clipping the gradient
        # before sending it to the optimizer. This is a good option when the
        # gradient is an outlier.
        accumulator_update = grad_values * clipping_factor
      else:
        # Does not clip the accumulator update: This is a good option in cases
        # where the gradient increases during training, and allows for quicker
        # adjustment to the increase by the accumulator.
        accumulator_update = grad_values

      if isinstance(grad, tf.IndexedSlices):
        accumulator.scatter_add(
            tf.IndexedSlices(tf.math.square(accumulator_update), grad.indices))
      else:
        accumulator.assign_add(tf.math.square(accumulator_update))

    if isinstance(grad, tf.IndexedSlices):
      variable.scatter_sub(tf.IndexedSlices(clipped_delta, grad.indices))
    else:
      variable.assign_sub(clipped_delta)

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()

    config.update({
        "learning_rate": self._serialize_hyperparameter(self._learning_rate),
        "initial_accumulator_value": self.initial_accumulator_value,
        "variable_relative_threshold": self.variable_relative_threshold,
        "absolute_threshold": self.absolute_threshold,
        "accumulator_relative_threshold": self.accumulator_relative_threshold,
        "epsilon": self.epsilon,
        "export_clipping_factors": self.export_clipping_factors,
        "clip_accumulator_update": self.clip_accumulator_update,
    })
    return config
