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

"""Composite Optimizer."""

import collections
from typing import Callable, List, Optional, Sequence, Tuple, Union

import tensorflow as tf

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]


class CompositeOptimizer(tf.keras.optimizers.Optimizer):
  """An optimizer that composes multiple individual optimizers.

  It allows different optimizers to be applied to different subsets of the
  model's variables. For example, it makes it possible to apply one
  optimizer to the model's embeddings (sparse variables) and another
  optimizer to the rest of its variables.

  To specify which optimizer should apply to each variable, pass a list of
  pairs of (optimizer instance, function returning a list of variables the
  optimizer should apply to).

  For example:
  ```python
    optimizer = CompositeOptimizer([
        (tf.keras.optimizers.SGD(), lambda: model.sparse_trainable_variables),
        (tf.keras.optimizers.Adam(), lambda: model.dense_trainable_variables),
    ])
  ```
  """

  def __init__(self,
               optimizers_and_vars: Sequence[
                   Tuple[tf.keras.optimizers.Optimizer,
                         Callable[[], Sequence[tf.Variable]]]],
               name: str = "CompositeOptimizer") -> None:
    """Initializes an CompositeOptimizer instance.

    Args:
      optimizers_and_vars:  List of tuples of (optimizer instance, function
        returning variables that the optimizer should apply to).
      name: The optimizer name.
    """
    super().__init__(name=name)
    if not optimizers_and_vars:
      raise ValueError("`optimizers_and_vars` can't be empty")
    self._optimizers_and_vars = optimizers_and_vars
    for i, optimizer_and_var in enumerate(optimizers_and_vars):
      optimizer = optimizer_and_var[0]
      self._track_trackable(optimizer, name=f"Optimizer{i}")

  def apply_gradients(self, grads_and_vars: Sequence[Tuple[Tensor, Tensor]],
                      name: Optional[str] = None) -> None:
    """See base class."""
    var_optimizer_dict = {}

    for optimizer, var_callable in self._optimizers_and_vars:
      for v in var_callable():
        if v.ref() in var_optimizer_dict:
          raise ValueError(
              f"The set of variables handled by each optimizer should be "
              f"disjoint, but variable {v} is handled both "
              f"by {var_optimizer_dict[v.ref()]} and {optimizer}.")
        var_optimizer_dict[v.ref()] = optimizer

    optimizer_grads_and_vars = collections.defaultdict(list)
    for g, v in grads_and_vars:
      if v.ref() in var_optimizer_dict:
        optimizer = var_optimizer_dict[v.ref()]
        optimizer_grads_and_vars[optimizer].append((g, v))
      else:
        raise ValueError(f"Variable {v} is not handled by any optimizer. "
                         f"This would cause it to be not trained.")

    for optimizer, opt_grads_and_vars in optimizer_grads_and_vars.items():
      optimizer.apply_gradients(opt_grads_and_vars, name=name)

  def get_config(self):
    raise NotImplementedError("CompositeOptimizer cannot be serialized because"
                              " it uses callable to get variables.")

  @property
  def iterations(self):
    """See base class."""
    # Returning iterations from the first optimizer.
    return self._optimizers_and_vars[0][0].iterations

  @iterations.setter
  def iterations(self, variable):
    """See base class."""
    for optimizer, _ in self._optimizers_and_vars:
      optimizer.iterations = variable

  def variables(self):
    """Returns the optimizer's variables."""
    # OptimizerV2.variables() returns self._weights, so override that method.
    return self.weights

  @property
  def weights(self) -> List[tf.Variable]:
    """Returns the optimizer's variables."""
    weights = []
    for optimizer, _ in self._optimizers_and_vars:
      weights += optimizer.weights
    return weights

  @property
  def optimizers(self) -> List[tf.keras.optimizers.Optimizer]:
    """Returns the optimizers in composite optimizer (in the original order)."""
    return [optimizer for optimizer, _ in self._optimizers_and_vars]

