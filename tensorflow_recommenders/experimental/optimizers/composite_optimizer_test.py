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

"""Tests for CompositeOptimizer."""
import os.path
import tempfile

from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.experimental.optimizers.composite_optimizer import CompositeOptimizer


class CompositeOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ("sgd", "adam"),
      ("rmsprop", "sgd"),
      ("adam", "adagrad"),
      ("adagrad", "rmsprop"))
  def test_composite_optimizer(self, optimizer1_type, optimizer2_type):
    values1 = [1.0, 2.0, 3.0]
    values2 = [0.5, 0.0, -2.0]
    values3 = [0.1, 0.0, -1.0]

    grad1_values = [0.1, 0.2, 1.0]
    grad2_values = [-0.1, 0.05, 2.0]
    grad3_values = [2.1, 0.0, 0.3]

    var1 = tf.Variable(values1)
    var2 = tf.Variable(values2)
    var3 = tf.Variable(values3)

    grads1 = tf.constant(grad1_values)
    grads2 = tf.constant(grad2_values)
    grads3 = tf.constant(grad3_values)

    comp_optimizer1 = tf.keras.optimizers.get(optimizer1_type)
    comp_optimizer2 = tf.keras.optimizers.get(optimizer2_type)

    composite_optimizer = CompositeOptimizer([
        (comp_optimizer1, lambda: [var1]),
        (comp_optimizer2, lambda: [var2, var3]),
    ])

    self.assertSequenceEqual(
        composite_optimizer.optimizers, [comp_optimizer1, comp_optimizer2])

    optimizer1 = tf.keras.optimizers.get(optimizer1_type)
    optimizer2 = tf.keras.optimizers.get(optimizer2_type)

    grads_and_vars_1 = [(tf.constant(grad1_values), tf.Variable(values1))]
    grads_and_vars_2 = [(tf.constant(grad2_values), tf.Variable(values2)),
                        (tf.constant(grad3_values), tf.Variable(values3))]
    grads_and_vars = list(zip([grads1, grads2, grads3], [var1, var2, var3]))

    for _ in range(10):
      # Test that applying a composite optimizer has the same effect as
      # applying optimizer1 and optimizer2 separately on subset of gradients/
      # variables.
      composite_optimizer.apply_gradients(grads_and_vars)
      optimizer1.apply_gradients(grads_and_vars_1)
      optimizer2.apply_gradients(grads_and_vars_2)

      self.assertAllClose(grads_and_vars[:1], grads_and_vars_1)
      self.assertAllClose(grads_and_vars[1:], grads_and_vars_2)

  def test_incorrect_inputs(self):
    var1 = tf.Variable([0.1, 0.2, 1.0])
    var2 = tf.Variable([-5.1, 0.1, 0])
    var3 = tf.Variable([-2.1, 1.3, 0/3])

    grads1 = tf.constant([0.1, 0.2, 1.0])
    grads2 = tf.constant([0.5, 0.0, -2.0])
    grads3 = tf.constant([-0.2, 0.0, -1.0])

    # Test same variable in two optimizers.
    composite_optimizer = CompositeOptimizer([
        (tf.keras.optimizers.Adam(), lambda: [var1]),
        (tf.keras.optimizers.Adagrad(), lambda: [var1, var2]),
    ])

    grads_and_vars = list(zip([grads1, grads2], [var1, var2]))

    with self.assertRaises(ValueError):
      composite_optimizer.apply_gradients(grads_and_vars)

    # Test missing variable (var3) in optimizers.
    composite_optimizer = CompositeOptimizer([
        (tf.keras.optimizers.Adam(), lambda: [var1]),
        (tf.keras.optimizers.Adagrad(), lambda: [var2]),
    ])

    grads_and_vars = list(zip([grads1, grads2, grads3], [var1, var2, var3]))

    with self.assertRaises(ValueError):
      composite_optimizer.apply_gradients(grads_and_vars)

  def test_checkpoint_save_restore_export(self):
    # Use a simple Linear model to test checkpoint save/restore/export.
    def get_model() -> tf.keras.Model:
      model = tf.keras.experimental.LinearModel(units=10)

      composite_optimizer = CompositeOptimizer([
          (tf.keras.optimizers.Adam(),
           lambda: model.trainable_variables[:1]),
          (tf.keras.optimizers.Adagrad(),
           lambda: model.trainable_variables[1:]),
      ])
      model.compile(optimizer=composite_optimizer,
                    loss=tf.keras.losses.MSE)
      return model

    batch_size = 16
    num_of_batches = 8
    rng = np.random.RandomState(42)

    x = rng.normal(size=(num_of_batches * batch_size, 5))
    y = rng.normal(size=(num_of_batches * batch_size, 1))
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_dataset = training_dataset.batch(batch_size)

    model = get_model()
    model.fit(training_dataset, epochs=1)

    # Check that optimizer iterations match dataset size.
    self.assertEqual(model.optimizer.iterations.numpy(), num_of_batches)
    # Check that it has state for all the model's variables
    self.assertLen(model.optimizer.variables(), 5)

    # Save checkpoint.
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_path = self.get_temp_dir()
    checkpoint.write(checkpoint_path)

    # Restore to a fresh instance and check.
    new_model = get_model()
    # Run only one epoch: if the restore fails, we can tell
    # by the number of iterations being 1 rather than `num_batches`.
    new_model.fit(training_dataset.take(1))

    checkpoint = tf.train.Checkpoint(model=new_model)
    checkpoint.read(checkpoint_path).assert_consumed()

    # After restoring the checkpoint, optimizer iterations should also be
    # restored to its original value.
    self.assertEqual(new_model.optimizer.iterations.numpy(), num_of_batches)
    # Same for the rest of its variables.
    self.assertAllClose(
        new_model.optimizer.variables(),
        model.optimizer.variables()
    )

    model_pred = new_model.predict(training_dataset)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "model_with_composite_optimizer")
      new_model.save(
          path,
          include_optimizer=False,
          options=tf.saved_model.SaveOptions(namespace_whitelist=["Addons"]))
      loaded_model = tf.keras.models.load_model(path)
      loaded_pred = loaded_model.predict(training_dataset)

    self.assertEqual(
        model.layers[0].get_config(), loaded_model.layers[0].get_config())
    self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
  tf.test.main()
