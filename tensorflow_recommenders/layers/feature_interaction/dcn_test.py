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

"""Tests for Cross layer."""

import os
import tempfile

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers.feature_interaction.dcn import Cross


class CrossTest(tf.test.TestCase):
  # Do not use layer_test due to multiple inputs.

  def test_full_matrix(self):
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = Cross(projection_dim=None, kernel_initializer="ones")
    output = layer(x0, x)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(np.asarray([[0.55, 0.8, 1.05]]), output)

  def test_low_rank_matrix(self):
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = Cross(projection_dim=1, kernel_initializer="ones")
    output = layer(x0, x)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(np.asarray([[0.55, 0.8, 1.05]]), output)

  def test_one_input(self):
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    layer = Cross(projection_dim=None, kernel_initializer="ones")
    output = layer(x0)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(np.asarray([[0.16, 0.32, 0.48]]), output)

  def test_unsupported_input_dim(self):
    with self.assertRaisesRegexp(ValueError,
                                 r"dimension mismatch"):
      x0 = np.random.random((12, 5))
      x = np.random.random((12, 7))
      layer = Cross()
      layer(x0, x)

  def test_invalid_diag_scale(self):
    with self.assertRaisesRegexp(ValueError,
                                 r"`diag_scale` should be non-negative"):
      x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
      x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
      layer = Cross(diag_scale=-1.)
      layer(x0, x)

  def test_bias(self):
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = Cross(projection_dim=None, kernel_initializer="ones",
                  bias_initializer="ones")
    output = layer(x0, x)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(np.asarray([[0.65, 1., 1.35]]), output)

  def test_serialization(self):
    layer = Cross(projection_dim=None)
    serialized_layer = tf.keras.layers.serialize(layer)
    new_layer = tf.keras.layers.deserialize(serialized_layer)
    self.assertEqual(layer.get_config(), new_layer.get_config())

  def test_diag_scale(self):
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = Cross(
        projection_dim=None, diag_scale=1., kernel_initializer="ones")
    output = layer(x0, x)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(np.asarray([[0.59, 0.9, 1.23]]), output)

  def test_save_model(self):

    def get_model():
      x0 = tf.keras.layers.Input(shape=(13,))
      x1 = Cross(projection_dim=None)(x0, x0)
      x2 = Cross(projection_dim=None)(x0, x1)
      logits = tf.keras.layers.Dense(units=1)(x2)
      model = tf.keras.Model(x0, logits)
      return model

    model = get_model()
    random_input = np.random.uniform(size=(10, 13))
    model_pred = model.predict(random_input)

    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, "dcn_model")
      model.save(
          path,
          options=tf.saved_model.SaveOptions(namespace_whitelist=["Addons"]))
      loaded_model = tf.keras.models.load_model(path)
      loaded_pred = loaded_model.predict(random_input)
    for i in range(3):
      assert model.layers[i].get_config() == loaded_model.layers[i].get_config()
    self.assertAllEqual(model_pred, loaded_pred)


if __name__ == "__main__":
  tf.test.main()
