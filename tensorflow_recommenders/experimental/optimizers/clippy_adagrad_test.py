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

"""Tests for clippy_adagrad."""
import tensorflow as tf

from tensorflow_recommenders.experimental.optimizers import clippy_adagrad


class ClipByReferenceTest(tf.test.TestCase):

  def test_scalar_clip(self):
    self.assertAllCloseAccordingToType(
        (0.42, 0.21),
        clippy_adagrad.shrink_by_references(
            tensor=2.,
            references=[4.],
            relative_factors=[0.1],
            absolute_factor=0.02))
    self.assertAllCloseAccordingToType(
        (0.42, 0.21),
        clippy_adagrad.shrink_by_references(
            tensor=2.,
            references=[-4.],
            relative_factors=[0.1],
            absolute_factor=0.02))
    self.assertAllCloseAccordingToType(
        (-0.42, 0.21),
        clippy_adagrad.shrink_by_references(
            tensor=-2.,
            references=[4.],
            relative_factors=[0.1],
            absolute_factor=0.02))
    self.assertAllCloseAccordingToType(
        (-0.4, 0.2),
        clippy_adagrad.shrink_by_references(
            tensor=-2.,
            references=[4.],
            relative_factors=[0.1],
            absolute_factor=0.))
    self.assertAllCloseAccordingToType(
        (0., 0.),
        clippy_adagrad.shrink_by_references(
            tensor=-2.,
            references=[0.],
            relative_factors=[0.1],
            absolute_factor=0.))
    # No clipping needed.
    self.assertAllCloseAccordingToType(
        (2., 1.),
        clippy_adagrad.shrink_by_references(
            tensor=2.,
            references=[20.],
            relative_factors=[0.1],
            absolute_factor=0.1))
    self.assertAllCloseAccordingToType(
        (-2., 1.),
        clippy_adagrad.shrink_by_references(
            tensor=-2.,
            references=[20.],
            relative_factors=[0.1],
            absolute_factor=0.1))
    self.assertAllCloseAccordingToType(
        (0., 1.),
        clippy_adagrad.shrink_by_references(
            tensor=0.,
            references=[1.],
            relative_factors=[0.1],
            absolute_factor=0.1))
    self.assertAllCloseAccordingToType(
        (0., 1.),
        clippy_adagrad.shrink_by_references(
            tensor=0.,
            references=[1.],
            relative_factors=[0.1],
            absolute_factor=0.))
    self.assertAllCloseAccordingToType(
        (0., 1.),
        clippy_adagrad.shrink_by_references(
            tensor=0.,
            references=[0.],
            relative_factors=[0.],
            absolute_factor=0.))

  def test_scalar_multiple_clip(self):
    self.assertAllCloseAccordingToType(
        (4 * .1 + 5 * .2 + .02, (4 * .1 + 5 * .2 + .02) / 2),
        clippy_adagrad.shrink_by_references(
            tensor=2.,
            references=[4., -5.],
            relative_factors=[0.1, 0.2],
            absolute_factor=0.02))

  def test_scalar_empty_reference(self):
    self.assertAllCloseAccordingToType(
        (.02, .01),
        clippy_adagrad.shrink_by_references(
            tensor=2.,
            references=[],
            relative_factors=[],
            absolute_factor=0.02))
    self.assertAllCloseAccordingToType(
        (0., 1.),
        clippy_adagrad.shrink_by_references(
            tensor=0.,
            references=[],
            relative_factors=[],
            absolute_factor=0.))

  def test_tensor_clip(self):
    clipped, scale = clippy_adagrad.shrink_by_references(
        tensor=tf.convert_to_tensor([1., 1.]),
        references=[tf.convert_to_tensor([1., 0.1])],
        relative_factors=[0.1],
        absolute_factor=0.01
    )
    self.assertAllCloseAccordingToType([0.02, 0.02], clipped)
    self.assertAllCloseAccordingToType(0.02, scale)

  def test_tensor_clip_zero_absolute_factor(self):
    clipped, scale = clippy_adagrad.shrink_by_references(
        tensor=tf.convert_to_tensor([1., 1., 0., 0.]),
        references=[tf.convert_to_tensor([1., 0.1, 1., 0.])],
        relative_factors=[0.1],
        absolute_factor=0.
    )
    self.assertAllCloseAccordingToType([0.01, 0.01, 0., 0.], clipped)
    self.assertAllCloseAccordingToType(0.01, scale)

  def test_tensor_clip_zero_reference(self):
    clipped, scale = clippy_adagrad.shrink_by_references(
        tensor=tf.convert_to_tensor([1., 1., 0., 0.]),
        references=[tf.convert_to_tensor([1., 0., 1., 0.])],
        relative_factors=[0.1],
        absolute_factor=0.
    )
    self.assertAllCloseAccordingToType([0., 0., 0., 0.], clipped)
    self.assertAllCloseAccordingToType(0., scale)

  def test_broadcast(self):
    clipped, scale = clippy_adagrad.shrink_by_references(
        tensor=tf.convert_to_tensor([[1., 2.], [1., 2.]]),
        references=[tf.convert_to_tensor(1.)],
        relative_factors=[0.1],
        absolute_factor=0.1
    )
    self.assertAllCloseAccordingToType([[0.1, 0.2], [0.1, 0.2]], clipped)
    self.assertAllCloseAccordingToType(0.1, scale)


class ClippyAdagradTest(tf.test.TestCase):

  def test_single_step_no_clip(self):
    learning_rate = 0.1
    initial_accumulator_sqrt = 0.1
    optimizer = clippy_adagrad.ClippyAdagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_sqrt**2,
        export_clipping_factors=True)
    x = tf.Variable([1.0, 2.0], dtype=tf.float32)
    g = tf.convert_to_tensor([0.1, 0.15])
    sparse_x = tf.Variable([[3.0, 4.0], [1.0, 2.0]], dtype=tf.float32)
    sparse_g = tf.IndexedSlices(values=[[0.1, 0.15]], indices=[1])
    optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
    self.assertAllCloseAccordingToType(x, [
        1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
        2.0 - learning_rate * 0.15 / initial_accumulator_sqrt
    ])
    self.assertAllCloseAccordingToType(
        sparse_x, [[3.0, 4.0],
                   [
                       1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
                       2.0 - learning_rate * 0.15 / initial_accumulator_sqrt
                   ]])
    self.assertAllCloseAccordingToType(optimizer._accumulators[0], [
        initial_accumulator_sqrt**2 + 0.1**2,
        initial_accumulator_sqrt**2 + 0.15**2
    ])
    self.assertAllCloseAccordingToType(
        optimizer._accumulators[1],
        [[initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
         [
             initial_accumulator_sqrt**2 + 0.1**2,
             initial_accumulator_sqrt**2 + 0.15**2
         ]])
    self.assertAllCloseAccordingToType(optimizer.clipping_factors, [1.0, 1.0])

  def test_single_step_clip(self):
    learning_rate = 0.2
    initial_accumulator_sqrt = 0.1
    optimizer = clippy_adagrad.ClippyAdagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_sqrt**2,
        variable_relative_threshold=0.4,
        accumulator_relative_threshold=0.01,
        absolute_threshold=0.1,
        epsilon=0.0,
        export_clipping_factors=True)
    x = tf.Variable([1.0, 2.0], dtype=tf.float64)
    g = tf.convert_to_tensor([10.0, 10.0], dtype=tf.float64)
    sparse_x = tf.Variable([[3.0, 4.0], [1.0, 2.0]], dtype=tf.float32)
    sparse_g = tf.IndexedSlices(values=[[10.0, 10.0]], indices=[1])
    optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
    # Gradient is clipped so change in x in each coordinate is at
    # most 0.4 x + 0.01 / initial_accumulator_sqrt + 0.1.
    self.assertAllCloseAccordingToType(x, [0.4, 1.4])
    self.assertAllCloseAccordingToType(sparse_x, [[3.0, 4.0], [0.4, 1.4]])
    self.assertAllCloseAccordingToType(optimizer._accumulators[0], [
        initial_accumulator_sqrt**2 + 10.**2,
        initial_accumulator_sqrt**2 + 10.**2
    ])
    self.assertAllCloseAccordingToType(
        optimizer._accumulators[1],
        [[initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
         [
             initial_accumulator_sqrt**2 + 10.**2,
             initial_accumulator_sqrt**2 + 10.**2
         ]])
    # g * clipping_factor * learning_rate / initial_accumulator_sqrt == 0.6
    self.assertAllCloseAccordingToType(optimizer.clipping_factors, [
        0.6 * initial_accumulator_sqrt / (10.0 * learning_rate),
        0.6 * initial_accumulator_sqrt / (10.0 * learning_rate)
    ])

  def test_single_step_clip_with_accumulator(self):
    """Test clip_accumulator_update=True."""
    learning_rate = 0.2
    initial_accumulator_sqrt = 0.1
    optimizer = clippy_adagrad.ClippyAdagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_sqrt**2,
        variable_relative_threshold=0.4,
        accumulator_relative_threshold=0.01,
        absolute_threshold=0.1,
        epsilon=0.0,
        export_clipping_factors=True,
        clip_accumulator_update=True)
    x = tf.Variable([1.0, 2.0], dtype=tf.float64)
    g = tf.convert_to_tensor([10.0, 10.0], dtype=tf.float64)
    sparse_x = tf.Variable([[3.0, 4.0], [1.0, 2.0]], dtype=tf.float32)
    sparse_g = tf.IndexedSlices(values=[[10.0, 10.0]], indices=[1])
    optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
    # Gradient is clipped so change in x in each coordinate is at
    # most 0.4 x + 0.01 / initial_accumulator_sqrt + 0.1.
    self.assertAllCloseAccordingToType(x, [0.4, 1.4])
    self.assertAllCloseAccordingToType(sparse_x, [[3.0, 4.0], [0.4, 1.4]])
    # Make sure the accumulator update takes the clipping factor into account.
    self.assertAllCloseAccordingToType(optimizer._accumulators[0], [
        initial_accumulator_sqrt**2 + (optimizer.clipping_factors[0] * 10)**2,
        initial_accumulator_sqrt**2 + (optimizer.clipping_factors[0] * 10)**2
    ])
    self.assertAllCloseAccordingToType(
        optimizer._accumulators[1],
        [[initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
         [
             initial_accumulator_sqrt**2 +
             (optimizer.clipping_factors[1] * 10)**2,
             initial_accumulator_sqrt**2 +
             (optimizer.clipping_factors[1] * 10)**2
         ]])
    # g * clipping_factor * learning_rate / initial_accumulator_sqrt == 0.6
    self.assertAllCloseAccordingToType(optimizer.clipping_factors, [
        0.6 * initial_accumulator_sqrt / (10.0 * learning_rate),
        0.6 * initial_accumulator_sqrt / (10.0 * learning_rate)
    ])

  def test_single_step_clip_with_standard_update(self):
    """Test use_standard_accumulator_update=True."""
    learning_rate = 0.1
    initial_accumulator_sqrt = 0.
    optimizer = clippy_adagrad.ClippyAdagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_sqrt**2,
        export_clipping_factors=True,
        use_standard_accumulator_update=True)
    x = tf.Variable([1.0, 2.0], dtype=tf.float32)
    g = tf.convert_to_tensor([0.1, 0.15])
    sparse_x = tf.Variable([[3.0, 4.0], [1.0, 2.0]], dtype=tf.float32)
    sparse_g = tf.IndexedSlices(values=[[0.1, 0.15]], indices=[1])
    optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])
    # Since the accumulator was initialized as zero, the Adagrad delta is 1.
    self.assertAllCloseAccordingToType(
        x, [1.0 - learning_rate, 2.0 - learning_rate])
    self.assertAllCloseAccordingToType(
        sparse_x, [[3.0, 4.0], [1.0 - learning_rate, 2.0 - learning_rate]])
    self.assertAllCloseAccordingToType(optimizer._accumulators[0],
                                       tf.math.square(g))
    self.assertAllCloseAccordingToType(
        optimizer._accumulators[1],
        [[initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
         [0.1**2, 0.15**2]])
    self.assertAllCloseAccordingToType(optimizer.clipping_factors, [1.0, 1.0])

  def test_autograph(self):
    learning_rate = 0.1
    initial_accumulator_sqrt = 0.1
    optimizer = clippy_adagrad.ClippyAdagrad(
        learning_rate=learning_rate,
        initial_accumulator_value=initial_accumulator_sqrt**2)

    x = tf.Variable([1.0, 2.0], dtype=tf.float32)
    g = tf.convert_to_tensor([0.1, 0.15])
    sparse_x = tf.Variable([[3.0, 4.0], [1.0, 2.0]], dtype=tf.float32)
    sparse_g = tf.IndexedSlices(values=[[0.1, 0.15]], indices=[1])

    @tf.function()
    def _train_step():
      optimizer.apply_gradients([(g, x), (sparse_g, sparse_x)])

    _train_step()
    self.assertAllCloseAccordingToType(x, [
        1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
        2.0 - learning_rate * 0.15 / initial_accumulator_sqrt
    ])
    self.assertAllCloseAccordingToType(
        sparse_x, [[3.0, 4.0],
                   [
                       1.0 - learning_rate * 0.1 / initial_accumulator_sqrt,
                       2.0 - learning_rate * 0.15 / initial_accumulator_sqrt
                   ]])
    self.assertAllCloseAccordingToType(optimizer._accumulators[0], [
        initial_accumulator_sqrt**2 + 0.1**2,
        initial_accumulator_sqrt**2 + 0.15**2
    ])
    self.assertAllCloseAccordingToType(
        optimizer._accumulators[1],
        [[initial_accumulator_sqrt**2, initial_accumulator_sqrt**2],
         [
             initial_accumulator_sqrt**2 + 0.1**2,
             initial_accumulator_sqrt**2 + 0.15**2
         ]])

  def test_get_config(self):
    optimizer = clippy_adagrad.ClippyAdagrad(
        learning_rate=0.1,
        initial_accumulator_value=0.2,
        variable_relative_threshold=0.3,
        accumulator_relative_threshold=0.6,
        absolute_threshold=0.4,
        epsilon=0.5,
        export_clipping_factors=True)
    config = optimizer.get_config()
    restored_optimizer = clippy_adagrad.ClippyAdagrad.from_config(config)
    self.assertEqual(optimizer.learning_rate, restored_optimizer.learning_rate)
    self.assertEqual(optimizer.initial_accumulator_value,
                     restored_optimizer.initial_accumulator_value)
    self.assertEqual(optimizer.variable_relative_threshold,
                     restored_optimizer.variable_relative_threshold)
    self.assertEqual(optimizer.absolute_threshold,
                     restored_optimizer.absolute_threshold)
    self.assertEqual(optimizer.epsilon, restored_optimizer.epsilon)
    self.assertEqual(optimizer.export_clipping_factors,
                     restored_optimizer.export_clipping_factors)
    self.assertEqual(optimizer.accumulator_relative_threshold,
                     restored_optimizer.accumulator_relative_threshold)


if __name__ == '__main__':
  tf.test.main()
