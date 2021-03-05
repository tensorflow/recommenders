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

"""Tests for Keras TPUEmbeddingLayer with custom training loop."""

import functools
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers.embedding import tpu_embedding_layer

_MASTER = ''


def variable_creation_fn(name, shape, initializer, trainable, dtype):
  return tf.Variable(name=name,
                     initial_value=functools.partial(initializer,
                                                     shape=shape,
                                                     dtype=dtype),
                     trainable=trainable)


def create_distribute_input_option():
  # Add a try...except block as OSS tensorflow_recommenders is depending on
  # stable TF version, i.e. TF2.4.
  try:
    return tf.distribute.InputOptions(experimental_fetch_to_device=False)
  except TypeError:
    return tf.distribute.InputOptions(experimental_prefetch_to_device=False)


class TPUEmbeddingLayerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.embedding_values = np.array(list(range(8)), dtype=np.float64)
    self.initializer = tf.constant_initializer(self.embedding_values)
    # Embedding for video initialized to
    # 0 1 2 3
    # 4 5 6 7
    self.table_config_video = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=2,
        dim=4,
        initializer=self.initializer,
        combiner='sum',
        name='video_table')
    # Embedding for user initialized to
    # 0 1
    # 2 3
    # 4 5
    # 6 7
    self.table_config_user = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=4,
        dim=2,
        initializer=self.initializer,
        combiner='mean',
        name='user_table')
    self.feature_config = {
        'watched': tf.tpu.experimental.embedding.FeatureConfig(
            table=self.table_config_video, name='watched'),
        'favorited': tf.tpu.experimental.embedding.FeatureConfig(
            table=self.table_config_video, name='favorited'),
        'friends': tf.tpu.experimental.embedding.FeatureConfig(
            table=self.table_config_user, name='friends')
    }

    self.batch_size = 4

    # One (global) batch of inputs
    # sparse tensor for watched:
    # row 0: 0
    # row 1: 0, 1
    # row 2: 0, 1
    # row 3: 1
    self.feature_watched_indices = [[0, 0], [1, 0], [1, 1],
                                    [2, 0], [2, 1], [3, 0]]
    self.feature_watched_values = [0, 0, 1, 0, 1, 1]
    self.feature_watched_row_lengths = [1, 2, 2, 1]
    # sparse tensor for favorited:
    # row 0: 0, 1
    # row 1: 1
    # row 2: 0
    # row 3: 0, 1
    self.feature_favorited_indices = [[0, 0], [0, 1], [1, 0],
                                      [2, 0], [3, 0], [3, 1]]
    self.feature_favorited_values = [0, 1, 1, 0, 0, 1]
    self.feature_favorited_row_lengths = [2, 1, 1, 2]
    # sparse tensor for friends:
    # row 0: 3
    # row 1: 0, 1, 2
    # row 2: 3
    # row 3: 0, 1, 2
    self.feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2],
                                    [2, 0], [3, 0], [3, 1], [3, 2]]
    self.feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]
    self.feature_friends_row_lengths = [1, 3, 1, 3]
    self.labels = [2, 5]
    self.dense_feature_values = [10.7, 11.4, 10.7, 11.4]

  def _create_strategy_and_model(self,
                                 optimizer_name,
                                 has_labels=False,
                                 sparse=True):

    # Use the default strategy.
    strategy = tf.distribute.get_strategy()

    def model_fn(embedding_layer):
      input_args = {'batch_size': self.batch_size,
                    'shape': (),
                    'sparse' if sparse else 'ragged': True,
                    'dtype': tf.int32}
      embedding_inputs = {
          'watched': tf.keras.Input(**input_args, name='watched'),
          'favorited': tf.keras.Input(**input_args, name='favorited'),
          'friends': tf.keras.Input(**input_args, name='friends')}
      activations = embedding_layer(embedding_inputs)
      dense_input = tf.keras.Input(shape=())
      inputs = {'dense': dense_input}
      inputs.update(embedding_inputs)
      if has_labels:
        label_inputs = {'label': tf.keras.Input(shape=())}
        return tf.keras.Model(
            inputs=(inputs, label_inputs),
            outputs=(activations, dense_input, label_inputs))
      else:
        return tf.keras.Model(
            inputs=inputs,
            outputs=(activations, dense_input))

    # Create model with Keras.
    with strategy.scope():
      # The actual optimizer class here will not be used for TPU training as in
      # those cases there are no variables for this model. When not using
      # TPU strategy this optimizer will be used.
      optimizer = _create_optimizer(
          optimizer_name=optimizer_name)
      embedding_layer = tpu_embedding_layer.TPUEmbedding(
          feature_config=self.feature_config, optimizer=optimizer)
      model = model_fn(embedding_layer)

    return model, strategy, optimizer

  @parameterized.parameters(
      *itertools.product(
          ['sgd', 'adagrad', 'adam'],
          [True, False],
          [True, False]))
  def test_embedding_layer(self, optimizer_name, training, sparse):
    model, strategy, optimizer = (
        self._create_strategy_and_model(optimizer_name, sparse=sparse))
    if sparse:
      dataset = self._create_sparse_dataset(strategy)
    else:
      dataset = self._create_ragged_dataset(strategy)
    dist = strategy.experimental_distribute_dataset(
        dataset, options=create_distribute_input_option())
    dist_iter = iter(dist)

    @tf.function
    def test_fn():

      def step(features):
        """Create and run computation that returns the embedding activations."""
        if not training:
          activations, _ = model(features)
          total_loss = _get_total_loss_tensor(activations)
          activation_list_keys = ['watched', 'favorited', 'friends']
          activation_list = [activations[key] for key in activation_list_keys]
          ret_val = [total_loss] + activation_list
          return ret_val
        else:
          with tf.GradientTape() as tape:
            activations, _ = model(features)
            total_loss = _get_total_loss_tensor(activations)
            loss_per_replica = total_loss / strategy.num_replicas_in_sync
          gradients = tape.gradient(loss_per_replica, model.trainable_variables)
          optimizer.apply_gradients(list(zip(gradients,
                                             model.trainable_variables)))
          activation_list_keys = ['watched', 'favorited', 'friends']
          activation_list = [activations[key] for key in activation_list_keys]
          ret_val = [total_loss] + activation_list
          return ret_val

      result = strategy.run(step, args=(next(dist_iter),))
      return result

    # Run model.
    out_val = test_fn()

    self.assertIsNotNone(out_val)
    # Length should be 4: 1 for loss and 3 for activations.
    self.assertLen(out_val, 4)

  def _create_sparse_dataset(self, strategy):
    # Create dataset for enqueue operation
    sparse_features = {}
    sparse_features['watched'] = tf.SparseTensor(
        indices=self.feature_watched_indices,
        values=tf.convert_to_tensor(self.feature_watched_values,
                                    dtype=tf.int32),
        dense_shape=[self.batch_size, 2])
    sparse_features['favorited'] = tf.SparseTensor(
        indices=self.feature_favorited_indices,
        values=tf.convert_to_tensor(self.feature_favorited_values,
                                    dtype=tf.int32),
        dense_shape=[self.batch_size, 2])
    sparse_features['friends'] = tf.SparseTensor(
        indices=self.feature_friends_indices,
        values=tf.convert_to_tensor(self.feature_friends_values,
                                    dtype=tf.int32),
        dense_shape=[self.batch_size, 3])
    sparse_features['dense'] = tf.constant(
        self.dense_feature_values,
        dtype=tf.float32)

    return tf.data.Dataset.from_tensors(sparse_features)

  def _create_ragged_dataset(self, strategy, include_weights=False, weight=0.5):
    # Create dataset for enqueue operation
    ragged_features = {}
    ragged_features['watched'] = tf.RaggedTensor.from_row_lengths(
        row_lengths=self.feature_watched_row_lengths,
        values=tf.convert_to_tensor(self.feature_watched_values,
                                    dtype=tf.int32))
    ragged_features['favorited'] = tf.RaggedTensor.from_row_lengths(
        row_lengths=self.feature_favorited_row_lengths,
        values=tf.convert_to_tensor(self.feature_favorited_values,
                                    dtype=tf.int32))
    ragged_features['friends'] = tf.RaggedTensor.from_row_lengths(
        row_lengths=self.feature_friends_row_lengths,
        values=tf.convert_to_tensor(self.feature_friends_values,
                                    dtype=tf.int32))
    ragged_features['dense'] = tf.constant(
        self.dense_feature_values,
        dtype=tf.float32)
    return tf.data.Dataset.from_tensors(ragged_features)

  def _create_dense_input_fn(self, strategy, provide_labels=False):
    # Create dataset for enqueue operation
    features_list = []
    for i in range(strategy.num_replicas_in_sync):
      features = {}
      features['watched'] = tf.constant(self.feature_watched_values[-2:],
                                        dtype=tf.int32)
      features['favorited'] = tf.constant(self.feature_favorited_values[-2:],
                                          dtype=tf.int32)
      features['friends'] = tf.constant(self.feature_friends_values[-2:],
                                        dtype=tf.int32)
      features['dense'] = tf.constant(self.dense_feature_values,
                                      dtype=tf.float32)
      if i % 2 == 1:
        features['favorited'], features['watched'] = (
            features['watched'], features['favorited'])
      if provide_labels:
        labels = {'labels': tf.constant(self.labels, dtype=tf.int32)}
        features_list.append((features, labels))
      else:
        features_list.append(features)

    def input_fn(ctx):
      del ctx
      return tf.data.Dataset.from_tensors(features_list[0]).concatenate(
          tf.data.Dataset.from_tensors(features_list[1]))

    return input_fn


def _get_total_loss_tensor(activations):
  losses = []
  for key in activations:
    losses.append(
        tf.reduce_mean(
            tf.reduce_sum(
                tf.math.squared_difference(activations[key], 0), 1)))
  total_loss = tf.expand_dims(sum(losses), 0)
  return total_loss


def _create_optimizer(optimizer_name='adagrad'):
  if optimizer_name == 'sgd':
    return tf.keras.optimizers.SGD(learning_rate=0.1)
  elif optimizer_name == 'adagrad':
    return tf.keras.optimizers.Adagrad(
        learning_rate=0.1,
        initial_accumulator_value=10)
  elif optimizer_name == 'adam':
    return tf.keras.optimizers.Adam(
        learning_rate=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)
  raise ValueError('Unknown optimizer specified')


if __name__ == '__main__':
  tf.test.main()
