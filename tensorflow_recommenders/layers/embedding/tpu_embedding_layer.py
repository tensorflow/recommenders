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

"""Keras interface for TPU Embeddings in TF2."""

import tensorflow as tf


_SLOT_NAME_MAPPING = {
    # Slot names in Keras optimizer v2 are different compared to the slot names
    # in our API.
    tf.keras.optimizers.Adagrad: {"accumulators": "accumulator"},
    tf.keras.optimizers.Adam: {"momenta": "m", "velocities": "v"}
}
_OPTIMIZER_PARAMETERS = {
    # A tuple: first element is the embedding optimizer class. Second is the
    # list of supported hyper parameters and the second list is the unsupported
    # hyperparameters.
    tf.keras.optimizers.Adam: (tf.tpu.experimental.embedding.Adam,
                               ["learning_rate", "beta_1", "beta_2", "epsilon"],
                               ["decay", "amsgrad"]),
    tf.keras.optimizers.Adagrad: (
        tf.tpu.experimental.embedding.Adagrad,
        ["learning_rate", "initial_accumulator_value"],
        ["epsilon"]),
    tf.keras.optimizers.SGD: (tf.tpu.experimental.embedding.SGD,
                              ["learning_rate"],
                              ["decay", "momentum", "nesterov"])
}
_DUMMY_NAME = "tpu_embedding_helper_dummy"


def _get_batch_size_from_input_shapes(input_shape):
  """From a list of input shapes, gets the global batch size.

  We want to extract the first dimension for each TensorShape in input_shape and
  ensure that:
  1. They are all the same or None.
  2. They are not all None.

  If SparseTensors are fed directly into call (which in turn calls build),
  during tracing, the shape of the SparseTensor will itself be a tensor which
  results in unknown dimensions for the SparseTensor.

  Args:
    input_shape: A nested structure of `TensorShape`s.

  Returns:
    The global batch size.
  """
  flattened_input_shape = tf.nest.flatten(input_shape)
  if not flattened_input_shape:
    raise ValueError("No input passed to TPUEmbedding layer.")

  per_core_batch_size = None

  for tensor_shape in flattened_input_shape:
    if tensor_shape.rank < 1:
      raise ValueError(
          "Received input tensor of shape {}. Rank must be > 0.".format(
              tensor_shape))
    shape = tensor_shape.as_list()
    if shape[0] is not None:
      if per_core_batch_size is None:
        per_core_batch_size = shape[0]
      elif per_core_batch_size != shape[0]:
        raise ValueError("Found multiple batch sizes {} and {} in input. All "
                         "features must have the same batch size.".format(
                             per_core_batch_size, shape[0]))

  if per_core_batch_size is None:
    raise ValueError("Unable to determine batch dimension of any features. "
                     "This may happen if all inputs are SparseTensors. If you "
                     "are using tf.keras.Input, you must specify a batch size."
                     "If you are using the layer in a subclass model and "
                     "calling the layer directly/using build(), you must "
                     "specify a batch size when constructing the layer.")

  return per_core_batch_size


def _normalize_and_prepare_optimizer(optimizer):
  """Normalizes an optimizer into a mid level API optimizer class.

  In the case of a mid level API optimizer, this just passes it through.
  Passing optimizer names, "sgd", "adam", "adagrad" are supported and
  instantiate the mid level API object with default parameters. If a Keras
  optimizer is passed it will be converted to a mid level optimizer.

  Args:
    optimizer: A keras optimizer, string optimizer name or subclass of
  _OptimizationParameters.

  Returns:
    A subclass of tpu_embedding_v2._Optimizer or None.
  """

  if optimizer is None:
    return None
  elif isinstance(optimizer, (tf.tpu.experimental.embedding.SGD,
                              tf.tpu.experimental.embedding.Adagrad,
                              tf.tpu.experimental.embedding.Adam)):
    return optimizer
  elif isinstance(optimizer, str):
    if str(optimizer) == "sgd":
      return tf.tpu.experimental.embedding.SGD()
    elif str(optimizer) == "adagrad":
      return tf.tpu.experimental.embedding.Adagrad()
    elif str(optimizer) == "adam":
      return tf.tpu.experimental.embedding.Adam()
    else:
      raise ValueError("Unknown optimizer name '{}'. Please use one of 'sgd',"
                       "'adagrad' or 'adam'".format(optimizer))
  elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
    return translate_keras_optimizer(optimizer)
  else:
    raise ValueError(
        "Unknown optimizer type {}. Please pass a string optimizer name, a "
        "subclass of keras optimizer or an instance of one of the optimizer "
        "parameter classes under tf.tpu.experimental.embedding.".format(
            type(optimizer)))


def _clone_and_prepare_features(feature_config):
  """Prepares a nested structure of FeatureConfig objects for mid level api.

  Clones the feature_config structure and its contained
  `tf.tpu.experimental.embedding.TableConfig` objects. This is done so that
  TPUEmbedding layer doesn't touch the user's original configuration.

  Args:
    feature_config: A nested structure of
      `tf.tpu.experimental.embedding.FeatureConfig` objects.

  Returns:
    A nested structure of
    `tf.tpu.experimental.embedding.FeatureConfig` objects and list of tuples
    mapping user `tf.tpu.experimental.embedding.TableConfig` objects to the
    internal ones.
  """
  output_objects = []

  table_configs = {}

  for config in tf.nest.flatten(feature_config):
    # There should be a one-to-one mapping between new TableConfig objects and
    # old ones (as each TableConfig can be thought of as a table).
    table_configs[config.table] = table_configs.get(
        config.table,
        tf.tpu.experimental.embedding.TableConfig(
            vocabulary_size=config.table.vocabulary_size,
            dim=config.table.dim,
            initializer=config.table.initializer,
            optimizer=config.table.optimizer,
            combiner=config.table.combiner,
            name=config.table.name))

    output_objects.append(tf.tpu.experimental.embedding.FeatureConfig(
        table=table_configs[config.table],
        max_sequence_length=config.max_sequence_length,
        name=config.name))

  # Fix up the optimizers.
  for _, new_table in table_configs.items():
    if new_table.optimizer is not None:
      new_table.optimizer = _normalize_and_prepare_optimizer(
          new_table.optimizer)

  return (tf.nest.pack_sequence_as(feature_config, output_objects),
          list(table_configs.items()))


def _update_table_configs(feature_config, table_config_map):
  """Updates TableConfigs in a nested structure of FeatureConfigs.

  _clone_and_prepare_features clones a structure FeatureConfigs and returns a
  mapping of user TableConfig objects to internal TableConfig objects. This
  function will clone a nested structure of FeatureConfigs and apply the
  transformation of TableConfigs.

  Args:
    feature_config: A nested structure of
      `tf.tpu.experimental.embedding.FeatureConfig` objects.
    table_config_map: A list of tuples of
      `tf.tpu.experimental.embedding.TableConfig`, mapping user TableConfigs to
      internal TableConfigs.

  Returns:
    A clone of the feature_config with the table arguments updates via the
    mapping passed in by table_config_map.

  Raises:
    ValueError: if there is a TableConfig object that was not passed in on layer
      initialization.
  """
  output_objects = []
  table_config_dict = dict(table_config_map)
  for config in tf.nest.flatten(feature_config):
    if config.table not in table_config_dict:
      raise ValueError("TableConfig %s does not match any of the TableConfigs "
                       "used to configure this layer." % config.table)
    output_objects.append(tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config_dict[config.table],
        max_sequence_length=config.max_sequence_length,
        name=config.name))

  return tf.nest.pack_sequence_as(feature_config, output_objects)


def _is_tpu_strategy(strategy):
  return isinstance(strategy, (
      tf.distribute.experimental.TPUStrategy,
      tf.distribute.TPUStrategy))


class TPUEmbedding(tf.keras.layers.Layer):
  """A Keras layer for accelerating embedding lookups for large tables with TPU.

  ## Feature and table configuration

  When creating an instance of this layer, you must specify:

  1.  The complete set of embedding tables,
  2.  The features you expect to lookup in those tables and
  3.  The optimizer(s) you wish to use on the tables.

  See the documentation of `tf.tpu.experimental.embedding.TableConfig` and
  `tf.tpu.experimental.embedding.FeatureConfig` for more details on the complete
  set of options. We will cover the basic usage here.

  NOTE: multiple `FeatureConfig` objects can use the same `TableConfig` object,
  allowing different features to share the same table:

  ```python
  table_config_one = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  table_config_two = tf.tpu.experimental.embedding.TableConfig(
      vocabulary_size=...,
      dim=...)
  feature_config = {
      'feature_one': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_two': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_one),
      'feature_three': tf.tpu.experimental.embedding.FeatureConfig(
          table=table_config_two)}
  ```

  ## Optimizers

  An optimizer can be globally specified by passing one of the following types
  of input to the optimizer argument:

  1.  A string, one of 'sgd', 'adagrad' or 'adam', which uses the given
      optimizer with the default parameters.
  2.  An instance of a Keras optimizer.
  3.  An instance of an optimizer class from the `tf.tpu.experimental.embedding`
      module.

  You may also specify an optimizer as the table level via the optimizer
  argument of `tf.tpu.experimental.embedding.TableConfig`. This will completely
  override the global optimizer for this table. For performance reasons it is
  recommended that you minimize the total number of distinct optimizers.

  NOTE: If you wish to use Keras optimizer and access the embedding specific
  optimizer parameters, apply the `translate_keras_optimizer` function from this
  module to your Keras optmizer in order to convert it into a
  `tf.tpu.experimental.embedding` optimizer. In this case, the Keras optimizer
  class instance you specify will be used to manage the slot variables. This
  allows you to instantiate the model under a non-TPU strategy and still by able
  to train it. See below for a code example. Thus it is important to include the
  Keras optimizer instance in your checkpoint. If you use case 1 or create your
  own instance of an optimizer class from `tf.tpu.experimental.embedding`, the
  slot variables will be directly managed by the layer.

  ### Dynamic Learning Rate

  Using a dynamic learning rate is supported for all optimizers, all other
  hyper parameters are static. There are two ways of specifying a dynamic
  learning rate in your optimizer:

  1.  One of the objects in the `tf.keras.optimizers.schedules` name space.
  2.  A python callable takeing no parameters which returns a scalar tensor of
      type `tf.float32`.

  #### tf.keras.optimizers.schedules

  This method of specifying a learning schedule is only possible when using a
  Keras optimizer. In this case, set the learning rate of the optimizer to your
  desired `tf.keras.optimizers.schedules` object.

  NOTE: In this case, you *must* call `optimizer.apply_gradients` during your
  training loop so that the optimizer's iterations variable is incremented once
  per step. If you are using a separate optimizer for you embedding layers, see
  the 'Using this layer on CPU' section below for an example of how to do this
  correctly.

  #### Callable

  This method can be used if you use a Keras optimizer or one of the optimizer
  classes in the `tf.tpu.experimental.embedding` namespace.

  In either case you should create a callable function that returns a tensor.
  This function will be called once, but the ops it generates will be
  reevaluated each step. Thus it is recommended that you either create a
  `tf.Variable` representing your current step counter or use the `iterations`
  property of an optimizer you call `apply_gradients` on each trianing step.

  NOTE: If you create your own variable you must create the variable under the
  scope of a TPUStrategy if you are using the layer on the TPU. E.g.

  ```python
  with strategy.scope():
    step = tf.Variable(
        initial_value=0, trainable=False, dtype=tf.int64,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
  ```

  ## Model creation

  For a functional style Keras model:

  ```python
  strategy = tf.distribute.TPUStrategy(...)
  with strategy.scope():
    embedding_inputs = {
        'feature_one': tf.keras.Input(batch_size=1024, shape=(1,),
                                      dtype=tf.int32),
        'feature_two': tf.keras.Input(batch_size=1024, shape=(1,),
                                      dtype=tf.int32, ragged=True),
        'feature_three': tf.keras.Input(batch_size=1024, shape=(1,),
                                        dtype=tf.int32)}
    # embedding, feature_config and embedding_inputs all have the same nested
    # structure.
    embedding = tpu_embedding_layer.TPUEmbedding(
        feature_config=feature_config,
        optimizer=tf.tpu.experimental.embedding.SGD(0.1))(embedding_inputs)
    logits = tf.keras.layers.Dense(1)(
        tf.concat(tf.nest.flatten(embedding), axis=1)
    )
    model = tf.keras.Model(embedding_inputs, logits)
  ```

  For a subclass style model:

  ```python
  class ModelWithEmbeddings(tf.keras.Model):
    def __init__(self):
      super(ModelWithEmbeddings, self).__init__()
      self.embedding_layer = tpu_embedding_layer.TPUEmbedding(
          feature_config=feature_config,
          optimizer=tf.tpu.experimental.embedding.SGD(0.1))
      self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
      embedding = self.embedding_layer(inputs)
      logits = self.dense(tf.concat(tf.nest.flatten(embedding), axis=1))
      return logits

  with strategy.scope():
    model = ModelWithEmbeddings()
  ```

  NOTE: It is important that the `TPUEmbedding` layer is created under a
  `TPUStrategy` if you intend to use it under a `TPUStrategy`

  ## Input data

  When creating a distributed dataset that is to be passed to be used with a
  model that contains a TPUEmbedding layer, a special option must be specified
  when calling any of the dataset distribution methods of `TPUStrategy`:

  ```python
  distributed_dataset = (
      strategy.distribute_datasets_from_function(
          dataset_fn=...,
          options=tf.distribute.InputOptions(
              experimental_prefetch_to_device=False))
  dataset_iterator = iter(distributed_dataset)
  ```

  NOTE: All batches passed to the layer must have the same batch size for each
  input, moreover once you have called the layer with one batch size all
  subsequent calls must use the same batch_size. In the event that the batch
  size cannot be automatically determined by the enqueue method, you must
  specify the batch_size argument when instantiating the layer.

  ## Training and evaluation

  To use this API on TPU you should use a custom training loop. Below is an
  example of a training and evaluation step:

  ```python
  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(inputs):
      labels, features = inputs
      with tf.GradientTape() as tape:
        model_output = model(features)
        loss = ...  # some function of labels and model_output

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))

  @tf.function
  def evaluation_step(dataset_iterator, num_steps):
    def tpu_step(inputs):
      labels, features = inputs
      model_output = model(features)
      # Insert your evaluation code here.

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))
  ```

  In the above examples, we assume that the user has a dataset which returns
  a tuple where the second element of the tuple matches the structure of what
  was passed as the `feature_config` argument to the object initializer. Also we
  utilize `tf.range` to get a `tf.while_loop` in order to increase performance.

  The embedding layer does not affect checkpointing; simply checkpoint your
  model as normal, remembering that if you passed either a Keras optimizer or an
  optimizer converted from a Keras optimizer via `translate_keras_optimizer` you
  must checkpoint the optimizer to ensure that your slot variables are saved.

  ```python
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.save(...)
  ```

  NOTE: Do not use the `tf.saved_model` API on a model with TPUEmbedding layer
  if you want to load the `tf.saved_model` on TPU and continue training. At the
  current time, the `tf.saved_model` API should only be used for exporting a
  model for serving.

  ## Serving

  Serving is accomplished through the `tf.saved_model` API. The model may be
  exported directly from training.

  First we write a `tf.function` that represents the serving graph. Typically
  this may take as input a string tensor containing protos that are parsed into
  tensors and then passed to the model. I.e.

  ```python
  @tf.function(input_signature=[{
      'examples':
          tf.TensorSpec(
              shape=[None], dtype=tf.string, name='examples')}])
  def serve_examples(examples):
    input_data = ...  # parse the examples tensor to produce input tensors.
    return model(input_data)
  ```

  NOTE: It is important that the input_signature is specified here so that the
  exported graph has the correct shapes and types. Moreover the function should
  be a new, untraced function, to allow `tf.saved_model.save` to make a fresh
  trace of the function.


  ```python
  tf.saved_model.save(model,
                      export_dir=...,
                      signatures={'serving': serve_examples})
  ```

  The exported model can now be loaded (in python or c) and used for serving:

  ```python
  imported = tf.saved_model.load(...)
  predict_fn = imported.signatures['serving']
  predict_fn(...)
  ```

  ## Using this layer on CPU

  This layer can also be instantiated under a CPU strategy and used for local
  testing/training. The model created in such a way are checkpoint compatible
  with models created under `TPUStrategy`. In order to achieve checkpoint
  compatibility, you must use a Keras optimizers (or ones converted by
  `translate_keras_optimizer`) as your optimizers.

  In the simplest case, where you use the same optimizer for your embedding and
  dense layers, the `training_step` above will function exactly the same in both
  situations.

  If you use a separate Keras optimizer for your embedding layers (e.g. you want
  a different hyper parameter setting or an entirely different algorithm),
  special care must be observed to keep things the same. To understand why,
  there are a few technical details you need to know:

  When created under `TPUStrategy` the underlying table variables are not
  considered trainable and are not available under `model.trainable_variables`.
  The main reason for this is that the table variables are just a stand-in for
  the real data which lives in the HBM of the TPU. These variables are stale and
  are only updated when saving and restoring checkpoints.

  Because of this a standard `optimizer.apply_gradient` will not work on these
  variables. Instead a separate virtual trainable variable is added to the list
  of trainable variables and simply computing the gradient of this variable will
  cause the gradient for the embeddings to be computed *and the optimizer
  applied*.

  When created under a CPU strategy, the table variables are created normally
  are part of the model's trainiable variables. In this case, if you are using
  a different optimizer to embedding tables, you must manually partition the
  variables and gradients so that you can use the Keras optmizer you created for
  embedding tables on the tables.

  E.g.,

  ```python
  class ModelWithSeparateOptimizer(tf.keras.Model):
    def __init__(self, optimizer):
      self.embedding_layer = tpu_embedding_layer.TPUEmbedding(
          feature_config=feature_config,
          optimizer=optimizer)

    def call(self, inputs):
      embedding = self.embedding_layer(inputs)
      logits = tf.keras.layers.Dense(1)(
        tf.concat(tf.nest.flatten(embedding), axis=1)
      )

  with strategy.scope():
    embedding_optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    dense_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model = ModelWithSeparateOptimizer(embedding_optimizer)

  @tf.function
  def training_step(dataset_iterator, num_steps):
    def tpu_step(inputs):
      labels, features = inputs
      with tf.GradientTape() as tape:
        model_output = model(features)
        loss = ...  # some function of labels and model_output

      gradients = tape.gradient(loss, model.trainable_variables)
      grads_and_vars = zip(gradients, model.trainable_variables)

      # Note the use of 'id' here: 'x in y' uses x's equality method and if x is
      # a tensor this tf.math.equal rather than python object equality.
      embedding_var_ids = [
          id(v) for v in model.embedding_layer.trainable_variables]
      dense_grads_and_vars = [
          (g, v) for g, v in grads_and_vars
          if id(v) not in embedding_var_ids
      dense_optimizer.apply_gradients(dense_grads_and_vars)

      embedding_grads_and_vars = [
          (g, v) for g, v in grads_and_vars
          if id(v) in embedding_var_ids]
      embedding_optimizer.apply_gradients(embedding_grads_and_vars)

    for _ in tf.range(num_steps):
      strategy.run(tpu_step, args=(next(dataset_iterator), ))
  ```

  The above training step works both on TPU and on CPU.
  """

  def __init__(self, feature_config, optimizer,
               pipeline_execution_with_tensor_core=False,
               batch_size=None):
    """A Keras layer for accelerated embedding lookups on TPU.

    Args:
      feature_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` configs.
      optimizer: An instance of one of `tf.tpu.experimental.embedding.SGD`,
        `tf.tpu.experimental.embedding.Adagrad` or
        `tf.tpu.experimental.embedding.Adam`, a Keras optimizer or a string name
        of an optimizer (see `tf.keras.optimizers.get`). Or, if not created
        under a TPU strategy, None, which will avoid creation of the optimizer
        slot variable do reduce memory consumption during export.
      pipeline_execution_with_tensor_core: If True, the TPU embedding
        computations will overlap with the TensorCore computations (and hence
        will be one step old with potential correctness drawbacks). Set to True
        for improved performance.
      batch_size: If set, this will be used as the global batch size and
        overrides the autodetection of the batch size from the layer's input.
        This is necesarry if all inputs to the layer's call are SparseTensors.
    """
    super().__init__()
    self._feature_config, self._table_config_map = (
        _clone_and_prepare_features(feature_config))
    self._optimizer = _normalize_and_prepare_optimizer(optimizer)
    self._batch_size = batch_size

    self._tpu_embedding = tf.tpu.experimental.embedding.TPUEmbedding(
        self._feature_config,
        self._optimizer,
        pipeline_execution_with_tensor_core)

    self._tpu_call_id = 0

  def build(self, input_shape):
    super().build(input_shape)

    self._strategy = tf.distribute.get_strategy()
    self._using_tpu = _is_tpu_strategy(self._strategy)

    if self._batch_size is None and self._using_tpu:
      self._batch_size = _get_batch_size_from_input_shapes(input_shape)

    self._tpu_embedding.build(self._batch_size)

    if self._using_tpu:
      # Note that self.tpu_embedding_helper_dummy matches _DUMMY_NAME above,
      # or it will appear twice in the list of saveables. Note that the Python
      # variable name should be _DUMMY_NAME too, as it is used to name internal
      # objects: we enforce that by creating it with setattr.
      setattr(
          self, _DUMMY_NAME,
          self.add_weight(
              name=_DUMMY_NAME,
              shape=(1,),
              initializer=tf.zeros_initializer(),
              trainable=True,
              dtype=tf.float32))
    else:
      # When on CPU, ensure that the embedding tables are part of the trainable
      # variables list for this layer.
      for _, weight in self._tpu_embedding.embedding_tables.items():
        self._trainable_weights.append(weight)

  def _tpu_embedding_lookup(self, features, weights):
    """Uses TPU embedding lookup for embedding ids in features.

    Args:
      features: A nested structure of `tf.Tensor`s, `tf.SparseTensor`s or
        `tf.RaggedTensor`s, with the same structure as `feature_config` used
        when initializing this layer. Inputs will be downcast to `tf.int32`.
        Only one type out of `tf.SparseTensor` or `tf.RaggedTensor` is supported
        per call.
      weights: If not `None`, a nested structure of `tf.Tensor`s,
        `tf.SparseTensor`s or `tf.RaggedTensor`s, matching the above, except
        that the tensors should be of float type (and they will be downcast to
        `tf.float32`). For `tf.SparseTensor`s we assume the `indices` are the
        same for the parallel entries from `features` and similarly for
        `tf.RaggedTensor`s we assume the `row_splits` are the same.

    Returns:
      A dict of looked up embedding tensors with keys matching those of
      features_to_config_dict.
    """
    # Each call to this function increments the _tpu_call_id by 1, this allows
    # us to tag each of the main embedding ops with this call id so that we know
    # during graph rewriting passes which ops correspond to the same layer call.
    self._tpu_call_id += 1
    name = "{}".format(self._tpu_call_id)

    # Set training to true, even during eval. When name is set, this will
    # trigger a pass that updates the training based on if there is a send
    # gradients with the same name.
    self._tpu_embedding.enqueue(features, weights, training=True, name=name)

    # The gradient trap is a trick used to ensure we can compute the gradients
    # at the correct point of the model. By default GradientTape only tracks
    # the calculations which descend from variables. e.g. if you call
    # tape.gradient on something that does not come from a variable involved in
    # the computation, it will fail.
    # We need to call tpu_embedding.apply_gradients on the gradients computed
    # at tpu_embedding.dequeue. Since tpu_embedding.dequeue has no inputs, we
    # can't compute the gradient at its output. To get around that we wrap
    # the dequeue in a function with a custom gradient. This function takes one
    # input, throws it away and returns the result of the dequeue. If we pass a
    # dummy variable to this function and compute the gradient at the dummy
    # variable, then the custom gradient function will be called with the
    # graidents that we need to pass to tpu_embedding.apply_gradients.
    @tf.custom_gradient
    def gradient_trap(dummy):
      """Register a gradient function for activation.

      Its purpose is to send gradients back to TPU.

      Args:
        dummy: a variable to prevent this backward pass from being pruned.

      Returns:
        a tuple of list of activations and their gradient function.
      """
      activations = self._tpu_embedding.dequeue(name=name)

      def grad(*grad_wrt_activations):
        """Gradient function."""
        # Since the output of the function is flattened, the gradients
        # are also flattened. Hence we have to pack them back in to the correct
        # nested structure.
        gradients = tf.nest.pack_sequence_as(self._feature_config,
                                             grad_wrt_activations)
        self._tpu_embedding.apply_gradients(gradients, name=name)

        # This is the gradient for the input variable.
        return tf.zeros_like(dummy)

      # Custom gradient functions don't like nested structures of tensors, so we
      # flatten them here.
      return tf.nest.flatten(activations), grad

    activations_with_trap = gradient_trap(getattr(self, _DUMMY_NAME))
    return tf.nest.pack_sequence_as(self._feature_config, activations_with_trap)

  def call(self, features, weights=None, serving_config=None):
    """Look up features in the embedding tables and combine using weights.

    Args:
      features: a nested structure of `Tensor`s, `SparseTensor`s or
        `RaggedTensor`s with the same structure as `feature_config`. These
        tensors are used as ids to lookup rows in the embedding tables using the
        config as specified in the corresponding entry of `feature_config`. You
        can mix `Tensor`s and `SparseTensor`s, or `Tensor`s and `RaggedTensor`s,
        but not `SparseTensor`s and `RaggedTensor`s.
      weights: None, or a nested structure of Tensor`s, `SparseTensor`s or
        `RaggedTensor`s or None matching features. These are the weights used
        when combining the looked up rows for a given feature and examples. If
        None, weights of 1 will be used.
      serving_config: A nested structure of
        `tf.tpu.experimental.embedding.FeatureConfig` objects. If not None, this
        layer uses CPU based lookup using serving_config and the current set of
        embedding tables.

    Returns:
      The combined embedding activations for the input ids passed in via
      features.

    Raises:
      RuntimeError: If layer is not created under a TPU strategy and is called
        under a TPU strategy.
    """
    if serving_config is not None:
      # The TableConfig objects in the serving_config should match the ones
      # passed to the layer when it was created. Since we cloned those, we
      # need to update to the new TableConfig objects. Use the stored mapping
      # to do this.
      serving_config = _update_table_configs(serving_config,
                                             self._table_config_map)
      return tf.tpu.experimental.embedding.serving_embedding_lookup(
          features, weights, self._tpu_embedding.embedding_tables,
          serving_config)

    if not self._using_tpu and _is_tpu_strategy(tf.distribute.get_strategy()):
      raise RuntimeError("Layer is created under strategy %s but is being "
                         "called under a TPUStrategy. Please create the layer "
                         "under a TPUStrategy if you wish to run the layer on "
                         "TPU." % self.strategy)
    if self._using_tpu and _is_tpu_strategy(tf.distribute.get_strategy()):
      return self._tpu_embedding_lookup(features, weights)
    else:
      return tf.tpu.experimental.embedding.serving_embedding_lookup(
          features, weights, self._tpu_embedding.embedding_tables,
          self._feature_config)

  @property
  def embedding_tables(self):
    """A mapping from table configs to tables.

    When instantiated under a TPU strategy, this returns a sharded variable.
    This variable is strictly a placeholder used for saving and restoring.
    Attempting to assign values to this variable will not update the actual
    embedding tables and reading may result in reading a stale copy of the
    table. Should not be used for actual computation, only for exporting the
    model for serving.

    Returns:
      A dictionary of tables, keyed by the
      `tf.tpu.experimental.embedding.TableConfig` objected used in the
      `feature_config` passed to this layer's init.
    """
    tables = self._tpu_embedding.embedding_tables
    # Use the table config map to map from the cloned configs back to the
    # configs that where passed into the layer on init.
    return {old_config: tables[new_config]
            for old_config, new_config in self._table_config_map}

  @property
  def _checkpoint_dependencies(self):
    """All dependencies of this object.

    We use a dummy tensor to work around Keras pruning the backwards pass.
    We strip it here to ensure we don't save this tensor in the checkpoint.

    Returns:
      A list of `TrackableReference` objects indicating named
      `Trackable` dependencies which should be saved along with this
      object.
    """
    return [
        dep
        for dep in super()._checkpoint_dependencies
        if dep.name != _DUMMY_NAME
    ]


def _get_slot_variable_creation_fn(optimizer):
  """Create a function that uses the optimizer's add_slot to create slots."""

  no_dependency_fn = tf.Module()._no_dependency  # pylint: disable=protected-access

  # This is needed so that the mid level API can create slots using a user
  # passed optimizer rather than the built-in methods. This allows a user to
  # train the same model on CPU and TPU.
  def slot_variable_creation_fn(table, slot_names, slot_initializers):
    slots = {}
    for slot, initializer in zip(slot_names, slot_initializers):
      slots[slot] = no_dependency_fn(
          optimizer.add_slot(table, _SLOT_NAME_MAPPING[type(optimizer)][slot],
                             initializer))
    return slots
  return slot_variable_creation_fn


def translate_keras_optimizer(optimizer):
  """Translates a Keras optimizer to the tf.tpu.experimental.embedding version.

  Note that Keras optimizer params can accept Tensors or callables, whereas
  tpu_embedding optimizer params require floats. We call .get_config() on the
  Keras params, which evaluates each param immediately.

  NOTE: that the underlying Keras optimizer passed in will be used to create the
  slot variables for the embedding tables this optimizer is used for.

  Args:
    optimizer: A Keras optimizer parameter object.

  Raises:
    ValueError: if passed a Keras optimizer defining parameters unsupported by
        the corresponding tpu_embedding object, or an unsupported Keras
        optimizer.

  Returns:
    the tpu_embedding parameter object corresponding to optimizer.
  """

  if optimizer.__class__ in _OPTIMIZER_PARAMETERS:
    embedding_optimizer, supported, unsupported = (
        _OPTIMIZER_PARAMETERS[optimizer.__class__])
    config = optimizer.get_config()
    # We need to handle learning_rate specially so that we can properly support
    # dynamic learning rate. Depending on what the user passed for learning_rate
    # get_config does a few different things:
    # 1.  If it was a function, it calls the function (which we do not want, as
    #     we want to call the function in the strategy context so that all
    #     ops in the function are placed on the TPU). In this case the return
    #     type should generally be a tensor.
    # 2.  If it was a LearningRateSchedule, get_config calls
    #     serialize_keras_object on the schedule oject. In this case the return
    #     type is a dict.
    # 3.  A python numeric constant or something convertible to one.
    if isinstance(config["learning_rate"], tf.Tensor):
      config["learning_rate"] = lambda: optimizer.get_config()["learning_rate"]
    elif isinstance(config["learning_rate"], dict):
      schedule = tf.keras.optimizers.schedules.deserialize(
          config["learning_rate"])
      config["learning_rate"] = lambda: schedule(optimizer.iterations)

    # Check to make sure only support params are set?
    _ensure_unsupported_params_unchanged(optimizer, supported, unsupported)

    params = {k: config[k] for k in supported}
    # If the optimizer has slots, add the slot variable creation fn.
    if optimizer.__class__ in _SLOT_NAME_MAPPING:
      params["slot_variable_creation_fn"] = _get_slot_variable_creation_fn(
          optimizer)

    return embedding_optimizer(**params)

  elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
    raise ValueError("Keras optimizer %s is unsupported for TPU Embedding."
                     % optimizer.__class__.__name__)
  else:
    raise ValueError("%s is an unsupported optimizer class. Please pass a "
                     "Keras optimizer." % optimizer.__class__.__name__)


def _ensure_unsupported_params_unchanged(
    optimizer_params, supported_params, unsupported_params):
  """Helper function to raise exception if an unsupported param was set.

  The unsupported params generally have default values which we cannot
  rely upon to be falsy. Instead of duplicating the default values here
  in a way that is likely to drift out of sync, we construct a second
  copy of the optimizer param object and diff the config fields.
  The parameters "clipnorm" and "clipvalue" are universally unsupported and
  undefined by default, so we check these directly.

  Args:
    optimizer_params: The Keras optimizer param object.
    supported_params: The list of config options on the Keras optimizer which
        we will pass to the constructor.
    unsupported_params: The list of config options which must not be set
        on the Keras optimizer.

  Raises:
    ValueError: if the Keras optimizer set a config option which the
        tpu_embedding optimizer does not support.
  """
  error_template = (
      "Optimizer parameter %s is unsupported for TPU embeddings. Please "
      "construct a new optimizer for embedding if you wish to use this "
      "setting for model training.")

  for attr in ["clipnorm", "clipvalue"]:
    if getattr(optimizer_params, attr, None) is not None:
      raise ValueError(error_template % attr)

  config = optimizer_params.get_config()
  constructor_args = {p: config[p] for p in supported_params}
  reference = optimizer_params.__class__(**constructor_args)
  reference_config = reference.get_config()
  for p in unsupported_params:
    if config[p] != reference_config[p]:
      raise ValueError(error_template % p)
