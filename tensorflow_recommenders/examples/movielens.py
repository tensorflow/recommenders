# Copyright 2020 The TensorFlow Recommenders Authors.
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

# Lint as: python3
"""Functions supporting Movielens examples."""

import array
import collections

from typing import Dict, Optional, Text

import numpy as np
import tensorflow as tf


def evaluate(user_model: tf.keras.Model,
             movie_model: tf.keras.Model,
             test: tf.data.Dataset,
             movies: tf.data.Dataset,
             train: Optional[tf.data.Dataset] = None,
             k: int = 10) -> Dict[Text, float]:
  """Evaluates a Movielens model on the supplied datasets.

  Args:
    user_model: User representation model.
    movie_model: Movie representation model.
    test: Test dataset.
    movies: Dataset of movies.
    train: Training dataset. If supplied, recommendations for training watches
      will be removed.
    k: The cutoff value at which to compute precision and recall.

  Returns:
   Dictionary of metrics.
  """

  movie_ids = np.concatenate(
      list(movies.batch(1000).map(lambda x: x["movie_id"]).as_numpy_iterator()))

  movie_vocabulary = dict(zip(movie_ids.tolist(), range(len(movie_ids))))

  train_user_to_movies = collections.defaultdict(lambda: array.array("i"))
  test_user_to_movies = collections.defaultdict(lambda: array.array("i"))

  if train is not None:
    for row in train.as_numpy_iterator():
      user_id = row["user_id"]
      movie_id = movie_vocabulary[row["movie_id"]]
      train_user_to_movies[user_id].append(movie_id)

  for row in test.as_numpy_iterator():
    user_id = row["user_id"]
    movie_id = movie_vocabulary[row["movie_id"]]
    test_user_to_movies[user_id].append(movie_id)

  movie_embeddings = np.concatenate(
      list(movies.batch(4096).map(
          lambda x: movie_model({"movie_id": x["movie_id"]})
      ).as_numpy_iterator()))

  precision_values = []
  recall_values = []

  for (user_id, test_movies) in test_user_to_movies.items():
    user_embedding = user_model({"user_id": np.array([user_id])}).numpy()
    scores = (user_embedding @ movie_embeddings.T).flatten()

    test_movies = np.frombuffer(test_movies, dtype=np.int32)

    if train is not None:
      train_movies = np.frombuffer(
          train_user_to_movies[user_id], dtype=np.int32)
      scores[train_movies] = -1e6

    top_movies = np.argsort(-scores)[:k]
    num_test_movies_in_k = sum(x in top_movies for x in test_movies)
    precision_values.append(num_test_movies_in_k / k)
    recall_values.append(num_test_movies_in_k / len(test_movies))

  return {
      "precision_at_k": np.mean(precision_values),
      "recall_at_k": np.mean(recall_values)
  }
