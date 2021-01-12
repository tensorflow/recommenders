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

# Lint as: python3
"""Functions supporting Movielens examples."""

import array
import collections

from typing import Dict, List, Optional, Text, Tuple

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


def _create_feature_dict() -> Dict[Text, List[tf.Tensor]]:
  """Helper function for creating an empty feature dict for defaultdict."""
  return {"movie_title": [], "user_rating": []}


def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Function for sampling a list example from given feature lists."""
  if random_state is None:
    random_state = np.random.RandomState()

  sampled_indices = random_state.choice(
      range(len(feature_lists["movie_title"])),
      size=num_examples_per_list,
      replace=False,
  )
  sampled_movie_titles = [
      feature_lists["movie_title"][idx] for idx in sampled_indices
  ]
  sampled_ratings = [
      feature_lists["user_rating"][idx]
      for idx in sampled_indices
  ]

  return (
      tf.concat(sampled_movie_titles, 0),
      tf.concat(sampled_ratings, 0),
  )


def sample_listwise(
    rating_dataset: tf.data.Dataset,
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
  """Function for converting the MovieLens 100K dataset to a listwise dataset.

  Args:
      rating_dataset:
        The MovieLens ratings dataset loaded from TFDS with features
        "movie_title", "user_id", and "user_rating".
      num_list_per_user:
        An integer representing the number of lists that should be sampled for
        each user in the training dataset.
      num_examples_per_list:
        An integer representing the number of movies to be sampled for each list
        from the list of movies rated by the user.
      seed:
        An integer for creating `np.random.RandomState`.

  Returns:
      A tf.data.Dataset containing list examples.

      Each example contains three keys: "user_id", "movie_title", and
      "user_rating". "user_id" maps to a string tensor that represents the user
      id for the example. "movie_title" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.string. It represents the list
      of candidate movie ids. "user_rating" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.float32. It represents the
      rating of each movie in the candidate list.
  """
  random_state = np.random.RandomState(seed)

  example_lists_by_user = collections.defaultdict(_create_feature_dict)

  movie_title_vocab = set()
  for example in rating_dataset:
    user_id = example["user_id"].numpy()
    example_lists_by_user[user_id]["movie_title"].append(
        example["movie_title"],)
    example_lists_by_user[user_id]["user_rating"].append(
        example["user_rating"],
    )
    movie_title_vocab.add(example["movie_title"].numpy())

  tensor_slices = {"user_id": [], "movie_title": [], "user_rating": []}

  for user_id, feature_lists in example_lists_by_user.items():
    for _ in range(num_list_per_user):

      # Drop the user if they don't have enough ratings.
      if len(feature_lists["movie_title"]) < num_examples_per_list:
        continue

      sampled_movie_titles, sampled_ratings = _sample_list(
          feature_lists,
          num_examples_per_list,
          random_state=random_state,
      )
      tensor_slices["user_id"].append(user_id)
      tensor_slices["movie_title"].append(sampled_movie_titles)
      tensor_slices["user_rating"].append(sampled_ratings)

  return tf.data.Dataset.from_tensor_slices(tensor_slices)
