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

# Lint-as: python3
"""The movielens datasets.

MovieLens datasets from the GroupLens group at the University of Minnesota
(https://grouplens.org/datasets/movielens/)..

MovieLens data sets were collected by the GroupLens Research Project
at the University of Minnesota.

Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:

- The user may not state or imply any endorsement from the
  University of Minnesota or the GroupLens Research Group.
- The user must acknowledge the use of the data set in
  publications resulting from the use of the data set
  (see below for citation information).
- The user may not redistribute the data without separate
  permission.
- The user may not use this information for any commercial or
  revenue-bearing purposes without first obtaining permission
  from a faculty member of the GroupLens Research Project at the
  University of Minnesota.

If you have any further questions or comments, please contact GroupLens
at grouplens-info@cs.umn.edu.

Citation:

@article{harper2016movielens,
  title={The movielens datasets: History and context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={Acm transactions on interactive intelligent systems (tiis)},
  volume={5},
  number={4},
  pages={19},
  year={2016},
  publisher={ACM}
}
"""

import os
import pathlib
import shutil
from typing import Dict, Text, Tuple
import zipfile

import numpy as np
import requests
import tensorflow as tf

CACHE_PATH = pathlib.Path.home() / ".tfrs/"


class _Movielens:
  """Downloads and parses Movielens data."""
  NAME = ""
  URL = ""

  def __init__(self) -> None:
    self._name = self.NAME
    self._url = self.URL

  def archive_path(self):
    return CACHE_PATH / (self._name + ".zip")

  def data_dir(self):
    return CACHE_PATH / self._name

  def download(self):
    """Downloads the data archive."""
    if not self.data_dir().exists():
      os.makedirs(self.data_dir())

    temp_path = str(self.archive_path()) + ".partial"

    with requests.get(self._url, stream=True) as response:
      with open(temp_path, "wb") as destination:
        for chunk in response.iter_content(chunk_size=4096):
          destination.write(chunk)

    shutil.move(temp_path, str(self.archive_path()))

    archive = zipfile.ZipFile(self.archive_path())
    archive.extractall(path=str(self.data_dir()))

  def parse_ratings(self) -> Dict[Text, np.ndarray]:
    raise NotImplementedError()

  def parse_candidates(self) -> Dict[Text, np.ndarray]:
    raise NotImplementedError()

  def save(self, path: pathlib.Path, data: Dict[Text, np.ndarray]):
    np.savez(path, **data)

  def load(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Loads datasets."""

    ratings_path = self.data_dir() / "ratings.npz"
    candidates_path = self.data_dir() / "candidates.npz"

    if not (ratings_path.exists() and candidates_path.exists()):
      self.download()

      np.savez(ratings_path, **self.parse_ratings())
      np.savez(candidates_path, **self.parse_candidates())

    return (
        tf.data.Dataset.from_tensor_slices(dict(np.load(ratings_path))),
        tf.data.Dataset.from_tensor_slices(dict(np.load(candidates_path)))
    )


class _Movielens100K(_Movielens):
  """Movielens 100K dataset."""

  NAME = "movielens_100K"
  URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

  def parse_ratings(self):
    ratings_data = tf.data.experimental.CsvDataset(
        filenames=str(self.data_dir() / "ml-100k" / "u.data"),
        field_delim="\t",
        record_defaults=[tf.int32, tf.int32, tf.float32, tf.int32],
        header=False)

    def _add_names(user_id, movie_id, rating, timestamp):
      return {
          "user_id": user_id,
          "movie_id": movie_id,
          "rating": rating,
          "timestamp": timestamp
      }

    return list(
        ratings_data.map(_add_names).batch(100_000).as_numpy_iterator())[0]

  def parse_candidates(self):
    candidates_header = (
        "movie_id", "movie_title", "release_date", "video_release_date",
        "IMDb URL", "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    )
    candidates_data = tf.data.experimental.CsvDataset(
        filenames=str(self.data_dir() / "ml-100k" / "u.item"),
        field_delim="|",
        record_defaults=(
            [tf.int32, tf.string, "", "", ""] +
            [tf.float32] * 19
        ),
        header=False)

    add_header = lambda *x: dict(zip(candidates_header, x))

    candidates_data = list(
        candidates_data.batch(100_000).map(add_header)
        .as_numpy_iterator())[0]

    # Convert numpy object dtype to string.
    for key, value in candidates_data.items():
      if value.dtype == object:
        candidates_data[key] = value.astype(np.string_)

    return candidates_data


class _Movielens20M(_Movielens):
  """Movielens 20M dataset."""

  NAME = "movielens_20M"
  URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"

  def parse_ratings(self):
    ratings_data = tf.data.experimental.CsvDataset(
        filenames=str(self.data_dir() / "ml-20m" / "ratings.csv"),
        field_delim=",",
        record_defaults=[tf.int32, tf.int32, tf.float32, tf.int32],
        header=True)

    def _add_names(user_id, movie_id, rating, timestamp):
      return {
          "user_id": user_id,
          "movie_id": movie_id,
          "rating": rating,
          "timestamp": timestamp
      }

    return list(
        ratings_data.map(_add_names).batch(20_000_000).as_numpy_iterator())[0]

  def parse_candidates(self):
    candidates_header = (
        "movie_id", "movie_title"
    )
    candidates_data = tf.data.experimental.CsvDataset(
        filenames=str(self.data_dir() / "ml-20m" / "movies.csv"),
        field_delim=",",
        record_defaults=(
            [tf.int32, tf.string, tf.string]
        ),
        header=True)

    add_header = lambda *x: dict(zip(candidates_header, x[:2]))

    candidates_data = list(
        candidates_data.batch(100_000).map(add_header)
        .as_numpy_iterator())[0]

    # Convert numpy object dtype to string.
    for key, value in candidates_data.items():
      if value.dtype == object:
        candidates_data[key] = value.astype(np.string_)

    return candidates_data


def movielens_100K() -> Tuple[tf.data.Dataset, tf.data.Dataset]:  # pylint: disable=invalid-name
  """Movielens 100K dataset.

  Returns:
    A tuple of (ratings dataset, movie metadata dataset).
  """

  return _Movielens100K().load()


def movielens_20M() -> Tuple[tf.data.Dataset, tf.data.Dataset]:  # pylint: disable=invalid-name
  """Movielens 20M dataset.

  Returns:
    A tuple of (ratings dataset, movie metadata dataset).
  """

  return _Movielens20M().load()
