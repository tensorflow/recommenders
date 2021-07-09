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

"""Tests for DotInteraction layer."""

import numpy as np
import tensorflow as tf

from tensorflow_recommenders.layers.feature_interaction.dot_interaction import DotInteraction


class DotInteractionTest(tf.test.TestCase):

  def test_valid_input(self):
    feature1 = np.asarray([[0.1, -4.3, 0.2, 1.1, 0.3]]).astype(np.float32)
    feature2 = np.asarray([[2.0, 3.2, -1.0, 0.0, 1.0]]).astype(np.float32)
    feature3 = np.asarray([[0.0, 1.0, -3.0, -2.2, -0.2]]).astype(np.float32)
    layer = DotInteraction(self_interaction=True,
                           skip_gather=False)

    f11 = np.dot(feature1[0], feature1[0])
    f12 = np.dot(feature1[0], feature2[0])
    f13 = np.dot(feature1[0], feature3[0])
    f22 = np.dot(feature2[0], feature2[0])
    f23 = np.dot(feature2[0], feature3[0])
    f33 = np.dot(feature3[0], feature3[0])

    output = layer([feature1, feature2, feature3])
    self.assertAllClose(np.asarray([[f11,
                                     f12, f22,
                                     f13, f23, f33]]), output)

    layer = DotInteraction(self_interaction=True,
                           skip_gather=True)
    output = layer([feature1, feature2, feature3])

    self.assertAllClose(np.asarray([[f11, 0, 0,
                                     f12, f22, 0,
                                     f13, f23, f33]]), output)

    layer = DotInteraction(self_interaction=False,
                           skip_gather=False)
    output = layer([feature1, feature2, feature3])
    self.assertAllClose(np.asarray([[f12,
                                     f13, f23]]), output)

    layer = DotInteraction(self_interaction=False,
                           skip_gather=True)
    output = layer([feature1, feature2, feature3])

    self.assertAllClose(np.asarray([[0, 0, 0,
                                     f12, 0, 0,
                                     f13, f23, 0]]), output)

  def test_non_matching_dimensions(self):
    with self.assertRaisesRegexp(ValueError, r"dimensions must be equal"):
      feature1 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
      feature2 = np.asarray([[2.0, -1.0, 1.0]]).astype(np.float32)
      feature3 = np.asarray([[0.0, 1.0]]).astype(np.float32)
      layer = DotInteraction()
      layer([feature1, feature2, feature3])


if __name__ == "__main__":
  tf.test.main()
