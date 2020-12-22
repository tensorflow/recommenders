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

"""TensorFlow Recommenders, a TensorFlow library for recommender systems."""

import pathlib
import setuptools

VERSION = "0.3.2"

REQUIRED_PACKAGES = [
    "absl-py >= 0.1.6",
    "tensorflow >= 2.3",
]

long_description = (pathlib.Path(__file__).parent
                    .joinpath("README.md")
                    .read_text())

setuptools.setup(
    name="tensorflow-recommenders",
    version=VERSION,
    description="Tensorflow Recommenders, a TensorFlow library for recommender systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorflow/recommenders",
    author="Google Inc.",
    author_email="packages@tensorflow.org",
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        "docs": [
            "fire",
            "annoy",
            "scann == 1.1.1",
            "tensorflow == 2.3, < 2.4",],
    },
    # PyPI package information.
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="Apache 2.0",
    keywords="tensorflow recommenders recommendations",
)
