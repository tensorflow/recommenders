# Placeholder for internal Python library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":movielens",
    ],
)

py_library(
    name = "movielens",
    srcs = ["movielens.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/requests dep,
        # py/tensorflow dep,
    ],
)
