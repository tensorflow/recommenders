# Placeholder for internal Python library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "examples",
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
        # py/tensorflow dep,
    ],
)

py_binary(
    name = "nbtool",
    srcs = ["nbtool.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/fire dep,
        # py/nbformat dep,
    ],
)
