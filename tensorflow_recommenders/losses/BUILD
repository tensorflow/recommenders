# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "batch_softmax",
    srcs = ["batch_softmax.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers:loss",
    ],
)

py_library(
    name = "losses",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":batch_softmax",
    ],
)

# Tests
py_test(
    name = "batch_softmax_test",
    srcs = ["batch_softmax_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":batch_softmax",
        # py/absl/testing:parameterized dep,
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)
