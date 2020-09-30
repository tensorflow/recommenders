# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":ann",
        ":corpus",
        ":loss",
    ],
)

py_library(
    name = "ann",
    srcs = ["ann.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "corpus",
    srcs = ["corpus.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "loss",
    srcs = ["loss.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

# Tests
py_test(
    name = "ann_test",
    srcs = ["ann_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ann",
        # py/absl/testing:parameterized dep,
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_test(
    name = "corpus_test",
    srcs = ["corpus_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":corpus",
        # py/absl/testing:parameterized dep,
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_test(
    name = "loss_test",
    srcs = ["loss_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":loss",
        # py/absl/testing:parameterized dep,
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)
