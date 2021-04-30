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
        ":blocks",
        ":factorized_top_k",
        ":loss",
        "//tensorflow_recommenders/layers/embedding",
        "//tensorflow_recommenders/layers/feature_interaction",
        "//tensorflow_recommenders/layers/feature_interaction:dcn",
    ],
)

py_library(
    name = "blocks",
    srcs = ["blocks.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_recommenders:types",
    ],
)

py_library(
    name = "factorized_top_k",
    srcs = ["factorized_top_k.py"],
    srcs_version = "PY3",
    deps = [
        # py/scann dep,
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
    name = "factorized_top_k_test",
    size = "large",
    srcs = ["factorized_top_k_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":factorized_top_k",
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
