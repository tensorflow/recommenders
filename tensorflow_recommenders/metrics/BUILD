# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "metrics",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":factorized_top_k",
    ],
)

py_library(
    name = "factorized_top_k",
    srcs = ["factorized_top_k.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers",
    ],
)

py_test(
    name = "factorized_top_k_test",
    srcs = ["factorized_top_k_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics",
        # py/absl/testing:parameterized dep,
        # py/numpy dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers",
    ],
)
