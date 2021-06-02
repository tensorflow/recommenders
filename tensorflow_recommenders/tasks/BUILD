# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

exports_files(["retrieval_test.py"])

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "ranking",
    srcs = ["ranking.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # py/tensorflow dep,
    ],
)

py_library(
    name = "retrieval",
    srcs = ["retrieval.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers",
        "//tensorflow_recommenders/metrics",
    ],
)

py_library(
    name = "tasks",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":ranking",
        ":retrieval",
    ],
)

# Tests
py_test(
    name = "ranking_test",
    srcs = ["ranking_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ranking",
        # py/tensorflow dep,
    ],
)

py_test(
    name = "retrieval_test",
    srcs = ["retrieval_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":retrieval",
        # py/numpy dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders/metrics",
    ],
)
