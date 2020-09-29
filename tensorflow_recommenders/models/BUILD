# Placeholder for internal Python library.
# Placeholder for internal Python test library.

package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)

py_test(
    name = "base_test",
    srcs = ["base_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        # py/numpy dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders/metrics",
        "//tensorflow_recommenders/tasks",
    ],
)
