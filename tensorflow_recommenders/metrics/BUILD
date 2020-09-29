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
        ":corpus",
    ],
)

py_library(
    name = "corpus",
    srcs = ["corpus.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers",
    ],
)

py_test(
    name = "corpus_test",
    srcs = ["corpus_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics",
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)
