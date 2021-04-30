# Placeholder for internal Python library.

# TensorFlow Recommenders, a TensorFlow library for building recommender systems.
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_recommenders",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":types",
        "//tensorflow_recommenders/examples",
        "//tensorflow_recommenders/experimental",
        # Placeholder for internal extension library.
        "//tensorflow_recommenders/layers",
        "//tensorflow_recommenders/metrics",
        "//tensorflow_recommenders/models",
        "//tensorflow_recommenders/tasks",
    ],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)
