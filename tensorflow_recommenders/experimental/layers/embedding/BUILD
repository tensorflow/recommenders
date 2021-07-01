# Placeholder for internal Python library.
# Placeholder for internal Python test library.

licenses(["notice"])

package(
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "embedding",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":partial_tpu_embedding",
    ],
)

py_library(
    name = "partial_tpu_embedding",
    srcs = ["partial_tpu_embedding.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers/embedding:tpu_embedding_layer",
    ],
)

py_test(
    name = "partial_tpu_embedding_test",
    timeout = "short",
    srcs = ["partial_tpu_embedding_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders",
    ],
)
