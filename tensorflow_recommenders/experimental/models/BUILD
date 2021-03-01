# Placeholder for internal Python library.
# Placeholder for internal Python test library.

licenses(["notice"])

package(
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "models",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":ranking_model",
    ],
)

py_library(
    name = "ranking_model",
    srcs = ["ranking_model.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers/embedding",
        "//tensorflow_recommenders/models",
        "//tensorflow_recommenders/tasks",
    ],
)

py_test(
    name = "ranking_model_test",
    srcs = [
        "ranking_model_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders",
    ],
)
