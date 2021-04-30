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
        ":ranking",
    ],
)

py_library(
    name = "ranking",
    srcs = ["ranking.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
        "//tensorflow_recommenders/layers",
        "//tensorflow_recommenders/layers/feature_interaction",
        "//tensorflow_recommenders/models",
        "//tensorflow_recommenders/tasks",
    ],
)

py_test(
    name = "ranking_test",
    srcs = [
        "ranking_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
        "//tensorflow_recommenders",
    ],
)
