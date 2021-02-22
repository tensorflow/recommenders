load("//devtools/python/blaze:pytype.bzl", "py_library", "pytype_library")
# Placeholder for internal Python test library.

licenses(["notice"])

package(
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":composite_optimizer",
    ],
)

pytype_library(
    name = "composite_optimizer",
    srcs = ["composite_optimizer.py"],
    srcs_version = "PY3",
    deps = [
        # py/tensorflow dep,
    ],
)

py_test(
    name = "composite_optimizer_test",
    srcs = [
        "composite_optimizer_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":composite_optimizer",
        # py/absl/testing:parameterized dep,
        # py/tensorflow dep,
    ],
)
