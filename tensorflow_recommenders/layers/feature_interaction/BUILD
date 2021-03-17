# Placeholder for internal Python library.
# Placeholder for internal Python test library.

licenses(["notice"])

package(
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "feature_interaction",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":dcn",
        ":dot_interaction",
    ],
)

py_library(
    name = "dcn",
    srcs = ["dcn.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_test(
    name = "dcn_test",
    srcs = ["dcn_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dcn",
        ":feature_interaction",
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_library(
    name = "dot_interaction",
    srcs = ["dot_interaction.py"],
    srcs_version = "PY3",
    deps = [
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)

py_test(
    name = "dot_interaction_test",
    srcs = ["dot_interaction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dot_interaction",
        ":feature_interaction",
        # py/numpy dep,
        # py/tensorflow dep,
    ],
)
