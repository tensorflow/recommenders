package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

py_binary(
    name = "build_api_docs",
    srcs = ["build_api_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/fire dep,
        # py/tensorflow_docs/api_generator dep,
        "//tensorflow_recommenders",
    ],
)
