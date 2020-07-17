package(
    default_visibility = [
        "//tensorflow_recommenders:__subpackages__",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_binary(
    name = "docs",
    srcs = ["docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # py/fire dep,
        # py/tensorflow_docs/api_generator dep,
        "//tensorflow_recommenders",
    ],
)
