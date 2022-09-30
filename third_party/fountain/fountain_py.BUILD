load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "fountain_lib",
    srcs = [
        "python/fountain.py",
        "python/byted_euclid_fountain/__init__.py",
        "python/byted_euclid_fountain/data_flow_base.py",
        "python/byted_euclid_fountain/data_flow.py",
        "python/byted_euclid_fountain/fountain.py",
        "python/byted_euclid_fountain/fountain_pb2.py",
        "python/byted_euclid_fountain/proto_reader.py",
        "python/byted_euclid_fountain/utils.py"
    ],
    imports = [
        "python",
        "python/byted_euclid_fountain",
    ],
)
