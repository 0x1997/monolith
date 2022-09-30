load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])


py_proto_library(
    name = "monolith_model_py_proto",
    srcs = ["monolith_model.proto"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//monolith/native_training/runtime/hash_table/compressor:float_compressor_py_proto",
        "//monolith/native_training/runtime/hash_table/initializer:initializer_config_py_proto",
        "//monolith/native_training/runtime/hash_table/optimizer:optimizer_py_proto",
    ],
)

py_library(
    name = "graph_utils",
    srcs = ["graph_utils.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//idl:line_id_py_proto",
        "//monolith/native_training:utils",
    ]
)

py_library(
    name = "dump_utils",
    srcs = ["dump_utils.py"],
    deps = [
        ":monolith_model_py_proto",
        ":graph_utils",
        "//monolith/native_training/data:item_pool_hook",
        "//monolith/native_training/data:parsers_py",
    ]
)

py_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.py"],
    deps = [
        ":dump_utils",
        ":monolith_model_py_proto",
        ":graph_utils",
        "//monolith:utils",
        "//monolith/native_training/data:data",
        "//monolith/native_training:native_model",
    ],
    data = ["test_data/model_dump"]
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":dump_utils",
        ":monolith_model_py_proto",
        ":graph_utils",
        "//monolith:utils",
        "//monolith/native_training/data:data",
        "//monolith/native_training:native_model",
        "//monolith/native_training:runner_utils",
        "//monolith/native_training:estimator",
        "//monolith/core:base_embedding_host_call",
        "//monolith:init",
    ],
)

py_binary(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":model",
        ":dump_utils",
        ":monolith_model_py_proto",
        ":graph_utils",
        "//monolith:utils",
        "//monolith/native_training/data:data",
        "//monolith/native_training:native_model",
        "//monolith/native_training:runner_utils",
        "//monolith/native_training:estimator",
    ],
    data = [
        "test_data/model_dump",
        "feature_list.conf",
        "//monolith/native_training/data/training_instance:training_data",
    ]
)
