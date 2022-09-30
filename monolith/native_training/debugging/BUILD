load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_test")

package(default_visibility = ["//visibility:public"])

py_binary(
    name = "debugging_server",
    srcs = ["debugging_server.py"],
    data = [
        "//monolith/native_training:monolith_custom_ops.so",
        "//monolith/native_training/data/training_instance:pb_datasource_ops.so",
        "//monolith/native_training/optimizers:training_ops.so",
        "//monolith/native_training/runtime/ops:clip_ops.so",
        "//monolith/native_training/runtime/ops:hash_filter_ops.so",
        "//monolith/native_training/runtime/ops:hash_filter_restore_op.so",
        "//monolith/native_training/runtime/ops:hash_filter_save_op.so",
    ],
    deps = [
        "//monolith:utils",
        "//monolith/native_training:cluster_manager",
        "//monolith/native_training:env_utils",
        "//monolith/native_training:multi_type_hash_table",
        "//monolith/native_training:utils",
        "//monolith/native_training/proto:debugging_info_py_proto",
        requirement("flask"),
        requirement("flask_api"),
    ],
)

py_binary(
    name = "debugging_client",
    srcs = ["debugging_client.py"],
    deps = [
        ":debugging_server",
    ],
)
