load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_custom_op_library")

package(
    default_visibility = ["//visibility:public"],
)

tf_custom_op_library(
    name = "training_ops.so",
    srcs = [
        "cc/training_op_helpers.h",
        "cc/training_ops.cc",
    ],
    deps = [],
)

py_library(
    name = "adamom",
    srcs = ["adamom.py"],
    data = [
        ":training_ops.so",
    ],
    deps = ["//monolith:utils",
            "@org_tensorflow//tensorflow:tensorflow_py"]
)

py_test(
    name = "adamom_test",
    srcs = ["adamom_test.py"],
    deps = [
        ":adamom",
    ],
)

py_library(
    name = "shampoo",
    srcs = ["shampoo.py"],
    deps = ["//monolith:utils",
            "@org_tensorflow//tensorflow:tensorflow_py"]
)

py_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    data = [
        ":training_ops.so",
    ],
    deps = ["//monolith:utils",
            "@org_tensorflow//tensorflow:tensorflow_py"]
)

py_test(
    name = "rmsprop_test",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":rmsprop",
    ],
)

py_test(
    name = "rmspropv2_test",
    srcs = ["rmspropv2_test.py"],
    deps = [
        ":rmsprop",
    ],
)

py_test(
    name = "rmsprop_dense_v2_test",
    srcs = ["rmsprop_dense_v2_test.py"],
    deps = [
        ":rmsprop",
    ],
)

py_test(
    name = "rmsprop_dense_v1_test",
    srcs = ["rmsprop_dense_v1_test.py"],
    deps = [
        ":rmsprop",
    ],
)
