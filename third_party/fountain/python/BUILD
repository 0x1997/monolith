load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_visibility = ["//visibility:public"],
)

py_test(
    name = "fountain_lib_test",
    srcs = ["fountain_lib_test.py"],
    deps = [
        "@fountain_py//:fountain_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
