load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//monolith/native_training/alert:__subpackages__"])

py_library(
    name = "messager",
    srcs = ["messager.py"],
    deps = [
        "//monolith/monitoring",
    ],
)

py_library(
    name = "alert_rules",
    srcs = ["alert_rules.py"],
    deps = [
        ":messager",
        "//monolith/monitoring",
        "//monolith/native_training/alert:alert_py_proto",
    ],
)

py_test(
    name = "alert_rules_test",
    srcs = ["alert_rules_test.py"],
    deps = [
        ":alert_rules",
    ],
)

py_binary(
    name = "test_main",
    srcs = ["test_main.py"],
    deps = [
        "//monolith/native_training/alert",
    ],
)
