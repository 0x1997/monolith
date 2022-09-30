load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = [
    "//monolith/agent_service:__subpackages__",
    "//monolith/integration_test:__subpackages__",
    "//monolith/native_training:__subpackages__",
])

py_library(
    name = "model_loader",
    srcs = ["model_loader.py"],
    srcs_version = "PY3",
    deps = [
        "//monolith/agent_service:data_def",
        "//monolith/agent_service:dispatcher",
        "//monolith/agent_service:resource_utils",
        "//monolith/agent_service:utils",
        "//monolith/agent_service:zk_mirror",
    ],
)

py_library(
    name = "resource_report",
    srcs = ["resource_report.py"],
    srcs_version = "PY3",
    deps = [
        "//monolith/agent_service:data_def",
        "//monolith/agent_service:dispatcher",
        "//monolith/agent_service:resource_utils",
        "//monolith/agent_service:utils",
        "//monolith/agent_service:zk_mirror",
    ],
)

py_library(
    name = "status_report",
    srcs = ["status_report.py"],
    srcs_version = "PY3",
    deps = [
        "//monolith/agent_service:data_def",
        "//monolith/agent_service:dispatcher",
        "//monolith/agent_service:resource_utils",
        "//monolith/agent_service:utils",
        "//monolith/agent_service:zk_mirror",
    ],
)

py_library(
    name = "scheduler",
    srcs = ["scheduler.py"],
    srcs_version = "PY3",
    deps = [
        "//monolith/agent_service:agent_service_py_pb2",
        "//monolith/agent_service:agent_service_py_pb2_grpc",
        "//monolith/agent_service:data_def",
        "//monolith/agent_service:dispatcher",
        "//monolith/agent_service:resource_utils",
        "//monolith/agent_service:utils",
        "//monolith/agent_service:zk_mirror",
        requirement("grpcio"),
    ],
)

py_test(
    name = "scheduler_test",
    srcs = ["scheduler_test.py"],
    deps = [
        ":scheduler",
    ],
)

py_test(
    name = "handlers_test",
    srcs = ["handlers_test.py"],
    srcs_version = "PY3",
    deps = [
        ":model_loader",
        ":resource_report",
        ":scheduler",
        ":status_report",
        "//monolith/agent_service:agent_service_py_pb2",
        "//monolith/agent_service:mocked_tfserving",
        "//monolith/agent_service:mocked_zkclient",
        "//monolith/agent_service:tfs_monitor",
    ],
)
