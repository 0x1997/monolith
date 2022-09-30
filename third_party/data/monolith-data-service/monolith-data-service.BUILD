load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test", "cc_binary")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name="fountain_client",
    srcs=[
        "fountain_client.cpp",
    ],
    deps=[
        ":fountain_io",
    ],
    copts = ["-Wno-error=maybe-uninitialized"],
)

cc_library(
    name="fountain_io",
    srcs=[
        "zookeeper_helper.cpp",
        "fountain_io.cpp",
    ],
    hdrs=[
        "zookeeper_helper.h",
        "fountain_io.h",
        "common.h",
        "instance_parser.pb.h",
    ],
    includes = [
        "external/org_apache_zookeeper/include",
    ],
    deps=[
        ":fountain_service",
        "@org_apache_zookeeper//:libzookeeper_mt",
        "@com_google_glog//:glog",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

proto_library(
    name="fountain_service_proto",
    srcs=["fountain_service.proto"],
    deps=[
        ":instance_parser_proto",
        "@//euclid/fountain/proto:fountain_proto",
    ],
)

cc_proto_library(
    name="fountain_service_cc_proto",
    srcs = [
        "fountain_service.proto",
    ],
    deps=[
        ":instance_parser_cc_proto",
        "@//euclid/fountain/proto:fountain_cc_proto",
    ],
)

cc_grpc_library(
    name = "fountain_service",
    srcs = [
        ":fountain_service_proto",
    ],
    grpc_only = True,
    deps = [
        ":fountain_service_cc_proto",
    ],
)

proto_library(
    name="instance_parser_proto",
    srcs=["instance_parser.proto"],
)

cc_proto_library(
    name="instance_parser_cc_proto",
    srcs=["instance_parser.proto"],
)

py_proto_library(
    name = "instance_parser_py_proto",
    srcs = ["instance_parser.proto"],
)
