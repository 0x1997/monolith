load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "databus_proto",
    srcs = ["collector.proto"],
    import_prefix = "cpputil/databusclient",
)

cc_proto_library(
    name = "databus_cc_proto",
    deps = [
        ":databus_proto",
    ],
)

cc_library(
    name = "databus",
    srcs = [
        "channel_impl.cc",
        "client.cc",
        "databus_cpp_cache.cc",
    ],
    hdrs = glob(["include/*"]),
    copts = ["-Wno-pointer-arith"],
    # Remote bazel build has some problem with handling the include_prefix. Removing it to mitigate it.
    #include_prefix = "cpputil/databusclient",
    includes = ["include"],
    deps = [
        ":databus_cc_proto",
        "@boost//:thread",
        "@cpputil_metrics2//:metrics2",
    ],
)

cc_test(
    name = "databus_test",
    srcs = [
        "test/client_test.cc",
        "test/databus_cpp_cache_test.cc",
    ],
    deps = [
        ":databus",
        "@com_google_googletest//:gtest_main",
    ],
)
