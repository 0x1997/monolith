load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "metrics2",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    include_prefix = "cpputil/metrics2",
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@msgpack",
    ],
)
