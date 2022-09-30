load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sparsehash",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["include/sparsehash/*", "include/sparsehash/internal/*"]),
    includes = ["include"],
    deps = [],
)
