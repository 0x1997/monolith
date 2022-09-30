load("@rules_foreign_cc//tools/build_defs:configure.bzl", "configure_make")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "src",
    srcs = glob(["**"]),
)

configure_make(
    name = "libtcmalloc",
    binaries = ["pprof"],
    lib_source = ":src",
    shared_libraries = ["libtcmalloc.so"],
    static_libraries = ["libtcmalloc.a"],
)
