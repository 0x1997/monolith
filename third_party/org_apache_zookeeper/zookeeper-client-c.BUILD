load("@rules_cc//cc:defs.bzl", "cc_library")


genrule(
    name = "config_h",
    srcs = [
        "@//third_party/org_apache_zookeeper:genfiles",
    ] + glob([
        "src/**",
    ]),
    outs = [
        "config.h",
        "include/zookeeper.jute.c",
        "include/zookeeper.jute.h",
    ],
    cmd = "cp third_party/org_apache_zookeeper/configure external/org_apache_zookeeper/configure " +
          "&& cp third_party/org_apache_zookeeper/install-sh external/org_apache_zookeeper/install-sh " +
          "&& cp third_party/org_apache_zookeeper/missing external/org_apache_zookeeper/missing " +
          "&& cp third_party/org_apache_zookeeper/config.guess external/org_apache_zookeeper/config.guess " +
          "&& cp third_party/org_apache_zookeeper/config.sub external/org_apache_zookeeper/config.sub " +
          "&& cp third_party/org_apache_zookeeper/Makefile.in external/org_apache_zookeeper/Makefile.in " +
          "&& cp third_party/org_apache_zookeeper/config.h.in external/org_apache_zookeeper/config.h.in " +
          "&& cp third_party/org_apache_zookeeper/ltmain.sh external/org_apache_zookeeper/ltmain.sh " +
          "&& cp -r third_party/org_apache_zookeeper/generated external/org_apache_zookeeper/generated " +
          "&& pushd external/org_apache_zookeeper " +
          "&& ./configure --without-cppunit " +
          "&& popd " +
          "&& mv external/org_apache_zookeeper/config.h $(location config.h)" +
          "&& cp external/org_apache_zookeeper/generated/zookeeper.jute.c $(location include/zookeeper.jute.c)" +
          "&& cp external/org_apache_zookeeper/generated/zookeeper.jute.h $(location include/zookeeper.jute.h)",
)

cc_library(
    name = "config",
    hdrs = [":config.h"],
)

cc_library(
    name = "libzookeeper_mt",
    srcs = [
        "include/zookeeper.jute.c",
        "src/addrvec.c",
        "src/mt_adaptor.c",
        "src/recordio.c",
        "src/zk_hashtable.c",
        "src/zk_log.c",
        "src/zookeeper.c",
        "src/hashtable/hashtable.c",
        "src/hashtable/hashtable_itr.c",
        "src/addrvec.h",
        "src/zk_adaptor.h",
        "src/zk_hashtable.h",
        "src/hashtable/hashtable.h",
        "src/hashtable/hashtable_itr.h",
        "src/hashtable/hashtable_private.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "src/winport.h",
            "src/winport.c",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "include/zookeeper.jute.h",
        "include/proto.h",
        "include/recordio.h",
        "include/zookeeper.h",
        "include/zookeeper_log.h",
        "include/zookeeper_version.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "include/winconfig.h",
        ],
        "//conditions:default": [],
    }),
    defines = [
        "THREADED",
        "USE_STATIC_LIB",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "_WINDOWS",
            "WIN32",
        ],
        "//conditions:default": [],
    }),
    includes = [
        ".",
        # "generated",
        "include",
        "src",
        "src/hashtable",
    ],
    linkopts = select({
        "@bazel_tools//src/conditions:windows": ["-DEFAULTLIB:ws2_32.lib"],
        "//conditions:default": ["-lpthread"],
    }),
    linkstatic = True,
    visibility = ["//visibility:public"],
    data = [
        "@//third_party/org_apache_zookeeper:genfiles",
    ],
    deps = select({
        "@bazel_tools//src/conditions:windows": [
            "@//third_party/zookeeper-client-c:config",
        ],
        "//conditions:default": [
            ":config",
        ],
    }),
)
