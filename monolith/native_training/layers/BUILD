load("@rules_python//python:defs.bzl", "py_library", "py_binary", "py_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test", "tf_custom_op_library")

package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name="utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//monolith:utils",
        "//monolith/native_training:monolith_export",
        "//monolith/native_training:utils",
    ],
)

py_library(
    name="add_bias",
    srcs = ["add_bias.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name="add_bias_test",
    srcs = ["add_bias_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":add_bias",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="dense",
    srcs = ["dense.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name="dense_test",
    srcs = ["dense_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":dense",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="advanced_activations",
    srcs = ["advanced_activations.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name="advanced_activations_test",
    srcs = ["advanced_activations_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":advanced_activations",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="agru",
    srcs = ["agru.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name="agru_test",
    srcs = ["agru_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":agru",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="norms",
    srcs = [
        "norms.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name="norms_test",
    srcs = ["norms_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":norms",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="mlp",
    srcs = [
        "mlp.py"
        ],
    srcs_version = "PY3",
    deps = [
        ":norms",
        ":dense",
        ":advanced_activations",
        ":utils",
    ],
)

py_test(
    name="mlp_test",
    srcs = ["mlp_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":mlp",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="feature_cross",
    srcs = [
        "feature_cross.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":agru",
        "//monolith/core:base_layer",
        ":utils",
        ":layer_ops"
    ],
)

py_test(
    name="feature_cross_test",
    srcs = ["feature_cross_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":feature_cross",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="feature_trans",
    srcs = [
        "feature_trans.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":agru",
        "//monolith/core:base_layer",
        ":utils",
    ],
)

py_test(
    name="feature_trans_test",
    srcs = ["feature_trans_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":feature_trans",
        "//monolith/core:testing_utils",
    ],
)


py_library(
    name="feature_seq",
    srcs = [
        "feature_seq.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":agru",
        "//monolith/core:base_layer",
        ":utils",
    ],
)

py_test(
    name="feature_seq_test",
    srcs = ["feature_seq_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":feature_seq",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="pooling",
    srcs = [
        "pooling.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

py_test(
    name="pooling_test",
    srcs = ["pooling_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":pooling",
        "//monolith/core:testing_utils",
    ],
)


py_library(
    name="logit_correction",
    srcs = [
        "logit_correction.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":utils",
    ],
)

py_test(
    name="logit_correction_test",
    srcs = ["logit_correction_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":logit_correction",
        "//monolith/core:testing_utils",
    ],
)


py_library(
    name="lhuc",
    srcs = [
        "lhuc.py"
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":dense",
        ":advanced_activations",
        ":utils",
    ],
)

py_test(
    name="lhuc_test",
    srcs = ["lhuc_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":lhuc",
        "//monolith/core:testing_utils",
    ],
)

py_library(
    name="multi_task",
    srcs = [
        "multi_task.py"
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":dense",
        ":advanced_activations",
        ":utils",
    ],
)

py_test(
    name="multi_task_test",
    srcs = ["multi_task_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":multi_task",
        "//monolith/core:testing_utils",
    ],
)


py_library(
    name="layers",
    srcs = [
        "__init__.py",
        ],
    srcs_version = "PY3",
    deps = [
        ":mlp",
        ":norms",
        ":agru",
        ":dense",
        ":add_bias",
        ":advanced_activations",
        ":pooling",
        ":logit_correction",
        ":feature_cross",
        ":feature_trans",
        ":feature_seq",
        ":sparse_nas",
        ":lhuc",
        ":multi_task",
        "//monolith/native_training:utils",
    ],
)


tf_custom_op_library(
    name = "layer_tf_ops.so",
    srcs = [
        "ops/nas_ops.cc",
        "kernels/nas_kernels.cc",
        "ops/ffm_ops.cc",
        "kernels/ffm_kernels.h",
        "kernels/ffm_kernels.cc",
    ],
    gpu_srcs = [
        "kernels/ffm_kernels.h",
        "kernels/ffm_kernels.cu.cc",
    ],
    copts = ["-DNDEBUG"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array_for_custom_op",
    ],
)

py_library(
    name="layer_ops",
    srcs = ["layer_ops.py"],
    deps = [
        "//monolith:utils",
        "//monolith/core:testing_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
    data = [':layer_tf_ops.so'],
)


py_test(
    name="layer_ops_test",
    srcs = ["layer_ops_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":layer_ops",
    ],
)

py_library(
    name="sparse_nas",
    srcs = ["sparse_nas.py"],
    deps = [
        ":utils",
        ":layer_ops",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name="sparse_nas_test",
    srcs = ["sparse_nas_test.py"],
    srcs_version = "PY3",
    python_version = "PY3",
    deps = [
        ":sparse_nas",
    ],
)

exports_files([
    "kernels/ffm_kernels.cc",
    "kernels/ffm_kernels.cu.cc",
    "ops/ffm_ops.cc",
    "kernels/nas_kernels.cc",
    "ops/nas_ops.cc",
])
