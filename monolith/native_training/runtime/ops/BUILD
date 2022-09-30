load("@rules_cc//cc:defs.bzl", "cc_library", "cc_proto_library", "cc_test")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "cc_header_only_library", "tf_cc_test", "tf_custom_op_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//monolith:__subpackages__"])

cc_header_only_library(
    name = "traceme",
    deps = [
        "@org_tensorflow//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "tracelib",
    deps = [
        ":traceme",
        "//monolith/native_training/runtime/common:metrics",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "embedding_hash_table_tf_bridge",
    srcs = ["embedding_hash_table_tf_bridge.cc"],
    hdrs = ["embedding_hash_table_tf_bridge.h"],
    deps = [
        ":neo_filter_tf_bridge",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
        "//monolith/native_training/runtime/hopscotch:hopscotch_hash_set",
        "//monolith/native_training/runtime/neo_filter:filter",
        "//monolith/native_training/runtime/neo_filter:probabilistic_filter",
        "//monolith/native_training/runtime/neo_filter:sliding_hash_filter",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "neo_filter_tf_bridge",
    srcs = ["neo_filter_tf_bridge.cc"],
    hdrs = ["neo_filter_tf_bridge.h"],
    deps = [
        ":file_utils",
        "//monolith/native_training/data/training_instance:reader_util",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_cc_proto",
        "//monolith/native_training/runtime/neo_filter:filter",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "touched_key_set_tf_bridge",
    srcs = [],
    hdrs = ["touched_key_set_tf_bridge.h"],
    deps = [
        "//monolith/native_training/runtime/hopscotch:hopscotch_hash_set",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "parameter_sync_tf_bridge",
    srcs = ["parameter_sync_tf_bridge.cc"],
    hdrs = ["parameter_sync_tf_bridge.h"],
    deps = [
        ":embedding_hash_table_tf_bridge",
        ":multi_hash_table",
        "//monolith/native_training/runtime/parameter_sync:dummy_sync_client",
        "//monolith/native_training/runtime/parameter_sync:dummy_sync_server",
        "//monolith/native_training/runtime/parameter_sync:sync_client_manager",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "file_utils",
    srcs = ["file_utils.cc"],
    hdrs = ["file_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf_lite",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

tf_cc_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":file_utils",
        "//monolith/native_training/data/training_instance:reader_util",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

tf_custom_op_library(
    name = "clip_ops.so",
    srcs = [
        "clip_by_global_norm.h",
        "clip_by_global_norm_op.cc",
    ],
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    gpu_srcs = [
        "clip_by_global_norm.h",
        "clip_by_global_norm.cu.cc",
    ],
    linkopts = [],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array_for_custom_op",
    ],
)

cc_library(
    name = "multi_hash_table",
    hdrs = ["multi_hash_table.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "hash_table_ops",
    srcs = [
        "hash_table/misc_ops.cc",
        "hash_table_lookup_op.cc",
        "hash_table_op.cc",
        "hash_table_restore_op.cc",
        "hash_table_save_op.cc",
        "hash_table_update_op.cc",
        "multi_hash_table.h",
        "multi_hash_table_lookup_op.cc",
        "multi_hash_table_op.cc",
        "multi_hash_table_save_restore_ops.cc",
        "multi_hash_table_update_op.cc",
    ],
    deps = [
        ":embedding_hash_table_tf_bridge",
        ":file_utils",
        ":multi_hash_table",
        ":neo_filter_tf_bridge",
        ":parameter_sync_tf_bridge",
        "//monolith/native_training/data/training_instance:reader_util",
        "//monolith/native_training/runtime/concurrency:queue",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "distribution_ops",
    srcs = [
        "fused_embedding_to_layout.cc",
        "fused_reorder_by_indices.cc",
        "map_id_to_embedding_op.cc",
        "reduce_op.cc",
        "split_by_indices_op.cc",
        "static_reshape_op.cc",
        "unique_mapping_ops.cc",
    ],
    deps = [
        "//idl:example_cc_proto",
        "//monolith/native_training/data/training_instance:data_reader",
        "//monolith/native_training/data/training_instance:parse_instance_lib",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
        "//monolith/native_training/runtime/ops:traceme",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "distribution_ops.so",
    copts = [
        "-D_ENABLE_AVX",
        "-mavx",
        "-mfma",
    ],
    gpu_srcs = [
        "map_id_to_embedding.cu.cc",
        "reduce_op.cu.cc",
    ],
    deps = [
        ":distribution_ops",
        # TODO: Figure out how to link "@org_tensorflow//tensorflow/core/kernels:cwise_lib_hdrs" for fill_functor.h
        "@org_tensorflow//tensorflow/core/kernels:gpu_device_array_for_custom_op",
        "//monolith/native_training/data/training_instance:parse_instance_lib",
        "//monolith/native_training/runtime/hash_table:embedding_hash_table_factory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "//idl:example_cc_proto",
        "//monolith/native_training/data/training_instance:data_reader",
        "//monolith/native_training/runtime/ops:traceme",
    ],
)

tf_custom_op_library(
    name = "hash_filter_ops.so",
    srcs = [
        "hash_filter_intercept_gradient_op.cc",
        "hash_filter_op.cc",
    ],
    deps = [
        ":file_utils",
        ":neo_filter_tf_bridge",
        "//monolith/native_training/runtime/neo_filter:dummy_hash_filter",
        "//monolith/native_training/runtime/neo_filter:hash_filter",
        "//monolith/native_training/runtime/neo_filter:probabilistic_filter",
        "//monolith/native_training/runtime/neo_filter:sliding_hash_filter",
    ],
)

tf_custom_op_library(
    name = "hash_filter_save_op.so",
    srcs = [
        "hash_filter_save_op.cc",
    ],
    deps = [
        ":file_utils",
        ":neo_filter_tf_bridge",
        "//monolith/native_training/runtime/neo_filter:dummy_hash_filter",
        "//monolith/native_training/runtime/neo_filter:hash_filter",
        "//monolith/native_training/runtime/neo_filter:sliding_hash_filter",
    ],
)

tf_custom_op_library(
    name = "file_ops.so",
    srcs = [
        "file_ops.cc",
    ],
)

tf_custom_op_library(
    name = "hash_filter_restore_op.so",
    srcs = [
        "hash_filter_restore_op.cc",
    ],
    deps = [
        ":file_utils",
        ":neo_filter_tf_bridge",
        "//monolith/native_training/runtime/neo_filter:dummy_hash_filter",
        "//monolith/native_training/runtime/neo_filter:hash_filter",
        "//monolith/native_training/runtime/neo_filter:sliding_hash_filter",
    ],
)

tf_custom_op_library(
    name = "touched_key_set_ops.so",
    srcs = [
        "touched_key_set_insert_op.cc",
        "touched_key_set_op.cc",
        "touched_key_set_steal_op.cc",
    ],
    deps = [
        ":touched_key_set_tf_bridge",
    ],
)

tf_custom_op_library(
    name = "gen_seq_mask.so",
    srcs = [
        "gen_seq_mask.cc",
    ],
)

cc_library(
    name = "inbatch_auc_loss",
    srcs = [
        "inbatch_auc_loss.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "inbatch_auc_loss.so",
    deps = [
        ":inbatch_auc_loss",
    ],
)

cc_library(
    name = "distributed_serving_ops",
    srcs = [
        "parameter_sync_ops.cc",
        "remote_predict_op_grpc.cc",
    ],
    hdrs = [
        "remote_predict_op.h",
    ],
    deps = [
        ":agent_heartbeat",
        ":deep_insight_ops",
        ":parameter_sync_tf_bridge",
        ":prediction_service_grpc",
        ":tracelib",
        "//monolith/native_training/runtime/common:metrics",
        "@com_github_grpc_grpc//:grpc++_reflection",
    ],
)

proto_library(
    name = "logging_ops_proto",
    srcs = ["logging_ops.proto"],
)

cc_proto_library(
    name = "logging_ops_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":logging_ops_proto"],
)

py_proto_library(
    name = "logging_ops_py_proto",
    srcs = ["logging_ops.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "logging_ops",
    srcs = [
        "logging_ops.cc",
    ],
    deps = [
        ":logging_ops_cc_proto",
        "//monolith/native_training/runtime/common:metrics",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "deep_insight_client_tf_bridge",
    hdrs = ["deep_insight_client_tf_bridge.h"],
    deps = [
        "//monolith/native_training/runtime/deep_insight",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core/kernels:ops_util_hdrs",
    ],
)

cc_library(
    name = "deep_insight_ops",
    srcs = [
        "deep_insight_ops.cc",
    ],
    deps = [
        ":deep_insight_client_tf_bridge",
    ],
)

cc_library(
    name = "prediction_service_grpc",
    srcs = [
        "prediction_service_grpc.cc",
    ],
    hdrs = [
        "prediction_service_grpc.h",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

cc_library(
    name = "prediction_service_archon",
    srcs = [
        "prediction_service_archon.cc",
    ],
    hdrs = [
        "prediction_service_archon.h",
    ],
    deps = [
        "//idl:serving_idl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@data_archon_master//:archon",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

cc_library(
    name = "agent_heartbeat",
    srcs = [
        "agent_heartbeat.cc",
    ],
    hdrs = [
        "agent_heartbeat.h",
    ],
    deps = [
        ":net_utils",
        "//monolith/agent_service:agent_service_cc_proto_grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/core/platform:logging",
        "@org_tensorflow_serving//tensorflow_serving/apis:prediction_service_proto",
    ],
)

tf_cc_test(
    name = "agent_heartbeat_test",
    srcs = ["agent_heartbeat_test.cc"],
    extra_copts = [
        "-DTEST_USE_GRPC",
    ],
    deps = [
        ":agent_heartbeat",
        ":prediction_service_grpc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "agent_heartbeat_test_archon_lib",
    srcs = ["agent_heartbeat_test.cc"],
    copts = [
        "-DTEST_USE_ARCHON",
    ],
    deps = [
        ":agent_heartbeat",
        ":prediction_service_archon",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "net_utils",
    srcs = ["net_utils.cc"],
    hdrs = ["net_utils.h"],
)

cc_test(
    name = "net_utils_test",
    srcs = ["net_utils_test.cc"],
    deps = [
        ":net_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

# Expose monolith ops for tf serving
cc_library(
    name = "serving_ops_cc",
    srcs = [
        # hash filter ops
        "hash_filter_intercept_gradient_op.cc",
        "hash_filter_op.cc",
        # instace parsing ops
        "//monolith/native_training/data/training_instance:cc/parse_instance_kernel.cc",
        "//monolith/native_training/data/training_instance:cc/parse_instance_ops.cc",
        # realtime sync
        "parameter_sync_ops.cc",
        # remote predict ops using archon
        # "remote_predict_op_archon.cc",
        # file_ops
        "file_ops.cc",
        # new data ops and kernels
        "//monolith/native_training/data:ops/feature_utils_ops.cc",
        "//monolith/native_training/data:ops/parse_input_data_ops.cc",
        "//monolith/native_training/data:ops/pb_dataset_ops.cc",
        # gen seq mask
        "gen_seq_mask.cc",
        # nas
        "//monolith/native_training/layers:kernels/nas_kernels.cc",
        "//monolith/native_training/layers:ops/nas_ops.cc",
        # ffm
        "//monolith/native_training/layers:kernels/ffm_kernels.cc",
        "//monolith/native_training/layers:ops/ffm_ops.cc",
    ] + select({
        "//:grpc_remote_call": [":remote_predict_op_grpc.cc"],
        "//conditions:default": ["remote_predict_op_archon.cc"],
    }),
    hdrs = [
        "remote_predict_op.h",
        "//monolith/native_training/layers:kernels/ffm_kernels.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = select({
        "//:grpc_remote_call": [":prediction_service_grpc"],
        "//conditions:default": [":prediction_service_archon"],
    }) + [
        # archon should be before grpc(:agent_heartbeat), to let folly use openssl .h
        # ":prediction_service_archon",
        ":agent_heartbeat",
        ":embedding_hash_table_tf_bridge",
        ":file_utils",
        ":hash_table_ops",
        ":logging_ops",
        ":neo_filter_tf_bridge",
        ":distribution_ops",
        ":parameter_sync_tf_bridge",
        ":inbatch_auc_loss",
        "//idl:compression_cc_float16",
        "//monolith/agent_service:agent_service_cc_proto_grpc",
        "//monolith/native_training/data/training_instance:pb_datasource_lib",
        "//monolith/native_training:monolith_custom_ops.so",
        "//monolith/native_training/data:pb_data_lib",
        "//monolith/native_training/runtime/common:metrics",
        "//monolith/native_training/runtime/concurrency:queue",
        "//monolith/native_training/runtime/neo_filter:dummy_hash_filter",
        "//monolith/native_training/runtime/neo_filter:sliding_hash_filter",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow_serving//tensorflow_serving/apis:model_cc_proto",
        "@org_tensorflow_serving//tensorflow_serving/apis:predict_cc_proto",
        ":tracelib",
    ],
    alwayslink = 1,
)
