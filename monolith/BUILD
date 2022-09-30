load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "base_runner",
    srcs = ["base_runner.py"],
)

py_binary(
    name = "gpu_runner",
    srcs = ["gpu_runner.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_runner",
        "//monolith/core:model_registry",
        "//monolith/tasks/image/efficientnet/params:efficientnet_imagenet",
        "//monolith/tasks/image/vit/params:vit_imagenet",
        "//monolith/tasks/tiktok/video_emb/params:diversify",
        "//monolith/tasks/tiktok/video_emb/params:imagenet",
        "//monolith/tasks/tiktok/video_emb/params:multitask",
        "//monolith/tasks/tiktok/video_emb/params:noisy",
        "//monolith/tasks/tiktok/video_emb/params:weakly",
    ],
)

py_binary(
    name = "tpu_runner",
    srcs = ["tpu_runner.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":base_runner",
        "//monolith/core:model_registry",
        "//monolith/tasks/cross_media/image/resnet:resnet_model",
        "//monolith/tasks/cross_media/tt_ads_ctr/params:cotrain",
        "//monolith/tasks/douyin/ads/cvr/params:ads_cvr",
        "//monolith/tasks/image/efficientnet/params:efficientnet_imagenet",
        "//monolith/tasks/image/vit/params:vit_imagenet",
        "//monolith/tasks/multimodal/clip/params:clip_tt",
        "//monolith/tasks/multimodal/vbert/params:vbert_tt",
        "//monolith/tasks/image/resnet/params:resnet_imagenet",
        "//monolith/tasks/tiktok/ads/ctr/params:ads_ctr",
        "//monolith/tasks/tiktok/ads/ctr_nas/params:ads_ctr",
        "//monolith/tasks/tiktok/ads/cvr/params:ads_cvr",
        "//monolith/tasks/tiktok/recommendation/stay_finish_skip/params:stay_finish_skip",
        "//monolith/tasks/tiktok/video_emb/params:diversify",
        "//monolith/tasks/tiktok/video_emb/params:imagenet",
        "//monolith/tasks/tiktok/video_emb/params:multitask",
        "//monolith/tasks/tiktok/video_emb/params:noisy",
        "//monolith/tasks/tiktok/video_emb/params:weakly",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
    ],
)

py_library(
    name = "init",
    srcs = ["__init__.py"],
    deps = [
        "//monolith/native_training/data:data",
        "//monolith/native_training/layers:layers",
        "//monolith/native_training/model_export:model_export",
        "//monolith/native_training:entry",
         "//monolith/native_training:native_model",
        "//monolith/native_training:estimator",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
