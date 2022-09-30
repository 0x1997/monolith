// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/hash/internal/city.h"
#include "absl/strings/str_format.h"
#include "monolith/native_training/data/kernels/internal/label_utils.h"
#include "monolith/native_training/data/training_instance/cc/instance_utils.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using Example = ::monolith::io::proto::Example;
using LineId = ::idl::matrix::proto::LineId;

class UseF100MultiHeadOp : public OpKernel {
 public:
  explicit UseF100MultiHeadOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));
    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }
  }

  void Compute(OpKernelContext *context) override {
    /* Parse data fields from input tensor. */
    const Tensor &input_tensor = context->input(0);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    bool is_instance = variant_type_ == "instance";
    if (is_instance) {
      Instance instance;
      instance.CopyFrom(*input_tensor.scalar<Variant>()().get<Instance>());
      output_tensor->scalar<Variant>()() = std::move(instance);
    } else {
      Example example;
      example.CopyFrom(*input_tensor.scalar<Variant>()().get<Example>());
      output_tensor->scalar<Variant>()() = std::move(example);
    }

    auto label = GetLabel(output_tensor, is_instance);
    std::vector<uint64_t> fids = GetFids(output_tensor, is_instance);

    /* use_f100_multihead() from matrix processor:
     */

    uint64_t channel_fid;
    uint64_t house_type_fid;
    for (const auto &fid : fids) {
      auto slot = fid >> 54;
      if (slot == 3) {
        channel_fid = fid;
      } else if (slot == 534) {
        house_type_fid = fid;
      }
    }
    std::set<uint64_t> related_channels = {67389660361559877, 60624132260595558,
                                           56437689166555693,
                                           56929448261580177};
    std::unordered_map<uint64_t, float> house_type_fids = {
        {9628416976777710561ULL, 1},
        {9622800525541736359ULL, 2},
        {9631589690660363017ULL, 3},
    };
    if (related_channels.find(channel_fid) != related_channels.end()) {
      label->Add(0);
    } else {
      label->Add(1);
    }

    // house_type
    auto iter = house_type_fids.find(house_type_fid);
    if (iter != house_type_fids.end()) {
      label->Add(iter->second);
    } else {
      label->Add(2);
    }
  }

 private:
  static LineId *GetLineId(Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_line_id();
    } else {
      return output_tensor->scalar<Variant>()()
          .get<Example>()
          ->mutable_line_id();
    }
  }

  static ::google::protobuf::RepeatedField<float> *GetLabel(
      Tensor *output_tensor, bool is_instance) {
    if (is_instance) {
      return output_tensor->scalar<Variant>()()
          .get<Instance>()
          ->mutable_label();
    } else {
      return output_tensor->scalar<Variant>()().get<Example>()->mutable_label();
    }
  }

  static std::vector<uint64_t> GetFids(Tensor *output_tensor,
                                       bool is_instance) {
    std::vector<uint64_t> fids;
    if (is_instance) {
      auto instance = output_tensor->scalar<Variant>()().get<Instance>();
      for (uint64_t fid : instance->fid()) {
        fids.push_back(fid);
      }
    } else {
      auto example = output_tensor->scalar<Variant>()().get<Example>();
      for (const auto &named_feature : example->named_feature()) {
        if (named_feature.feature().has_fid_v1_list()) {
          for (const auto &fid :
               named_feature.feature().fid_v1_list().value()) {
            fids.push_back(fid);
          }
        }
      }
    }
    return fids;
  }

  std::string variant_type_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("UseF100MultiHead").Device(DEVICE_CPU),
                        UseF100MultiHeadOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
