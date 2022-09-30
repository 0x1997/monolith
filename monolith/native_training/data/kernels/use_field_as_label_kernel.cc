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

class UseFieldAsLabelOp : public OpKernel {
 public:
  explicit UseFieldAsLabelOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("field_name", &field_name_));
    OP_REQUIRES_OK(context, context->GetAttr("overwrite_invalid_value",
                                             &overwrite_invalid_value_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("label_threshold", &label_threshold_));
    OP_REQUIRES_OK(context, context->GetAttr("variant_type", &variant_type_));

    if (variant_type_ != "instance" && variant_type_ != "example") {
      LOG(FATAL) << "Invalid 'variant_type', please choose on from "
                    "['instance', 'example']!";
    }
  }

  // Reference:
  void Compute(OpKernelContext *context) override {
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
    auto labels = GetLabel(output_tensor, is_instance);

    float field_value = 0;
    if (!GetNewLabel(*output_tensor, is_instance, &field_value)) {
      LOG(FATAL) << "Cannot get label from !" << field_name_;
      return;
    }
    for (int i = 0; i < labels->size(); ++i) {
      const auto original_label = labels->Get(i);
      float new_label = field_value;
      if (overwrite_invalid_value_) {
        if (new_label >= label_threshold_) {
          new_label = 0;
        }
      }
      new_label = std::max(original_label, new_label);
      labels->Set(i, new_label);
    }
  }

 private:
  bool GetNewLabel(const Tensor &output_tensor, bool is_instance,
                   float *new_label) {
    auto line_id = GetLineId(output_tensor, is_instance);
    const google::protobuf::Descriptor *descriptor = LineId::GetDescriptor();
    const google::protobuf::Reflection *reflection = LineId::GetReflection();
    const google::protobuf::FieldDescriptor *field =
        descriptor->FindFieldByName(field_name_);
    if (field == nullptr || field->is_repeated()) {
      return false;
    }
    switch (field->cpp_type()) {
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_FLOAT: {
        *new_label = reflection->GetFloat(line_id, field);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_DOUBLE: {
        *new_label = reflection->GetDouble(line_id, field);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT32: {
        *new_label = reflection->GetInt32(line_id, field);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_INT64: {
        *new_label = reflection->GetInt64(line_id, field);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT32: {
        *new_label = reflection->GetUInt32(line_id, field);
        break;
      }
      case google::protobuf::FieldDescriptor::CppType::CPPTYPE_UINT64: {
        *new_label = reflection->GetUInt64(line_id, field);
        break;
      }
      default:
        LOG(INFO) << "dtype is " << field->cpp_type();
        return false;
    }
    return true;
  }
  static const LineId &GetLineId(const Tensor &output_tensor,
                                 bool is_instance) {
    if (is_instance) {
      return output_tensor.scalar<Variant>()().get<Instance>()->line_id();
    } else {
      return output_tensor.scalar<Variant>()().get<Example>()->line_id();
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
  std::string field_name_;
  bool overwrite_invalid_value_;
  float label_threshold_;
  std::string variant_type_;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("UseFieldAsLabel").Device(DEVICE_CPU),
                        UseFieldAsLabelOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
