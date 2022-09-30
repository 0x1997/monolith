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

#include "idl/matrix/proto/example.pb.h"
#include "idl/matrix/proto/proto_parser.pb.h"
#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/coding.h"

namespace tensorflow {
namespace monolith_tf {

using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using Instance = ::parser::proto::Instance;

class TStringReader : public BaseStreamReader {
 public:
  explicit TStringReader(const tstring& data, const bool& has_header,
                         const ReaderOptions& options)
      : BaseStreamReader(options),
        cur_(0),
        buf_(data),
        has_header_(has_header) {}

  virtual ~TStringReader() = default;

  uint64 GetOffset() override { return static_cast<uint64>(cur_); }

  Status SetOffset(uint64* offset) override {
    cur_ = *offset;
    return Status::OK();
  }

  Status ReadPBBytes(uint8_t* pb_type, uint32_t* data_source_key,
                     tstring* instance) {
    size_t size;
    if (has_header_) {
      TF_RETURN_IF_ERROR(ReadDataHeader(pb_type, data_source_key));
      ReadBinarySize(&size);
      CHECK(buf_.size() - cur_ == size);
    } else {
      *pb_type = 0;
      *data_source_key = 0;
      size = buf_.size() - cur_;
    }

    if (size > 0) {
      instance->assign(buf_.data() + cur_, size);
    }
    return Status::OK();
  }

  const char* GetData(size_t* size, uint8_t* pb_type,
                      uint32_t* data_source_key) {
    if (has_header_) {
      ReadDataHeader(pb_type, data_source_key);
      ReadBinarySize(size);
      if (buf_.size() - cur_ != *size) {
        LOG(WARNING) << "Data Error: " << buf_;
        *size = 0;
        return nullptr;
      }
    } else {
      *pb_type = 0;
      *data_source_key = 0;
      *size = buf_.size() - cur_;
    }

    if (*size > 0) {
      return buf_.data() + cur_;
    } else {
      return nullptr;
    }
  }

 private:
  size_t cur_;
  const tstring& buf_;
  bool has_header_;

 protected:
  Status ReadBinarySize(size_t* size) override {
    if (cur_ + 8 >= buf_.size() && cur_ < buf_.size()) {
      *size = 0;
      cur_ = buf_.size();
      return errors::DataLoss("truncated record");
    } else if (cur_ + 8 < buf_.size()) {
      *size = static_cast<size_t>(
          ::tensorflow::core::DecodeFixed64(buf_.data() + cur_));
      cur_ += 8;
      return Status::OK();
    } else {
      *size = 0;
      cur_ = buf_.size();
      return errors::OutOfRange("eof");
    }
  }

  Status ReadNBytes(size_t n, tstring* result) override {
    if (n > 0 && cur_ + n >= buf_.size() && cur_ < buf_.size()) {
      cur_ = buf_.size();
      return errors::DataLoss("truncated record");
    } else if (cur_ + n < buf_.size()) {
      result->assign(buf_.data() + cur_, n);
      cur_ += n;
      return Status::OK();
    } else {
      cur_ = buf_.size();
      return errors::OutOfRange("eof");
    }
  }
};

class StringToVariantOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  using ConstFlatSplits = typename TTypes<int64>::ConstFlat;

  explicit StringToVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_type", &variant_type_));

    std::unordered_set<std::string> variant_type_set_ = {
        "instance", "example", "examplebatch", "example_batch"};
    OP_REQUIRES(
        ctx, variant_type_set_.count(variant_type_) != 0,
        errors::InvalidArgument("variant_type can only be instance, example "
                                "and examplebatch/example_batch"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("has_header", &has_header_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("has_sort_id", &options_.has_sort_id));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("lagrangex_header", &options_.lagrangex_header));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("kafka_dump_prefix", &options_.kafka_dump_prefix));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("kafka_dump", &options_.kafka_dump));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tstring>();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<Variant>();

    size_t size;
    uint8_t pb_type;
    uint32_t data_source_key;
    for (size_t i = 0; i < input.size(); ++i) {
      const tstring& buf = input(i);
      TStringReader reader(buf, has_header_, options_);
      const char* data = reader.GetData(&size, &pb_type, &data_source_key);
      if (variant_type_ == "instance") {
        Instance pb;
        if (size > 0) {
          CHECK(pb.ParseFromArray(data, size));
          pb.set_data_source_key(data_source_key);
        }
        output_flat(i) = std::move(pb);
      } else if (variant_type_ == "example") {
        Example pb;
        if (size > 0) {
          CHECK(pb.ParseFromArray(data, size));
          pb.set_data_source_key(data_source_key);
        }
        output_flat(i) = std::move(pb);
      } else {
        ExampleBatch pb;
        if (size > 0) {
          CHECK(pb.ParseFromArray(data, size));
          pb.set_data_source_key(data_source_key);
        }
        output_flat(i) = std::move(pb);
      }
    }
  }

 private:
  std::string variant_type_;
  bool has_header_ = false;
  ReaderOptions options_;
};

class VariantToZerosOp : public OpKernel {
 public:
  explicit VariantToZerosOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int64>();
    output_flat.setZero();
  }
};


class HasVariantOp : public OpKernel {
 public:
  explicit HasVariantOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("variant_type", &variant_type_));

    std::unordered_set<std::string> variant_type_set_ = {
        "instance", "example", "examplebatch", "example_batch"};
    OP_REQUIRES(
        ctx, variant_type_set_.count(variant_type_) != 0,
        errors::InvalidArgument("variant_type can only be instance, example "
                                "and examplebatch/example_batch"));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_scalar = output_tensor->scalar<bool>();

    int byte_size = 0;
    if (variant_type_ == "instance") {
      const auto *instance = input_tensor.scalar<Variant>()().get<Instance>();
      byte_size = instance->ByteSize();
    } else if (variant_type_ == "example") {
      const auto *example = input_tensor.scalar<Variant>()().get<Example>();
      byte_size = example->ByteSize();
    } else {
      const auto *example_batch = input_tensor.scalar<Variant>()().get<ExampleBatch>();
      byte_size = example_batch->ByteSize();
    }

    output_scalar() = byte_size > 0;
  }

 private:
  std::string variant_type_;
};


namespace {
REGISTER_KERNEL_BUILDER(Name("StringToVariant").Device(DEVICE_CPU),
                        StringToVariantOp);
REGISTER_KERNEL_BUILDER(Name("VariantToZeros").Device(DEVICE_CPU),
                        VariantToZerosOp);
REGISTER_KERNEL_BUILDER(Name("HasVariant").Device(DEVICE_CPU),
                        HasVariantOp);
}  // namespace

}  // namespace monolith_tf
}  // namespace tensorflow
