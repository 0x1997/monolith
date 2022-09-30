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

#ifndef MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_READER_H_
#define MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_READER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include "monolith/native_training/data/training_instance/cc/pb_variant.h"
#include "monolith/native_training/data/training_instance/cc/reader_util.h"

namespace tensorflow {

class RandomAccessFile;
namespace monolith_tf {

enum FeaturePruningType {
  AS_IS = 0,
  PRUNING_FEATURE = 1,
  PRUNING_RAW_FEATURE = 2
};

void ExtendExample(::monolith::io::proto::Example *pb,
                   FeatureNameMapper *mapper = nullptr);
Status ExampleToInstance(::monolith::io::proto::Example *example,
                         ::parser::proto::Instance *instance);
Status InstanceToExample(::parser::proto::Instance *instance,
                         ::monolith::io::proto::Example *example);

struct ReaderOptions {
  int64 buffer_size = 64 * 1024 * 1024;
  bool lagrangex_header = false;
  bool use_snappy = false;
  bool kafka_dump_prefix = false;
  bool has_sort_id = false;
  bool kafka_dump = false;
};

class BaseStreamReader {
 public:
  explicit BaseStreamReader(const ReaderOptions &options);
  virtual ~BaseStreamReader() = default;

  Status ReadPBBytes(uint8_t *pb_type, uint32_t *data_source_key,
                     tstring *instance);
  virtual uint64 GetOffset() = 0;
  virtual Status SetOffset(uint64 *offset) = 0;

 protected:
  Status ReadDataHeader(uint8_t *pb_type, uint32_t *data_source_key);
  virtual Status ReadNBytes(size_t n, tstring *result) = 0;
  virtual Status ReadBinarySize(size_t *size) = 0;

  ReaderOptions options_;
};

class StdinStreamReader : public BaseStreamReader {
 public:
  explicit StdinStreamReader(const ReaderOptions &options);
  virtual ~StdinStreamReader() = default;
  uint64 GetOffset() override;
  Status SetOffset(uint64 *offset) override;

 protected:
  Status ReadNBytes(size_t n, tstring *result) override;
  Status ReadBinarySize(size_t *size) override;

 private:
  std::shared_ptr<std::istream> input_stream_;
  std::unique_ptr<char> buffer_;
  uint64 offset_;

  TF_DISALLOW_COPY_AND_ASSIGN(StdinStreamReader);
};

class FileStreamReader : public BaseStreamReader {
 public:
  FileStreamReader(Env *env, const string &file_name,
                   const ReaderOptions &options);
  virtual ~FileStreamReader() = default;
  uint64 GetOffset() override;
  Status SetOffset(uint64 *offset) override;

 private:
  Status ReadNBytes(size_t n, tstring *result) override;
  Status ReadBinarySize(size_t *size) override;

  std::unique_ptr<io::InputStreamInterface> input_stream_;
  std::unique_ptr<RandomAccessFile> file_;
  bool last_read_failed_;

  TF_DISALLOW_COPY_AND_ASSIGN(FileStreamReader);
};

class PBIterator {
 public:
  PBIterator() = default;
  explicit PBIterator(FeaturePruningType feature_pruning_type,
                      const ReaderOptions &options);
  PBIterator(Env *env, const string &file_name,
             FeaturePruningType feature_pruning_type,
             const ReaderOptions &options);
  virtual ~PBIterator() = default;

  virtual Status next(uint64 *offset, uint32_t *data_source_key,
                      tstring *serialized);

  virtual Status next(uint64 *offset, ::parser::proto::Instance *pb);

  virtual Status next(uint64 *offset, ::monolith::io::proto::Example *pb);

  virtual Status next(uint64 *offset, ::monolith::io::proto::ExampleBatch *pb);

  uint64 GetOffset();
  Status SetOffset(uint64 *offset);

 protected:
  FeaturePruningType feature_pruning_type_ = PRUNING_RAW_FEATURE;
  std::unique_ptr<BaseStreamReader> reader_;
  std::unique_ptr<FeaturePruningByteCounter> counter_;

  TF_DISALLOW_COPY_AND_ASSIGN(PBIterator);
};

class ExampleBatchIterator : public PBIterator {
 public:
  ExampleBatchIterator() = default;
  explicit ExampleBatchIterator(FeaturePruningType feature_pruning_type,
                                const ReaderOptions &options,
                                FeatureNameMapper *mapper);
  ExampleBatchIterator(Env *env, const string &file_name,
                       FeaturePruningType feature_pruning_type,
                       const ReaderOptions &options, FeatureNameMapper *mapper);

  Status next(uint64 *offset, uint32_t *data_source_key, tstring *serialized);
  Status next(uint64 *offset, ::monolith::io::proto::ExampleBatch *pb);
  Status next(uint64 *offset, ::parser::proto::Instance *pb) override;
  Status next(uint64 *offset, ::monolith::io::proto::Example *pb) override;

 private:
  Status next_internal(uint64 *offset);
  int index_ = 0, batch_size_ = 0;
  monolith::io::proto::ExampleBatch *cur_;
  std::unique_ptr<google::protobuf::Arena> arena_;
  FeatureNameMapper *mapper_;
  TF_DISALLOW_COPY_AND_ASSIGN(ExampleBatchIterator);
};
}  // namespace monolith_tf
}  // namespace tensorflow
#endif  // MONOLITH_NATIVE_TRAINING_DATA_TRAINING_INSTANCE_CC_DATA_READER_H_
