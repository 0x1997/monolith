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

#include <bitset>
#include <climits>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#include "monolith/native_training/data/training_instance/cc/data_reader.h"
#include "monolith/native_training/data/training_instance/cc/snappy_inputbuffer.h"

namespace tensorflow {
namespace monolith_tf {
using EFeature = ::monolith::io::proto::Feature;
using Example = ::monolith::io::proto::Example;
using ExampleBatch = ::monolith::io::proto::ExampleBatch;
using FeatureListType = ::monolith::io::proto::FeatureListType;
using IFeature = ::idl::matrix::proto::Feature;
using Instance = ::parser::proto::Instance;
using LineId = ::idl::matrix::proto::LineId;

static const size_t kLengthSize = 8;
const int kDEFAULT_SNAPPY_BUFFER_SIZE = 64 * 1024 * 1024;  // 64MB
size_t FALLBACK_RESERVE_VALUE = 0xfefefefe;

Status AddFeature(const std::string &name, const EFeature &efeat,
                  Instance *instance) {
  if (name == "__LINE_ID__") {
    const auto &line_id = efeat.bytes_list().value(0);
    bool ok = instance->mutable_line_id()->ParseFromArray(line_id.data(),
                                                          line_id.size());
    if (!ok) {
      return errors::FailedPrecondition("Failed to parse the LineId");
    }
  } else if (name == "__LABEL__") {
    const auto &float_list = efeat.float_list();
    for (const auto &value : float_list.value()) {
      instance->add_label(value);
    }
  } else if (name == "instance_weight") {
    float instance_weight = 1.0;
    if (efeat.float_list().value_size() > 0) {
      instance_weight = efeat.float_list().value(0);
    }
    instance->set_instance_weight(instance_weight);
  } else {
    switch (efeat.type_case()) {
      case EFeature::TypeCase::TYPE_NOT_SET:
        break;
      case EFeature::TypeCase::kFidV1List:
        for (const auto &value : efeat.fid_v1_list().value()) {
          instance->add_fid(value);
        }
        break;
      default:
        IFeature *ifeat = instance->add_feature();
        ifeat->set_name(name);
        switch (efeat.type_case()) {
          case EFeature::TypeCase::kFidV2List:
            for (const auto &fid : efeat.fid_v2_list().value()) {
              ifeat->add_fid(fid);
            }
            break;
          case EFeature::TypeCase::kFloatList:
            for (const auto &fv : efeat.float_list().value()) {
              ifeat->add_float_value(fv);
            }
            break;
          case EFeature::TypeCase::kInt64List:
            for (const auto &iv : efeat.int64_list().value()) {
              ifeat->add_int64_value(iv);
            }
            break;
          case EFeature::TypeCase::kBytesList:
            for (const auto &bv : efeat.bytes_list().value()) {
              ifeat->add_bytes_value(bv);
            }
            break;
          case EFeature::TypeCase::kFidV2Lists:
            for (const auto &elist : efeat.fid_v2_lists().list()) {
              auto *ilist = ifeat->add_fid_list();
              for (const auto &value : elist.value()) {
                ilist->add_value(value);
              }
            }
            break;
          case EFeature::TypeCase::kFloatLists:
            for (const auto &elist : efeat.float_lists().list()) {
              auto *ilist = ifeat->add_float_list();
              for (const auto &value : elist.value()) {
                ilist->add_value(value);
              }
            }
            break;
          case EFeature::TypeCase::kInt64Lists:
            for (const auto &elist : efeat.int64_lists().list()) {
              auto *ilist = ifeat->add_int64_list();
              for (const auto &value : elist.value()) {
                ilist->add_value(value);
              }
            }
            break;
          case EFeature::TypeCase::kBytesLists:
            for (const auto &elist : efeat.bytes_lists().list()) {
              auto *ilist = ifeat->add_bytes_list();
              for (const auto &value : elist.value()) {
                ilist->add_value(value);
              }
            }
            break;
          default:
            break;
        }
        break;
    }
  }

  return Status::OK();
}

void ExtendExample(Example *pb, FeatureNameMapper *mapper /* = nullptr*/) {
  bool has_line_id = false, has_label = false, has_instance_weight = false;
  for (uint i = 0; i < pb->named_feature_size(); ++i) {
    auto &named_feature = *(pb->mutable_named_feature(i));
    if (mapper) {
      int id;
      int32_t sorted_id = -1;
      if (mapper->GetIdByName(named_feature.name(), &id, &sorted_id)) {
        named_feature.set_sorted_id(sorted_id);
      }
    }
    if (named_feature.name() == "__LINE_ID__") {
      has_line_id = true;
      const auto &line_id = named_feature.feature().bytes_list().value(0);
      pb->mutable_line_id()->ParseFromArray(line_id.data(), line_id.size());
    } else if (named_feature.name() == "__LABEL__") {
      has_label = true;
      const auto &float_list = named_feature.feature().float_list();
      for (const auto &value : float_list.value()) {
        pb->add_label(value);
      }
    } else if (named_feature.name() == "instance_weight") {
      has_instance_weight = true;
      float instance_weight = 1.0;
      if (named_feature.feature().float_list().value_size() > 0) {
        instance_weight = named_feature.feature().float_list().value(0);
      }
      pb->set_instance_weight(instance_weight);
    }

    if (has_line_id && has_label && has_instance_weight) {
      break;
    }
  }
}

Status ExampleToInstance(Example *example, Instance *instance) {
  for (const auto &named_feature : example->named_feature()) {
    std::string name = named_feature.name();
    const EFeature &efeat = named_feature.feature();
    TF_RETURN_IF_ERROR(AddFeature(name, efeat, instance));
  }

  // (todo): named_raw_feature is not supported in instance
  return Status::OK();
}

Status InstanceToExample(Instance *instance, Example *example) {
  int index = 0;
  if (instance->has_line_id()) {
    example->mutable_line_id()->CopyFrom(instance->line_id());
  }

  if (instance->label_size() > 0) {
    for (const auto &value : instance->label()) {
      example->add_label(value);
    }
  }

  if (instance->has_instance_weight()) {
    example->set_instance_weight(instance->instance_weight());
  } else {
    example->set_instance_weight(1.0);
  }

  if (instance->value_size() > 0) {
    auto *named_feature = example->add_named_feature();
    named_feature->set_name("value");
    named_feature->set_id(index++);
    auto *efeat = named_feature->mutable_feature();
    auto *float_list = efeat->mutable_float_list();
    for (const auto &value : instance->value()) {
      float_list->add_value(value);
    }
  }

  std::unordered_map<int, ::monolith::io::proto::FidList *> slot_to_efeat_;
  const auto &fids = instance->fid();
  for (const auto &fid : fids) {
    int slot_id = fid >> 54;
    auto it = slot_to_efeat_.find(slot_id);
    ::monolith::io::proto::FidList *fid_v1_list = nullptr;
    if (it != slot_to_efeat_.end()) {
      fid_v1_list = it->second;
    } else {
      auto *named_feature = example->add_named_feature();
      named_feature->set_name(absl::StrCat("fc_slot_", slot_id));
      named_feature->set_id(index++);
      fid_v1_list = named_feature->mutable_feature()->mutable_fid_v1_list();
      slot_to_efeat_.emplace(slot_id, fid_v1_list);
    }

    fid_v1_list->add_value(fid);
  }

  for (const auto &ifeat : instance->feature()) {
    auto *named_feature = example->add_named_feature();
    named_feature->set_name(ifeat.name());
    named_feature->set_id(index++);

    auto *efeat = named_feature->mutable_feature();

    if (ifeat.fid_size() > 0) {
      auto *list = efeat->mutable_fid_v2_list();
      for (const auto &value : ifeat.fid()) {
        list->add_value(value);
      }
    } else if (ifeat.float_value_size() > 0) {
      auto *list = efeat->mutable_float_list();
      for (const auto &value : ifeat.float_value()) {
        list->add_value(value);
      }
    } else if (ifeat.int64_value_size() > 0) {
      auto *list = efeat->mutable_int64_list();
      for (const auto &value : ifeat.int64_value()) {
        list->add_value(value);
      }
    } else if (ifeat.bytes_value_size() > 0) {
      auto *bytes_list = efeat->mutable_bytes_list();
      for (const auto &value : ifeat.bytes_value()) {
        bytes_list->add_value(value);
      }
    } else if (ifeat.fid_list_size() > 0) {
      auto *elists = efeat->mutable_fid_v2_lists();
      for (const auto &ilist : ifeat.fid_list()) {
        auto *elist = elists->add_list();
        for (const auto &value : ilist.value()) {
          elist->add_value(value);
        }
      }
    } else if (ifeat.float_list_size() > 0) {
      auto *elists = efeat->mutable_float_lists();
      for (const auto &ilist : ifeat.float_list()) {
        auto *list = elists->add_list();
        for (const auto &value : ilist.value()) {
          list->add_value(value);
        }
      }
    } else if (ifeat.int64_list_size() > 0) {
      auto *elists = efeat->mutable_int64_lists();
      for (const auto &ilist : ifeat.int64_list()) {
        auto *list = elists->add_list();
        for (const auto &value : ilist.value()) {
          list->add_value(value);
        }
      }
    } else if (ifeat.bytes_list_size() > 0) {
      auto *elists = efeat->mutable_bytes_lists();
      for (const auto &ilist : ifeat.bytes_list()) {
        auto *list = elists->add_list();
        for (const auto &value : ilist.value()) {
          list->add_value(value);
        }
      }
    } else {
      LOG(INFO) << absl::StrCat("empty ", ifeat.name());
    }
  }

  return Status::OK();
}

Status ExampleBatchToInstance(ExampleBatch *example_batch, int index,
                              Instance *instance) {
  for (const auto &named_feature_list :
       example_batch->named_feature_list()) {  // NamedFeatureList
    const std::string &name = named_feature_list.name();
    const EFeature &efeat = named_feature_list.type() == FeatureListType::SHARED
                                ? named_feature_list.feature(0)
                                : named_feature_list.feature(index);
    TF_RETURN_IF_ERROR(AddFeature(name, efeat, instance));
  }
  instance->set_data_source_key(example_batch->data_source_key());

  // (todo): named_raw_feature is not supported in instance
  return Status::OK();
}

Status ExampleBatchToExample(ExampleBatch *example_batch, int index,
                             Example *example,
                             FeaturePruningType feature_pruning_type,
                             FeatureNameMapper *mapper) {
  profiler::TraceMe activity([]() { return "ExampleBatchToExample"; });
  for (const auto &named_feature : example_batch->named_feature_list()) {
    const auto &efeat = named_feature.type() == FeatureListType::SHARED
                            ? named_feature.feature(0)
                            : named_feature.feature(index);
    if (named_feature.name() == "__LINE_ID__") {
      const auto &line_id = efeat.bytes_list().value(0);
      bool ok = example->mutable_line_id()->ParseFromArray(line_id.data(),
                                                           line_id.size());
      if (!ok) {
        return errors::FailedPrecondition("Failed to parse the LineId");
      }
    } else if (named_feature.name() == "__LABEL__") {
      const auto &float_list = efeat.float_list();
      for (const auto &value : float_list.value()) {
        example->add_label(value);
      }
    } else if (named_feature.name() == "instance_weight") {
      float instance_weight = 1.0;
      if (efeat.float_list().value_size() > 0) {
        instance_weight = efeat.float_list().value(0);
      }
      example->set_instance_weight(instance_weight);
    } else if (feature_pruning_type != PRUNING_FEATURE) {
      // FeatureNameMapper
      if (mapper == nullptr) {
        return errors::InvalidArgument(
            "FeatureNameMapper should be specified, while we got "
            "mapper==nullptr");
      }
      if (mapper->IsAvailable()) {
        LOG_FIRST_N(INFO, 1) << mapper->DebugString();
        int32_t id = -1;
        int32_t sorted_id = -1;
        bool found = mapper->GetIdByName(named_feature.name(), &id, &sorted_id);
        if (found) {
          auto *out = example->add_named_feature();
          out->set_name(named_feature.name());
          out->set_id(named_feature.id());
          out->set_sorted_id(sorted_id);
          out->mutable_feature()->MergeFrom(efeat);
        }
      } else {
        auto *out = example->add_named_feature();
        out->set_name(named_feature.name());
        out->set_id(named_feature.id());
        out->mutable_feature()->MergeFrom(efeat);
      }
    }
  }

  if (feature_pruning_type != PRUNING_RAW_FEATURE) {
    for (const auto &named_feature : example_batch->named_raw_feature_list()) {
      const auto &efeat = named_feature.type() == FeatureListType::SHARED
                              ? named_feature.raw_feature(0)
                              : named_feature.raw_feature(index);
      auto *out = example->add_named_raw_feature();
      out->set_name(named_feature.name());
      out->set_id(named_feature.id());
      out->mutable_raw_feature()->MergeFrom(efeat);
    }
  }

  example->set_data_source_key(example_batch->data_source_key());

  return Status::OK();
}

BaseStreamReader::BaseStreamReader(const ReaderOptions &options)
    : options_(options) {}

Status BaseStreamReader::ReadDataHeader(uint8_t *pb_type,
                                        uint32_t *data_source_key) {
  size_t size = 0, aggregate_page_sortid_size = 0;
  if (options_.lagrangex_header) {
    // *dtype = ins_type == 0 ? PROTO_INSTANCE : EXAMPLE_PB;
    TF_RETURN_IF_ERROR(ReadBinarySize(&size));
    uint64_t lgx_header = static_cast<uint64_t>(size);
    *pb_type = static_cast<uint8_t>(lgx_header & 0xff);
    uint32_t source = static_cast<uint32_t>(lgx_header);
    *data_source_key = (source >> 8) << 8;
  } else {
    *pb_type = 0;
    if (options_.kafka_dump_prefix) {
      TF_RETURN_IF_ERROR(ReadBinarySize(&size));
      if (size == 0) {
        TF_RETURN_IF_ERROR(ReadBinarySize(&size));
      } else {
        aggregate_page_sortid_size = size;
      }
    }
    if (options_.has_sort_id) {
      if (aggregate_page_sortid_size == 0) {
        TF_RETURN_IF_ERROR(ReadBinarySize(&size));
      } else {
        size = aggregate_page_sortid_size;
      }
      tstring sort_id;
      TF_RETURN_IF_ERROR(ReadNBytes(size, &sort_id));
    }
    if (options_.kafka_dump) {
      TF_RETURN_IF_ERROR(ReadBinarySize(&size));
    }
  }

  return Status::OK();
}

Status BaseStreamReader::ReadPBBytes(uint8_t *pb_type,
                                     uint32_t *data_source_key,
                                     tstring *instance) {
  TF_RETURN_IF_ERROR(ReadDataHeader(pb_type, data_source_key));
  size_t size;
  TF_RETURN_IF_ERROR(ReadBinarySize(&size));
  // Don't know whether FALLBACK_RESERVE_VALUE is in use.
  DCHECK_NE(size, FALLBACK_RESERVE_VALUE);
  TF_RETURN_IF_ERROR(ReadNBytes(size, instance));
  return Status::OK();
}

FileStreamReader::FileStreamReader(Env *env, const string &file_name,
                                   const ReaderOptions &options)
    : BaseStreamReader(options), last_read_failed_(false) {
  env->NewRandomAccessFile(file_name, &file_);

  if (options.use_snappy) {
    int64 buffer_size = options.buffer_size > 0 ? options.buffer_size
                                                : kDEFAULT_SNAPPY_BUFFER_SIZE;
    input_stream_.reset(
        new io::ByteSnappyInputBuffer(file_.get(), buffer_size, buffer_size));
  } else {
    input_stream_.reset(new io::RandomAccessInputStream(file_.get()));
    if (options.buffer_size > 0) {
      input_stream_.reset(new io::BufferedInputStream(
          input_stream_.release(), options.buffer_size, true));
    }
  }
}

Status FileStreamReader::ReadNBytes(size_t n, tstring *result) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(n, result));
  if (result->size() != n) {
    last_read_failed_ = true;
    if (result->empty()) {
      return errors::OutOfRange("eof");
    } else {
      return errors::DataLoss("truncated record");
    }
  }

  return Status::OK();
}

Status FileStreamReader::ReadBinarySize(size_t *size) {
  tstring result;
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(sizeof(size_t), &result));
  *size = static_cast<size_t>(core::DecodeFixed64(result.data()));
  return Status::OK();
}

uint64 FileStreamReader::GetOffset() { return input_stream_->Tell(); }

Status FileStreamReader::SetOffset(uint64 *offset) {
  int64 curr_pos = input_stream_->Tell();
  int64 desired_pos = static_cast<int64>(*offset);
  if (curr_pos > desired_pos || curr_pos < 0 /* EOF */ ||
      (curr_pos == desired_pos && last_read_failed_)) {
    last_read_failed_ = false;
    TF_RETURN_IF_ERROR(input_stream_->Reset());
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos));
  } else if (curr_pos < desired_pos) {
    TF_RETURN_IF_ERROR(input_stream_->SkipNBytes(desired_pos - curr_pos));
  }
  DCHECK_EQ(desired_pos, input_stream_->Tell());
  return Status::OK();
}

// TODO(leqi.zou): Make input stream async and static.
// Currently the problem is if we are unable to read N bytes,
// the code is not cancellable (CTRL + C not working).
StdinStreamReader::StdinStreamReader(const ReaderOptions &options)
    : BaseStreamReader(options) {
  input_stream_.reset(&std::cin, [](...) {});
  buffer_.reset(new char[options.buffer_size]);
}

Status StdinStreamReader::ReadNBytes(size_t n, tstring *result) {
  if (input_stream_->eof() || n > options_.buffer_size ||
      !input_stream_->read(buffer_.get(), n)) {
    if (n > options_.buffer_size) {
      LOG(WARNING) << "Buffer size may be too small! Should be bigger than "
                   << n;
    }
    LOG(INFO) << "stdin meets EOF";
    size_t size = n < options_.buffer_size ? n : options_.buffer_size;
    if (size > 0) {
      tstring remaining;
      remaining.assign(buffer_.get(), size);
      LOG(INFO) << "EOF > " << remaining;
    } else {
      LOG(INFO) << "EOF > empty ";
    }

    return errors::OutOfRange("eof");
  }
  offset_ += n;
  result->assign(buffer_.get(), n);
  return Status::OK();
}

Status StdinStreamReader::ReadBinarySize(size_t *size) {
  bool result = static_cast<bool>(
      input_stream_->read(reinterpret_cast<char *>(size), sizeof(*size)));
  if (input_stream_->eof()) {
    LOG(INFO) << "stdin meets EOF";
    return errors::OutOfRange("eof");
  } else if (input_stream_->fail() || !result) {
    return errors::DataLoss("streaming load broken");
  }
  offset_ += kLengthSize;
  return Status::OK();
}

uint64 StdinStreamReader::GetOffset() { return offset_; }

Status StdinStreamReader::SetOffset(uint64 *offset) {
  if (offset_ < *offset) {
    tstring buf;
    size_t size = *offset - offset_;
    bool result = static_cast<bool>(input_stream_->read(buf.data(), size));
    if (input_stream_->eof()) {
      return errors::OutOfRange("eof");
    } else if (input_stream_->fail() || !result) {
      return errors::DataLoss("streaming load broken");
    }
    offset_ = *offset;
    return Status::OK();
  }
  if (offset_ == *offset) {
    return Status::OK();
  } else {
    return errors::FailedPrecondition(
        "Cannot set the offset of stdin ahead of current position");
  }
}

PBIterator::PBIterator(FeaturePruningType feature_pruning_type,
                       const ReaderOptions &options)
    : feature_pruning_type_(feature_pruning_type),
      reader_{std::make_unique<StdinStreamReader>(options)},
      counter_(std::make_unique<FeaturePruningByteCounter>()) {}

PBIterator::PBIterator(Env *env, const string &file_name,
                       FeaturePruningType feature_pruning_type,
                       const ReaderOptions &options)
    : feature_pruning_type_(feature_pruning_type),
      reader_{std::make_unique<FileStreamReader>(env, file_name, options)},
      counter_(std::make_unique<FeaturePruningByteCounter>()) {}

Status PBIterator::next(uint64 *offset, uint32_t *data_source_key,
                        tstring *serialized) {
  uint8_t pb_type;
  reader_->SetOffset(offset);
  TF_RETURN_IF_ERROR(
      reader_->ReadPBBytes(&pb_type, data_source_key, serialized));
  return Status::OK();
}

Status PBIterator::next(uint64 *offset, Instance *pb) {
  tstring buf;
  uint32_t data_source_key;
  TF_RETURN_IF_ERROR(next(offset, &data_source_key, &buf));
  bool ok = pb->ParseFromArray(buf.data(), buf.size());
  pb->set_data_source_key(data_source_key);
  if (ok) {
    return Status::OK();
  } else {
    return errors::FailedPrecondition("Failed to parse the Instance.");
  }
}

Status PBIterator::next(uint64 *offset, Example *pb) {
  tstring buf;
  uint32_t data_source_key;
  TF_RETURN_IF_ERROR(next(offset, &data_source_key, &buf));
  bool ok = pb->ParseFromArray(buf.data(), buf.size());
  pb->set_data_source_key(data_source_key);

  if (ok) {
    ExtendExample(pb);
    counter_->AddByteSize(pb->ByteSizeLong());
    if (feature_pruning_type_ == PRUNING_FEATURE) {
      auto *named_features = pb->mutable_named_feature();
      named_features->erase(named_features->begin(), named_features->end());
    } else if (feature_pruning_type_ == PRUNING_RAW_FEATURE) {
      auto *named_raw_feature = pb->mutable_named_raw_feature();
      named_raw_feature->erase(named_raw_feature->cbegin(),
                               named_raw_feature->cend());
    }
    counter_->AddByteSizePruned(pb->ByteSizeLong());
    LOG_EVERY_N_SEC(INFO, 180) << counter_->DebugString();

    return Status::OK();
  } else {
    return errors::FailedPrecondition("Failed to parse the Example.");
  }
}

Status PBIterator::next(uint64 *offset, ExampleBatch *pb) {
  tstring buf;
  uint32_t data_source_key;
  TF_RETURN_IF_ERROR(next(offset, &data_source_key, &buf));
  bool ok = pb->ParseFromArray(buf.data(), buf.size());
  pb->set_data_source_key(data_source_key);

  counter_->AddByteSize(pb->ByteSizeLong());
  if (feature_pruning_type_ == PRUNING_FEATURE) {
    auto *named_feature_list = pb->mutable_named_feature_list();
    auto it = named_feature_list->begin();
    while (it != named_feature_list->end()) {
      if (it->name() != "__LABEL__" && it->name() != "__LINE_ID__" &&
          it->name() != "instance_weight") {
        // if erase, it will move to the next element
        named_feature_list->erase(it);
      } else {
        ++it;
      }
    }
  } else if (feature_pruning_type_ == PRUNING_RAW_FEATURE) {
    auto *named_raw_feature_list = pb->mutable_named_raw_feature_list();
    named_raw_feature_list->erase(named_raw_feature_list->begin(),
                                  named_raw_feature_list->end());
  }

  counter_->AddByteSizePruned(pb->ByteSizeLong());
  LOG_EVERY_N_SEC(INFO, 180) << counter_->DebugString();

  if (ok) {
    return Status::OK();
  } else {
    return errors::FailedPrecondition("Failed to parse the ExampleBatch.");
  }
}

uint64 PBIterator::GetOffset() { return reader_->GetOffset(); }

Status PBIterator::SetOffset(uint64 *offset) {
  return reader_->SetOffset(offset);
}

ExampleBatchIterator::ExampleBatchIterator(
    FeaturePruningType feature_pruning_type, const ReaderOptions &options,
    FeatureNameMapper *mapper)
    : PBIterator(feature_pruning_type, options), mapper_(mapper) {
  arena_ = std::make_unique<google::protobuf::Arena>();
  cur_ = google::protobuf::Arena::CreateMessage<ExampleBatch>(arena_.get());
}

ExampleBatchIterator::ExampleBatchIterator(
    Env *env, const string &file_name, FeaturePruningType feature_pruning_type,
    const ReaderOptions &options, FeatureNameMapper *mapper)
    : PBIterator(env, file_name, feature_pruning_type, options),
      mapper_(mapper) {
  arena_ = std::make_unique<google::protobuf::Arena>();
  cur_ = google::protobuf::Arena::CreateMessage<ExampleBatch>(arena_.get());
}

Status ExampleBatchIterator::next_internal(uint64 *offset) {
  if (index_ < batch_size_ - 1) {
    index_++;
    return Status::OK();
  }

  profiler::TraceMe activity([]() { return "ReadAndDeserialize"; });
  uint8_t pb_type;
  uint32_t data_source_key;
  tstring buf;
  reader_->SetOffset(offset);
  arena_ = std::make_unique<google::protobuf::Arena>();
  cur_ = google::protobuf::Arena::CreateMessage<ExampleBatch>(arena_.get());

  TF_RETURN_IF_ERROR(reader_->ReadPBBytes(&pb_type, &data_source_key, &buf));
  bool ok = cur_->ParseFromArray(buf.data(), buf.size());
  counter_->AddByteSize(cur_->ByteSizeLong());
  cur_->set_data_source_key(data_source_key);

  if (!ok) {
    return errors::FailedPrecondition("Failed to parse the ExampleBatch.");
  } else {
    index_ = 0;
    batch_size_ = cur_->batch_size();
    return Status::OK();
  }
}

Status ExampleBatchIterator::next(uint64 *offset, uint32_t *data_source_key,
                                  tstring *serialized) {
  uint8_t pb_type;
  reader_->SetOffset(offset);
  TF_RETURN_IF_ERROR(
      reader_->ReadPBBytes(&pb_type, data_source_key, serialized));
  return Status::OK();
}

Status ExampleBatchIterator::next(uint64 *offset, ExampleBatch *pb) {
  uint8_t pb_type;
  uint32_t data_source_key;
  tstring buf;
  reader_->SetOffset(offset);
  TF_RETURN_IF_ERROR(reader_->ReadPBBytes(&pb_type, &data_source_key, &buf));
  bool ok = pb->ParseFromArray(buf.data(), buf.size());
  pb->set_data_source_key(data_source_key);
  counter_->AddByteSize(pb->ByteSizeLong());

  if (feature_pruning_type_ == PRUNING_FEATURE) {
    auto *named_feature_list = pb->mutable_named_feature_list();
    auto it = named_feature_list->begin();
    while (it != named_feature_list->end()) {
      if (it->name() != "__LABEL__" && it->name() != "__LINE_ID__" &&
          it->name() != "instance_weight") {
        // if erase, it will move to the next element
        named_feature_list->erase(it);
      } else {
        ++it;
      }
    }
  } else if (feature_pruning_type_ == PRUNING_RAW_FEATURE) {
    auto *named_raw_feature_list = pb->mutable_named_raw_feature_list();
    named_raw_feature_list->erase(named_raw_feature_list->begin(),
                                  named_raw_feature_list->end());
  }

  counter_->AddByteSizePruned(pb->ByteSizeLong());
  LOG_EVERY_N_SEC(INFO, 180) << counter_->DebugString();

  if (!ok) {
    return errors::FailedPrecondition("Failed to parse the ExampleBatch.");
  } else {
    return Status::OK();
  }
}

Status ExampleBatchIterator::next(uint64 *offset, Instance *pb) {
  TF_RETURN_IF_ERROR(next_internal(offset));
  return ExampleBatchToInstance(cur_, index_, pb);
}

Status ExampleBatchIterator::next(uint64 *offset, Example *pb) {
  profiler::TraceMe activity([]() { return "ExampleBatchIteratorNext"; });
  TF_RETURN_IF_ERROR(next_internal(offset));
  Status s =
      ExampleBatchToExample(cur_, index_, pb, feature_pruning_type_, mapper_);
  counter_->AddByteSizePruned(pb->ByteSizeLong());
  LOG_EVERY_N_SEC(INFO, 3600) << counter_->DebugString();
  return s;
}
}  // namespace monolith_tf
}  // namespace tensorflow
