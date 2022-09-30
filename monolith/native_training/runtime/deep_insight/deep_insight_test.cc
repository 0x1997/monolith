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

#include "monolith/native_training/runtime/deep_insight/deep_insight.h"

#include <string>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

using monolith::deep_insight::ExtraField;
using monolith::deep_insight::FloatExtraField;
using monolith::deep_insight::Int64ExtraField;
using monolith::deep_insight::StringExtraField;
using json = nlohmann::json;

namespace monolith {
namespace deep_insight {

TEST(DeepInsightTest, Basic) {
  DeepInsight deep_insight(false);

  const std::string& model_name = "deep_insight_test_1130";
  const std::string& target = "ctr_head";

  const int BYTEDANCE_BORN = 1331481600;  // 2012/03/12
  std::time_t req_time = BYTEDANCE_BORN;

  uint32_t dim = 3;
  std::vector<float> labels = {0.1, 0.2, 0.3};
  std::vector<float> preds = {0.4, 0.5, 0.6};
  std::vector<float> sample_rates = {0.7, 0.8, 0.9};
  std::vector<std::string> targets({target});
  float di_sample_rate = 0.01f;
  int64_t train_time = deep_insight.GenerateTrainingTime();
  for (uint32_t uid = 0; uid < 1050; uid++) {
    float label = labels[uid % dim];
    float pred = preds[uid % dim];
    float sample_rate = sample_rates[uid % dim];
    std::vector<float> labelss({label}), predss({pred}),
        sample_ratess({sample_rate});
    std::vector<std::shared_ptr<ExtraField>> extra_fields;
    auto msg = deep_insight.SendV2(model_name, targets, uid, req_time,
                                   train_time, labelss, predss, sample_ratess,
                                   di_sample_rate, extra_fields, true);

    if (uid % 1000 < 1000 * di_sample_rate) {
      EXPECT_FALSE(msg.empty());
      json parsed = json::parse(msg);
      EXPECT_FLOAT_EQ(parsed["sample_rate"]["ctr_head"], sample_rate);
      EXPECT_EQ(parsed["training_time"], train_time);
    } else {
      EXPECT_TRUE(msg.empty());
    }
  }

  LOG(INFO) << "model_name: " << model_name << ", target: " << target;
  EXPECT_EQ(deep_insight.GetTotalSendCounter(), 20);
}

TEST(DeepInsightTest, PUTENV) {
  putenv(const_cast<char*>("DBUS_NAME=deep_insight_kafka_v2_auth"));
  DeepInsight deep_insight(false);

  const std::string& model_name = "deep_insight_test_1130";
  const std::string& target = "ctr_head";

  const int BYTEDANCE_BORN = 1331481600;  // 2012/03/12
  std::time_t req_time = BYTEDANCE_BORN;

  uint32_t dim = 3;
  std::vector<float> labels = {0.1, 0.2, 0.3};
  std::vector<float> preds = {0.4, 0.5, 0.6};
  std::vector<float> sample_rates = {0.7, 0.8, 0.9};
  std::vector<std::string> targets({target});
  float di_sample_rate = 0.01f;
  int64_t train_time = deep_insight.GenerateTrainingTime();
  for (uint32_t uid = 0; uid < 1050; uid++) {
    float label = labels[uid % dim];
    float pred = preds[uid % dim];
    float sample_rate = sample_rates[uid % dim];
    std::vector<float> labelss({label}), predss({pred}),
        sample_ratess({sample_rate});
    std::vector<std::shared_ptr<ExtraField>> extra_fields;
    auto msg = deep_insight.SendV2(model_name, targets, uid, req_time,
                                   train_time, labelss, predss, sample_ratess,
                                   di_sample_rate, extra_fields, true);
    if (uid % 1000 < 1000 * di_sample_rate) {
      EXPECT_FALSE(msg.empty());
      json parsed = json::parse(msg);
      EXPECT_FLOAT_EQ(parsed["sample_rate"]["ctr_head"], sample_rate);
      EXPECT_EQ(parsed["training_time"], train_time);
    } else {
      EXPECT_TRUE(msg.empty());
    }
  }

  LOG(INFO) << "model_name: " << model_name << ", target: " << target;
  EXPECT_EQ(deep_insight.GetTotalSendCounter(), 20);
}

TEST(DeepInsightTest, MultiTarget) {
  DeepInsight deep_insight(false);

  const std::string& model_name = "deep_insight_test_multitarget";

  const int BYTEDANCE_BORN = 1331481600;  // 2012/03/12
  std::time_t req_time = BYTEDANCE_BORN;

  // uint32_t dim = 3;
  std::vector<float> labels = {0.1, 0.2, 0.3};
  std::vector<float> preds = {0.4, 0.5, 0.6};
  std::vector<float> sample_rates = {0.7, 0.8, 0.9};
  std::vector<std::string> targets({"a_head", "b_head", "c_head"});
  float di_sample_rate = 1.0f;
  int64_t train_time = deep_insight.GenerateTrainingTime();
  uint32_t uid = 0;
  std::string extra_str = "extra_string_field_value";
  std::vector<std::shared_ptr<ExtraField>> extra_fields;
  extra_fields.push_back(
      std::make_shared<StringExtraField>("extra_string_field_key", extra_str));
  auto msg = deep_insight.SendV2(model_name, targets, uid, req_time, train_time,
                                 labels, preds, sample_rates, di_sample_rate,
                                 extra_fields, true);

  json parsed = json::parse(msg);
  EXPECT_FLOAT_EQ(parsed["sample_rate"]["a_head"], sample_rates.at(0));
  EXPECT_FLOAT_EQ(parsed["predict"]["b_head"], preds.at(1));
  EXPECT_FLOAT_EQ(parsed["label"]["c_head"], labels.at(2));
  EXPECT_EQ(parsed["training_time"], train_time);
  EXPECT_EQ(deep_insight.GetTotalSendCounter(), 1);
}

}  // namespace deep_insight
}  // namespace monolith
