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

#pragma once
#include <archon/client/raw_client.h>
#include <archon/lb/options.h>
#include <monolith_serving/monolith_serving_types.h>

#include "absl/status/status.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

namespace tensorflow {
namespace monolith_tf {

// Archon based communication point with PredictionService.
class PredictionServiceArchon {
 public:
  using DoneCallback = std::function<void()>;

  void Predict(
      tensorflow::serving::PredictRequest *request,
      tensorflow::serving::PredictResponse *response,
      std::function<void(absl::Status status, DoneCallback &&)> callback,
      int64_t max_rpc_deadline_millis, DoneCallback op_done);

  explicit PredictionServiceArchon(
      const std::vector<std::string> &address_list);

 private:
  explicit PredictionServiceArchon(const std::string &target_address);
  std::unique_ptr<archon::client::RawClient> raw_client_;

  archon::common::RequestOptions opt;
};

}  // namespace monolith_tf
}  // namespace tensorflow
