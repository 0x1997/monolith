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

#include "monolith/native_training/runtime/ops/prediction_service_archon.h"

#include <archon/log/rpc_log.h>
#include <archon/transport/socket_pool_options.h>
#include <cpputil/common/string/string_printf.h>
#include <cpputil/metrics2/metrics.h>
#include <gflags/gflags.h>

DEFINE_int32(archon_entry_to_ps_rpc_timeout, 10000, "");
DEFINE_int32(archon_entry_to_ps_conn_timeout, 30, "");
DEFINE_int32(archon_entry_to_ps_rpc_retry, 2, "");
DEFINE_bool(archon_entry_to_ps_long_conn_enable, true, "");
DEFINE_int32(archon_entry_to_ps_long_conn_num, 100, "");
DEFINE_bool(archon_entry_to_ps_extra_metrics, false, "");

namespace tensorflow {
namespace monolith_tf {
PredictionServiceArchon::PredictionServiceArchon(
    const std::string &target_address) {
  folly::SocketAddress addr;
  using archon::client::RawClient;
  using archon::common::ServiceMeta;
  using archon::lb::LoadBalanceOptions;

  INFO("[prediction_service_archon] target_address %s\n",
       target_address.c_str());

  addr.setFromHostPort(target_address);
  ServiceMeta meta(addr.getIPAddress().str(), addr.getPort(),
                   "data.monolith.ps");
  archon::transport::SocketPoolOptions spo(
      FLAGS_archon_entry_to_ps_long_conn_enable,
      FLAGS_archon_entry_to_ps_long_conn_num);
  raw_client_ = std::make_unique<RawClient>("remote_predict_archon", meta,
                                            LoadBalanceOptions(), spo);
}

PredictionServiceArchon::PredictionServiceArchon(
    const std::vector<std::string> &address_list) {
  using archon::client::RawClient;
  using archon::common::RequestOptions;
  using archon::common::ServiceMeta;
  using archon::lb::LoadBalanceOptions;

  size_t n = address_list.size();
  std::vector<std::string> host;
  std::vector<int> port;
  host.reserve(n);
  port.reserve(n);

  for (const auto &address : address_list) {
    folly::SocketAddress addr;
    addr.setFromHostPort(address);
    host.emplace_back(addr.getIPAddress().str());
    port.push_back(addr.getPort());
  }
  ServiceMeta meta(host, port, "data.monolith.ps");
  archon::transport::SocketPoolOptions spo(
      FLAGS_archon_entry_to_ps_long_conn_enable,
      FLAGS_archon_entry_to_ps_long_conn_num);
  raw_client_ = std::make_unique<RawClient>("remote_predict_archon", meta,
                                            LoadBalanceOptions(), spo);

  opt.set_timeout_options(
      RequestOptions::TimeoutOptions(FLAGS_archon_entry_to_ps_rpc_timeout,
                                     FLAGS_archon_entry_to_ps_conn_timeout));
  opt.set_retry_options(archon::common::RequestOptions::RetryOptions(
      FLAGS_archon_entry_to_ps_rpc_retry, 0.1));
}

void PredictionServiceArchon::Predict(
    tensorflow::serving::PredictRequest *request,
    tensorflow::serving::PredictResponse *response,
    std::function<void(absl::Status status, DoneCallback &&)> callback,
    int64_t max_rpc_deadline_millis, DoneCallback op_done) {
  using archon::client::ResponseState;
  using archon::common::RequestContext;
  using idl::monolith_serving::MonolithReq;
  using idl::monolith_serving::MonolithRsp;
  using TagkvList = std::vector<std::pair<std::string, std::string>>;

  auto req = std::make_shared<MonolithReq>();
  auto rsp_state = std::make_shared<ResponseState<MonolithRsp>>();
  size_t req_size = request->ByteSizeLong();
  req->req_data = folly::IOBuf::create(req_size);
  request->SerializeToArray(req->req_data->writableData(), req_size);
  req->req_data->append(req_size);

  RequestContext ctx;
  ctx.fill_req_base(*req);
  cpputil::TimeCost tc;
  TagkvList tagkv{{"model_name", request->model_spec().name()},
                  {"model_signature", request->model_spec().signature_name()}};

  if (FLAGS_archon_entry_to_ps_extra_metrics) {
    cpputil::metrics2::Metrics::emit_timer("fetch_ps.req_bytes", req_size,
                                           tagkv);
  }

  auto fut =
      raw_client_->async_call("predict", req, rsp_state, ctx, opt)
          .then([rsp_state, done = std::move(op_done), response,
                 callback = std::move(callback), tc = std::move(tc),
                 tagkv = std::move(tagkv),
                 log_id = ctx.get_log_id()](int ret) mutable {
            absl::Status status;
            if (ret) {
              status = absl::Status(static_cast<absl::StatusCode>(ret),
                                    "Archon error code " + std::to_string(ret));
            } else {
              auto &buf = rsp_state->get_response()->rsp_data;
              auto input_stream =
                  std::make_unique<archon::protobuf::ProtobufInputStream>(buf);
              response->ParseFromZeroCopyStream(input_stream.get());
            }
            if (ret == 0 && FLAGS_archon_entry_to_ps_extra_metrics) {
              cpputil::metrics2::Metrics::emit_timer(
                  "fetch_ps.rsp_bytes", response->ByteSizeLong(), tagkv);
            }

            archon::log::RpcLog::call(
                log_id, tc.get_elapsed(), ret, "-", "-",
                rsp_state->get_remote_dc(), "-", "-", "-", "predict", "predict",
                rsp_state->get_remote_host() + ":" +
                    std::to_string(rsp_state->get_remote_port()));
            callback(status, std::forward<DoneCallback>(done));
          });
}

}  // namespace monolith_tf
}  // namespace tensorflow
