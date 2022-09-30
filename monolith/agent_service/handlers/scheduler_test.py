# Copyright 2022 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
from unittest import mock
import threading
import unittest

import grpc

from monolith.agent_service import agent_service_pb2
from monolith.agent_service import agent_service_pb2_grpc
from monolith.agent_service import data_def
from monolith.agent_service.handlers import scheduler


class FakeAgentServicer(agent_service_pb2_grpc.AgentServiceServicer):

  def __init__(self, throw_error=False):
    self._throw_error = throw_error

  def GetResource(self, req: agent_service_pb2.GetResourceRequest,
                  ctx: grpc.ServicerContext):
    if self._throw_error:
      ctx.abort(grpc.StatusCode.UNAVAILABLE, "")
      return
    resp = agent_service_pb2.GetResourceResponse()
    resp.memory = 100
    return resp


class SchedulerTest(unittest.TestCase):

  def testBasic(self):
    address = "unix:basic"
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(2))
    agent_service_pb2_grpc.add_AgentServiceServicer_to_server(
        FakeAgentServicer(), server)
    server.add_insecure_port(address)
    server.start()

    zk = mock.MagicMock()
    zk.get_published_path.return_value = []
    zk.resources = []
    for replica_id in range(2):
      for shard_id in range(2):
        zk.resources.append(
            data_def.ResourceSpec(address=address,
                                  replica_id=replica_id,
                                  shard_id=shard_id))
    model_meta = data_def.ModelMeta(model_name="test_model",
                                    model_dir="test_path",
                                    ckpt="ckpt")
    event = data_def.Event(data=model_meta.serialize())
    model_info = {
        "entry": (50, "entry/12345"),
        "ps_0": (50, "ps_0/12345"),
        "ps_1": (50, "ps_1/12345"),
    }
    model_info_getter = mock.MagicMock(return_value=model_info)
    s = scheduler.Scheduler(zk, model_info_getter=model_info_getter)
    s.handle(event)

    def get_meta(replica_id, shard_id, sub_models):
      pm = data_def.PublishMeta(replica_id=replica_id,
                                shard_id=shard_id,
                                model_name="test_model",
                                num_ps=2,
                                sub_models={})
      for sub_model in sub_models:
        pm.sub_models[sub_model] = f"{sub_model}/12345"
      return pm

    publish_metas = [
        get_meta(0, 0, ["entry", "ps_0"]),
        get_meta(0, 1, ["entry", "ps_1"]),
        get_meta(1, 0, ["entry", "ps_0"]),
        get_meta(1, 1, ["entry", "ps_1"]),
    ]
    for meta in publish_metas:
      meta.total_publish_num = 4

    zk.publish_loadding.assert_called_with(publish_metas)

  def test_fallback(self):
    resources = [
        data_def.ResourceSpec(address="unix:fallback_invalid_addr",
                              shard_id=0,
                              replica_id=0,
                              memory=100)
    ]
    l = scheduler.get_realtime_resources(resources)
    self.assertEqual(l[0], scheduler.RealtimeResource(0, 0, 100))


if __name__ == "__main__":
  unittest.main()
