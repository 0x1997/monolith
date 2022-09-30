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

from absl import logging, flags
import grpc
import os
import socket

from monolith.agent_service import utils
from monolith.agent_service.agent_service_pb2 import HeartBeatRequest, ServerType, \
  GetReplicasRequest
from monolith.agent_service.agent_service_pb2_grpc import AgentServiceStub


flags.DEFINE_enum("cmd_type", 'hb', ['hb', 'gr'], help='cmd_type', short_name='c')
flags.DEFINE_string("config", '/opt/tiger/byterec_tce_ws/MonolithServingInst_1/agent.conf', 
                    help='agent_conf file', short_name='f')
flags.DEFINE_enum("server_type", 'ps', ['ps', 'entry'], help='server_type', short_name='st')
flags.DEFINE_integer("task", 0, help='task', short_name='t')
FLAGS = flags.FLAGS


class SvrClient(object):
  def __init__(self, config) -> None:
    if isinstance(config, str):
      self.agent_conf = utils.AgentConfig.from_file(config)
    else:
      self.agent_conf = config
    self._stub = None

  @property
  def stub(self):
    if self._stub is None:
      local_host = socket.gethostbyname(socket.gethostname())
      target = f'{os.environ.get("MY_HOST_IP", local_host)}:{self.agent_conf.agent_port}'
      channel = grpc.insecure_channel(target)
      self._stub = AgentServiceStub(channel)
    
    return self._stub
  
  def get_server_type(self, st):
    if isinstance(st, str):
      if FLAGS.server_type == 'ps':
        return ServerType.PS
      elif FLAGS.server_type == 'entry':
        return ServerType.ENTRY 
      else:
        raise Exception('server_type error')
    else:
      return st

  def heart_beat(self, server_type):
    server_type = self.get_server_type(server_type)
    request = HeartBeatRequest(server_type=server_type)
    resp = self.stub.HeartBeat(request)
    print(resp.addresses, flush=True)
    return resp

  def get_replicas(self, server_type, task):
    server_type = self.get_server_type(server_type)
    request = GetReplicasRequest(server_type=server_type, task=task)
    resp = self.stub.GetReplicas(request)
    print(resp.address_list.address, flush=True)
    return resp


def main(_):
  client = SvrClient(FLAGS.config)
  if FLAGS.cmd_type == 'hb':
    client.heart_beat(FLAGS.server_type)
  elif FLAGS.cmd_type == 'gr':
    client.get_replicas(FLAGS.server_type, FLAGS.task)
  else:
    raise Exception('cmd_type error')


if __name__ == "__main__":
  app.rum(main)
