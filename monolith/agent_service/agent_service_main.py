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

from absl import logging
import grpc
from kazoo.exceptions import NoNodeError, NodeExistsError
import os
import socket
import unittest

from monolith.agent_service import utils
from monolith.agent_service.agent_service import AgentService
from monolith.agent_service.agent_service_pb2 import HeartBeatRequest, ServerType, \
  GetReplicasRequest
from monolith.agent_service.agent_service_pb2_grpc import AgentServiceStub
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.replica_manager import ReplicaWatcher, ReplicaMeta, ModelState


MODEL_NAME = 'hupu_recall'
BASE_PATH = f'/hupu/{MODEL_NAME}/saved_models'
NUM_PS_REPLICAS = 1
NUM_ENTRY_REPLICAS = 1

def register(zk):
  path_prefix = f'/hupu/service/hupu_recall'
  path_to_meta = {}
  for task_id in range(3):
    meta = ReplicaMeta(address=f'127.0.0.1:8502', stat=ModelState.AVAILABLE)
    replica_path = f'{path_prefix}/ps:{task_id}/0'
    path_to_meta[replica_path] = meta

  
  replica_path = f'{path_prefix}/entry:0/0'
  meta = ReplicaMeta(address=f'127.0.0.1:8710', stat=ModelState.AVAILABLE)
  path_to_meta[replica_path] = meta
  
  for replica_path, meta in path_to_meta.items():
    replica_meta_bytes = bytes(meta.to_json(), encoding='utf-8')
    
    try:
      print("replica_path: ", replica_path)
      zk.retry(zk.create,
               path=replica_path, value=replica_meta_bytes, ephemeral=True, makepath=True)
    except NodeExistsError:
      logging.info(f'{replica_path} has already exists')
      zk.retry(zk.set, path=replica_path, value=replica_meta_bytes)

if __name__ == "__main__":
  zk = FakeKazooClient()
  zk.start()
  agent_conf: utils.AgentConfig = utils.AgentConfig(
      bzid='hupu', base_name=MODEL_NAME, deploy_type='ps', 
      replica_id=0, base_path=BASE_PATH, num_ps=1, num_shard=1)
  watcher = ReplicaWatcher(zk, agent_conf)
  watcher.watch_data()
  agent = AgentService(watcher, port=9899)
  register(zk)
  agent.start()
  agent.wait_for_termination()
