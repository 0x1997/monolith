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
from kazoo.exceptions import NodeExistsError
from queue import Queue
import os
import socket
import time
import threading
import unittest

from monolith.agent_service import utils
from monolith.agent_service.data_def import EventType, Event, ModelMeta, PublishMeta
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig
from monolith.agent_service.agent_v2 import AgentV2


MODEL_NAME = 'iptv_ctr'
BASE_PATH = f'/iptv/{MODEL_NAME}/saved_models'
NUM_REPLICAS = 3


def start_tfs(self):
  tfs_entry = FakeTFServing(num_versions=2, port=self.config.tfs_entry_port,
                            model_config_file=ModelServerConfig())
  tfs_ps = FakeTFServing(num_versions=2, port=self.config.tfs_ps_port,
                          model_config_file=ModelServerConfig())
  entry_thread = threading.Thread(target=lambda: tfs_entry.start())
  entry_thread.start()
  ps_thread = threading.Thread(target=lambda: tfs_ps.start())
  ps_thread.start()


def start_proxy(self):
  pass


class AgentV2Test(unittest.TestCase):  
  @classmethod
  def setUpClass(cls) -> None:
    os.environ['byterec_host_shard_n'] = '10'
    os.environ['SHARD_ID'] = '2'
    os.environ['REPLICA_ID'] = '2'
    cls.bzid = 'iptv'
    agent_conf = utils.AgentConfig(
      bzid='iptv', deploy_type='mixed', agent_version=2)

    cls.agent = AgentV2(config=agent_conf, 
                        conf_path='/opt/tiger/monolith_serving/conf', 
                        tfs_log='/opt/tiger/monolith_serving/logs/log.log')
    cls.agent.start()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.agent.stop()
    print('tearDownClass finished!')
  
  def test_step1_request_loadding(self):
    path = os.path.join(self.agent.zk.portal_base_path, MODEL_NAME)
    mm = ModelMeta(
      model_name=MODEL_NAME,
      model_dir=BASE_PATH,
      num_shard=5)
    self.agent.zk.create(path, mm.serialize())
  
  def test_step2_get_replica(self):
    curr_sleep, max_sleep = 0, 10
    while True:
      time.sleep(1)
      curr_sleep += 1
      if curr_sleep >= max_sleep:
        break
      
      # because only all ps entry 
      replicas = self.agent.zk.get_all_replicas(server_type='ps')
      if len(replicas) == 2:
        self.assertTrue('iptv_ctr:ps:1' in replicas)
        self.assertTrue('iptv_ctr:ps:6' in replicas)
        break
    
  def test_step3_resource(self):
    curr_sleep, max_sleep = 0, 10
    while True:
      time.sleep(1)
      curr_sleep += 1
      if curr_sleep >= max_sleep:
        break

      resources = self.agent.zk.resources
      if len(resources) > 0:
        address = f'{self.agent.tfs_monitor.host}:{self.agent.tfs_monitor._conf.agent_port}'
        self.assertEqual(resources[0].address, address)
        break


if __name__ == "__main__":
  logging.use_absl_handler()
  logging.get_absl_handler().setFormatter(fmt=logging.PythonFormatter())
  logging.set_verbosity(logging.INFO)
  
  unittest.main()
