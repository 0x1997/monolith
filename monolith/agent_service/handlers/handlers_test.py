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
from monolith.agent_service.mocked_tfserving import FakeTFServing
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.tfs_monitor import TFSMonitor
from monolith.agent_service.handlers.scheduler import SchedulerForTest
from monolith.agent_service.handlers.model_loader import ModelLoaderHandler
from monolith.agent_service.handlers.resource_report import ResourceReportHandler
from monolith.agent_service.handlers.status_report import StatusReportHandler
from monolith.agent_service.dispatcher import Dispatcher, Handler
from monolith.agent_service.data_def import EventType, Event, ModelMeta, PublishMeta
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig

MODEL_NAME = 'iptv_ctr'
BASE_PATH = f'/iptv/{MODEL_NAME}/saved_models'
NUM_REPLICAS = 3


class HandlersTest(unittest.TestCase):  
  @classmethod
  def setUpClass(cls) -> None:
    os.environ['byterec_host_shard_n'] = '10'
    os.environ['SHARD_ID'] = '2'
    os.environ['REPLICA_ID'] = '2'
    cls.bzid = 'iptv'
    agent_conf = utils.AgentConfig(bzid='iptv', deploy_type='mixed')
    
    cls.zk = ZKMirror(zk=FakeKazooClient(),
                      bzid=cls.bzid,
                      queue = Queue(),
                      tce_shard_id=agent_conf.shard_id, 
                      num_tce_shard=agent_conf.num_tce_shard)
    cls.zk.start()

    cls.tfs_entry = FakeTFServing(num_versions=2, port=agent_conf.tfs_entry_port,
                                  model_config_file=ModelServerConfig())
    cls.tfs_ps = FakeTFServing(num_versions=2, port=agent_conf.tfs_ps_port,
                               model_config_file=ModelServerConfig())
    
    entry_thread = threading.Thread(target=lambda: cls.tfs_entry.start())
    entry_thread.start()
    ps_thread = threading.Thread(target=lambda: cls.tfs_ps.start())
    ps_thread.start()

    time.sleep(1)
    cls.monitor = TFSMonitor(agent_conf)
    cls.monitor.connect()

    cls.dispatcher = Dispatcher(cls.zk)
    cls.dispatcher.start()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.dispatcher.stop()
    cls.monitor.stop()
    cls.tfs_entry.stop()
    cls.tfs_ps.stop()
    cls.zk.stop()

  def test_handlers(self):
    # test_step1_init
    scheduler = SchedulerForTest(self.zk, self.monitor._conf)
    model_loader = ModelLoaderHandler(self.zk, self.monitor)
    agent_port = self.monitor._conf.agent_port
    resource_report = ResourceReportHandler(self.zk, f'{self.monitor.host}:{agent_port}')
    status_report = StatusReportHandler(self.zk, tfs=self.monitor, interval=1)

    self.zk.watch_portal()
    self.zk.watch_resource()
    self.zk.set_leader()
    self.dispatcher.add_handler(scheduler)
    self.dispatcher.add_handler(model_loader)
    self.dispatcher.add_handler(resource_report)
    self.dispatcher.add_handler(status_report)

    # 2) test_step2_request_loadding
    path = os.path.join(self.zk.portal_base_path, MODEL_NAME)
    mm = ModelMeta(
      model_name=MODEL_NAME,
      model_dir=BASE_PATH,
      num_shard=5)
    self.zk.create(path, mm.serialize())
  
    # 3) test_step3_get_replica
    curr_sleep, max_sleep = 0, 10
    while True:
      time.sleep(1)
      curr_sleep += 1
      if curr_sleep >= max_sleep:
        break
      
      # because only all ps entry 
      replicas = self.zk.get_all_replicas(server_type='ps')
      if len(replicas) == 2:
        self.assertTrue('iptv_ctr:ps:1' in replicas)
        self.assertTrue('iptv_ctr:ps:6' in replicas)
        break
    
    # 4) test_step4_resource
    curr_sleep, max_sleep = 0, 10
    while True:
      time.sleep(1)
      curr_sleep += 1
      if curr_sleep >= max_sleep:
        break

      resources = self.zk.resources
      if len(resources) > 0:
        address = f'{self.monitor.host}:{self.monitor._conf.agent_port}'
        self.assertEqual(resources[0].address, address)
        break


if __name__ == "__main__":
  unittest.main()
