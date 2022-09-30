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
import os 
from queue import Queue
from kazoo.exceptions import NodeExistsError
import socket
import time
import threading

import unittest

from monolith.agent_service import utils
from monolith.agent_service.agent_service_pb2 import ServerType
from monolith.agent_service.mocked_tfserving import FakeTFServing
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.data_def import PublishMeta, PublishType, ReplicaMeta, ResourceSpec, \
  SubModelName, VersionPath, ModelMeta, EventType, Event
from monolith.agent_service.dispatcher import Dispatcher, Handler


MODEL_NAME = 'iptv_ctr'
BASE_PATH = f'/iptv/{MODEL_NAME}/saved_models'
NUM_REPLICAS = 3


class ResourceHandler(Handler):
  def __init__(self, zk: ZKMirror):
    super(ResourceHandler, self).__init__(zk)
    self.event = None

  def event_type(self) -> EventType:
    return EventType.RESOURCE

  def handle(self, event: Event):
    self.event = event
    print(event, flush=True)


class ZKMirrorTest(unittest.TestCase):
  tfs: FakeTFServing = None
  agent_conf: utils.AgentConfig = None

  @classmethod
  def setUpClass(cls) -> None:
    cls.zk = ZKMirror(zk=FakeKazooClient(),
                      bzid='iptv',
                      queue = Queue(1024),
                      tce_shard_id=2, 
                      num_tce_shard=10)
    cls.zk.start()

    cls.dispatcher = Dispatcher(cls.zk)
    cls.dispatcher.start()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.dispatcher.stop()
    cls.zk.stop()

  def test_dispatcher(self):
    handler = ResourceHandler(self.zk)
    self.dispatcher.add_handler(handler)
    event = Event(path='RESOURCE', etype=EventType.RESOURCE)
    self.zk.queue.put(event)
    if handler.event is not None:
      self.assertEqual(handler.event, event)
    

if __name__ == "__main__":
  unittest.main()
