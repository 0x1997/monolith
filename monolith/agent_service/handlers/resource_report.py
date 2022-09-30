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
import time
import os
from threading import Thread
from monolith.agent_service.data_def import EventType, Event, ResourceSpec
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.resource_utils import cal_available_memory_v2
from monolith.agent_service.dispatcher import Handler


class ResourceReportHandler(Handler):
  def __init__(self,
               zk: ZKMirror,
               agent_address: str,
               interval: int = 5):
    super(ResourceReportHandler, self).__init__(zk)
    self._agent_address = agent_address

    def target():
      while self.is_alive:
        event = Event(path='', etype=EventType.RESOURCE)
        self._zk.queue.put(event)
        if interval == 0:
          break
        else:
          time.sleep(interval)

    reporter = Thread(target=target)
    reporter.start()

  def event_type(self) -> EventType:
    return EventType.RESOURCE

  def handle(self, event: Event):
    if not self.is_alive:
      return
    assert self.event_type() == event.etype

    try:
      memory = cal_available_memory_v2()
      resource = ResourceSpec(
        address=self._agent_address,
        shard_id=self._zk.tce_shard_id,
        replica_id=self._zk.tce_replica_id,
        memory=memory)

      self._zk.report_resource(resource)
    except Exception as e:
      logging.info(e)
