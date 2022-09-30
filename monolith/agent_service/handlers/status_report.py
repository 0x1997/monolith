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
import time
from threading import Thread
from typing import Dict
from monolith.agent_service.data_def import EventType, Event, \
  PublishMeta, ModelVersionStatus, ReplicaMeta, ModelName, TFSModelName, VersionPath
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.tfs_monitor import TFSMonitor
from monolith.agent_service.utils import pasre_sub_model_name
from monolith.agent_service.dispatcher import Handler


class StatusReportHandler(Handler):
  def __init__(self,
               zk: ZKMirror,
               tfs: TFSMonitor,
               interval: int = 5):
    super(StatusReportHandler, self).__init__(zk)
    self._tfs = tfs

    def target():
      while self.is_alive:
        event = Event(path='', etype=EventType.SERVICE)
        self._zk.queue.put(event)
        time.sleep(interval)

    reporter = Thread(target=target)
    reporter.start()

  def event_type(self) -> EventType:
    return EventType.SERVICE

  def handle(self, event: Event):
    if not self.is_alive:
      return
    assert self.event_type() == event.etype

    zk: ZKMirror = self._zk
    # 1) get model_status from tfs
    expected_loading: Dict[ModelName, PublishMeta] = zk.expected_loading()
    need_load: Dict[TFSModelName, (VersionPath, ModelVersionStatus)] = {}
    if expected_loading is None:
      return
    elif isinstance(expected_loading, PublishMeta):
      # Dict[TFSModelName, (VersionPath, ModelVersionStatus)]
      model_status = self._tfs.get_model_status(expected_loading)
      if model_status is not None and len(model_status) > 0:
        need_load.update(model_status)
    else:
      for model_name, pm in expected_loading.items():
        # Dict[TFSModelName, (VersionPath, ModelVersionStatus)]
        model_status = self._tfs.get_model_status(pm)
        if model_status is not None and len(model_status) > 0:
          need_load.update(model_status)

    # 2) compose ReplicaMeta
    replica_metas = []
    for tfs_model_name, (vp, stat) in need_load.items():
      model_name, sub_model_name = tfs_model_name.strip().split(':')
      server_type, task = pasre_sub_model_name(sub_model_name)
      rm = ReplicaMeta(
        address=self._tfs.get_addr(sub_model_name),
        model_name=model_name,
        server_type=server_type,
        task=task,
        replica=zk.tce_replica_id,
        stat=stat.state)
      replica_metas.append(rm)

    # 3) update zk status
    if self.is_alive:
      # update_service even if replica_metas is empty
      zk.update_service(replica_metas)
