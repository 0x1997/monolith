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

import time
from absl import logging
from typing import Dict
from monolith.agent_service.data_def import EventType, Event, PublishMeta
from monolith.agent_service.data_def import ModelName
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.tfs_monitor import TFSMonitor
from monolith.agent_service.dispatcher import Handler
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig
from monolith.agent_service.utils import TFSServerType, DeployType


class PSNotReady(Exception):
  def __init__(self, message):
    super(PSNotReady, self).__init__(message)
    self.message = message


class ModelLoaderHandler(Handler):
  def __init__(self,
               zk: ZKMirror,
               tfs: TFSMonitor):
    super(ModelLoaderHandler, self).__init__(zk)
    self._tfs: TFSMonitor = tfs

  def event_type(self) -> EventType:
    return EventType.PUBLISH

  def handle(self, event: Event):
    if not self.is_alive:
      return

    zk: ZKMirror = self._zk
    # 1) get model_status from tfs
    expected_loading: Dict[ModelName, PublishMeta] = zk.expected_loading()    
    logging.info(f'expected_loading: {expected_loading}')
    server_type_to_model_config: Dict[str, ModelServerConfig] = self._tfs.gen_model_config(expected_loading.values())
    logging.info(f'server_type_to_model_config: {server_type_to_model_config}')

    ps_model_configs = server_type_to_model_config[TFSServerType.PS]
    if self._tfs._conf.deploy_type in {DeployType.PS, DeployType.MIXED} and self.is_alive:
      try:
        status = self._tfs.handle_reload_config_request(TFSServerType.PS, ps_model_configs)
        logging.info(status)
      except Exception as e:
        time.sleep(5)
        # in case tfs not start, try again
        self._zk.queue.put(event)
        raise e

    entry_model_configs = server_type_to_model_config[TFSServerType.ENTRY]
    while self._tfs._conf.deploy_type in {DeployType.ENTRY, DeployType.MIXED} and self.is_alive:
      try:
        for model_config in entry_model_configs.model_config_list.config:
          if not self.is_alive:  # fast stop
            break
          model_name, sub_model_name = model_config.name.strip().split(':')
          pm: PublishMeta = expected_loading[model_name]
          assert sub_model_name.startswith(TFSServerType.ENTRY)
          # f'{deploy_type}:{task_id}:{replica}' -> ReplicaMeta
          all_replicas = zk.get_all_replicas(TFSServerType.PS)
          logging.info(repr(all_replicas))
          pses = {':'.join(key.strip().split(':')) for key in all_replicas 
                  if key.startswith(f'{model_name}:')}
          if len(pses) != pm.num_ps:
            raise PSNotReady(f'total {pm.num_ps} ps, only [{",".join(pses)}] ready')
        break
      except PSNotReady as e:
        logging.warning(e.message)
        time.sleep(5)

    if self._tfs._conf.deploy_type in {DeployType.ENTRY, DeployType.MIXED}  and self.is_alive:
      try:
        status = self._tfs.handle_reload_config_request(TFSServerType.ENTRY, entry_model_configs)
        logging.info(status)
      except Exception as e:
        time.sleep(5)
        # in case tfs not start, try again
        self._zk.queue.put(event)
        raise e
