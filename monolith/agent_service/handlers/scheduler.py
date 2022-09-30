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

import collections
import concurrent.futures
import dataclasses
import os
from typing import List
import grpc
from threading import RLock
from absl import logging

from kazoo.protocol.states import EventType as MType
from monolith.agent_service import agent_service_pb2
from monolith.agent_service import agent_service_pb2_grpc
from monolith.agent_service import resource_utils
from monolith.agent_service import utils
from monolith.agent_service.data_def import ModelMeta, PublishMeta, \
  EventType, Event, SubModelName, VersionPath, ResourceSpec
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.dispatcher import Handler


class SchedulerForTest(Handler):

  def __init__(self, zk: ZKMirror, conf: utils.AgentConfig):
    super(SchedulerForTest, self).__init__(zk)
    self.conf = conf
    self._lock = RLock()

  def event_type(self) -> EventType:
    return EventType.PORTAL

  def handle(self, event: Event):
    with self._lock:
      mm = ModelMeta.deserialize(event.data)
      paths = self._zk.get_published_path(mm.model_name)
      if mm.action == MType.DELETED:
        # for delete
        for path in paths:
          self._zk.delete(path)
        return
      elif mm.action == MType.NONE:
        # switch leader or restart
        if paths:
          pm = PublishMeta.deserialize(self._zk.get(paths[0]))
          if pm.total_publish_num == len(paths):
            # all publish alive, nothing to do
            return
          else:
            # not all publish alive, remove it and reschedule
            for path in paths:
              self._zk.delete(path)

      version, num_ps = 123456, 10
      pms = []
      # scheduler
      cnt, base = 0, int(self.conf.num_tce_shard / mm.num_shard)
      for shard_id in range(self.conf.num_tce_shard):
        if shard_id % base == 0:
          sub_models: Dict[SubModelName, VersionPath] = {
              f'ps_{k}': f'{mm.model_dir}/ps_{k}/{version}'
              for k in range(num_ps)
              if k % mm.num_shard == cnt
          }
          sub_models['entry'] = f'{mm.model_dir}/entry/{version}'

          pm = PublishMeta(shard_id=shard_id,
                           model_name=mm.model_name,
                           num_ps=10,
                           sub_models=sub_models)
          pms.append(pm)
          cnt += 1

      for pm in pms:
        pm.total_publish_num = len(pms)
      self._zk.publish_loadding(pms)


@dataclasses.dataclass
class RealtimeResource:
  """The realtime resource"""
  replica_id: int
  shard_id: int
  memory: int


def get_realtime_resources(resources: List[ResourceSpec]):

  def get_realtime_resource(resource: ResourceSpec):
    channel = grpc.insecure_channel(resource.address)
    stub = agent_service_pb2_grpc.AgentServiceStub(channel)
    req = agent_service_pb2.GetResourceRequest()
    try:
      resp = stub.GetResource(req)
    except grpc.RpcError as e:
      logging.info('fall back to reported resources for %s', resource)
      return RealtimeResource(resource.replica_id,
                              resource.shard_id,
                              memory=resource.memory)
    return RealtimeResource(resource.replica_id,
                            resource.shard_id,
                            memory=resp.memory)

  with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [
        executor.submit(get_realtime_resource, resource)
        for resource in resources
        if resource.shard_id != -1
    ]
    realtime_resources = []
    for future in futures:
      try:
        realtime_resources.append(future.result())
      except grpc.RpcError as e:
        logging.warn("Fail to get realtime resource. %s", str(e))
        raise e

  return realtime_resources


@dataclasses.dataclass
class SubModel:
  name: SubModelName
  path: str
  size: int
  # If true, this model will be loaded on each resource
  load_everywhere: bool = False


def deploy_model(model_name: str, sub_models: List[SubModel],
                 resources: List[RealtimeResource]) -> List[PublishMeta]:
  results = {}

  def add_model_to_resource(sub_model: SubModel, resource: RealtimeResource):
    resource.memory -= sub_model.size
    key = (resource.replica_id, resource.shard_id)
    if key not in results:
      # Currently this is hard-coded since we have N-1 ps and 1 entry.
      # In the future, we may remove num_ps.
      num_ps = len(sub_models) - 1
      results[key] = PublishMeta(replica_id=resource.replica_id,
                                 shard_id=resource.shard_id,
                                 model_name=model_name,
                                 num_ps=num_ps,
                                 sub_models={})
    results[key].sub_models[sub_model.name] = sub_model.path

  def check_hard_constraints(sub_model: SubModel, resource: RealtimeResource):
    return sub_model.size <= resource.memory

  for sub_model in sub_models:

    if sub_model.load_everywhere:
      for resource in resources:
        if not check_hard_constraints(sub_model, resource):
          raise ValueError(f"Unable to schedule the model. {model_name}")
        add_model_to_resource(sub_model, resource)
    else:
      best_resource = None
      for resource in resources:
        if not check_hard_constraints(sub_model, resource):
          continue
        if best_resource is None:
          best_resource = resource
          continue
        if resource.memory > best_resource.memory:
          best_resource = resource

      if best_resource is None:
        raise ValueError(f"Unable to schedule the model. {model_name}")
      add_model_to_resource(sub_model, best_resource)

  return results.values()


class Scheduler(Handler):

  def __init__(self,
               zk: ZKMirror,
               model_info_getter=resource_utils.cal_model_info_v2):
    super().__init__(zk)
    self._model_info_getter = model_info_getter
    # TODO(leqi.zou): Do we need lock here?
    self._lock = RLock()

  def event_type(self) -> EventType:
    return EventType.PORTAL

  def handle(self, event: Event):
    with self._lock:
      mm = ModelMeta.deserialize(event.data)
      paths = self._zk.get_published_path(mm.model_name)
      if mm.action == MType.DELETED:
        # for delete
        for path in paths:
          self._zk.delete(path)
        return
      elif mm.action == MType.NONE:
        # switch leader or restart
        if paths:
          pm = PublishMeta.deserialize(self._zk.get(paths[0]))
          if pm.total_publish_num == len(paths):
            # all publish alive, nothing to do
            return
          else:
            # not all publish alive, remove it and reschedule
            for path in paths:
              self._zk.delete(path)

      logging.info(f'ModelMeta is {mm}')
      logging.info(f'get_realtime_resources ...')
      resources = get_realtime_resources(self._zk.resources)

      logging.info(repr(resources))
      logging.info(f'get_realtime_resources done!')
      per_replica_resources = collections.defaultdict(list)
      for resource in resources:
        per_replica_resources[resource.replica_id].append(resource)

      logging.info(f'get_model_info ...')
      try:
        model_info = self._model_info_getter(mm.model_dir, mm.ckpt)
      except Exception as e:
        logging.info(e)
        return
      logging.info(f'get_model_info done!')

      sub_models = []
      for sub_model_name, (size, version_path) in model_info.items():
        sub_model = SubModel(sub_model_name, version_path, size)
        if sub_model_name == "entry":
          sub_model.load_everywhere = True
        sub_models.append(sub_model)

      publish_metas = []
      for resources in per_replica_resources.values():
        publish_metas.extend(deploy_model(mm.model_name, sub_models, resources))
      for pm in publish_metas:
        pm.total_publish_num = len(publish_metas)
      logging.info("Schedule result: %s", repr(publish_metas))
      self._zk.publish_loadding(publish_metas)
