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
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.dispatcher import Dispatcher
from monolith.agent_service.handlers.scheduler import Scheduler


class Leader(object):
  def __init__(self, disp: Dispatcher):
    self._disp = disp
    self._is_shutdown = False

  def __call__(self, zk: ZKMirror, sched: Scheduler):
    if not self._is_shutdown:
      zk.set_leader()
      zk.watch_resource()
      zk.watch_portal()
      self._disp.add_handler(sched)

    while not self._is_shutdown:
      time.sleep(1)

  def cancel(self):
    self._is_shutdown = True
