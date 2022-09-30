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

import contextlib

from typing import NamedTuple
from monolith.agent_service.backends import SyncBackend


class NativeTaskContext(NamedTuple):
  num_ps: int
  num_workers: int
  worker_index: int
  # Model name is used to uniquely identify a model
  # It will influence how we export models and do the serving.
  model_name: str
  sync_backend: SyncBackend


_CTX = NativeTaskContext(num_ps=0, num_workers=1, worker_index=0, model_name="", sync_backend=None)


@contextlib.contextmanager
def with_ctx(ctx: NativeTaskContext):
  global _CTX
  old_ctx = _CTX
  _CTX = ctx
  try:
    yield
  finally:
    _CTX = old_ctx


def get():
  return _CTX
