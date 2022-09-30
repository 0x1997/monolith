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

from abc import ABCMeta, abstractmethod
from absl import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, RLock
from typing import Dict, List
from queue import Empty
import time 
from monolith.agent_service.data_def import EventType, Event
from monolith.agent_service.zk_mirror import ZKMirror


class Handler(metaclass=ABCMeta):
  def __init__(self, zk: ZKMirror):
    self._zk: ZKMirror = zk
    self._is_shutdown = False

  @abstractmethod
  def event_type(self) -> EventType:
    raise NotImplementedError("event_type not implemented")

  @abstractmethod
  def handle(self, event: Event):
    raise NotImplementedError("handle not implemented")

  def cancel(self):
    self._is_shutdown = True
  
  @property
  def is_alive(self):
    return not self._is_shutdown


class Dispatcher(object):
  def __init__(self,
               zk: ZKMirror,
               max_workers: int = 5,
               update_resource_interval: int = 30):
    self._handlers: Dict[EventType, List[Handler]] = {}
    self._zk = zk
    self._max_workers = max_workers
    self._queue = zk.queue
    self._update_resource_interval = update_resource_interval

    self._is_stoped = False
    self._lock = RLock()
    self._thread = Thread(target=Dispatcher._poll, args=(self,))

  def add_handler(self, handler: Handler):
    logging.info(f"add_handler {handler} ... ")
    with self._lock:
      event_type = handler.event_type()
      if event_type in self._handlers:
        self._handlers[event_type].append(handler)
      else:
        self._handlers[event_type] = [handler]

  def start(self):
    logging.info('start dispatcher ...')
    self._thread.start()
    logging.info('start dispatcher done!')

  def stop(self):
    self._is_stoped = True
    logging.info('stop handlers ...')
    for handlers in self._handlers.values():
      for handler in handlers:
        logging.info(f'cancel handler {handler}')
        handler.cancel()
    logging.info('stop handlers done!')
    
    logging.info('stop dispatcher ...')
    self._thread.join()
    logging.info('stop dispatcher done!')

  def _poll(self):
    with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
      while not self._is_stoped:
        try:
          event = self._queue.get(block=True, timeout=1)
        except Empty as e:
          continue
        with self._lock:
          etypes = set(self._handlers.keys())
          if event.etype in etypes:
            handlers = self._handlers[event.etype]
          else:
            handlers = []
        
        # the event have no handler, put the event back
        if handlers is None or len(handlers) == 0:
          self._queue.put(event)

        for handler in handlers:
          def fn(vnt: Event):
            handler.handle(vnt)
            return True

          pool.submit(fn, vnt=event)

