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

from absl import app, flags, logging
import os
import time
import signal
from queue import Queue
from subprocess import Popen, STDOUT
from monolith.native_training.zk_utils import MonolithKazooClient
from monolith.agent_service.mocked_zkclient import FakeKazooClient
from monolith.agent_service.utils import AgentConfig, TFSServerType, DeployType
from monolith.agent_service.zk_mirror import ZKMirror
from monolith.agent_service.leader import Leader
from monolith.agent_service.tfs_monitor import TFSMonitor
from monolith.agent_service.dispatcher import Dispatcher
from monolith.agent_service.handlers.scheduler import Scheduler
from monolith.agent_service.handlers.resource_report import ResourceReportHandler
from monolith.agent_service.handlers.model_loader import ModelLoaderHandler
from monolith.agent_service.handlers.status_report import StatusReportHandler
from monolith.agent_service.agent_service import AgentService
from monolith.agent_service.agent_base import AgentBase, ServingLog, get_cmd_and_port, \
    TFS_HOME, TFS_BINARY, PROXY_BINARY


class AgentV2(AgentBase):
  def __init__(self, config: AgentConfig, conf_path:str, tfs_log: str):
    super(AgentV2, self).__init__(config)
    self._conf_path = conf_path
    self._tfs_log = tfs_log

    self._stop: bool = False
    self._process = []
    signal.signal(signal.SIGTERM, self.signal_handler)
    signal.signal(signal.SIGINT, self.signal_handler)

    if config.zk_servers is None:
      _zk = FakeKazooClient(zk_server=config.zk_servers)
    else:
      _zk = MonolithKazooClient(hosts=config.zk_servers)
    
    self.zk = ZKMirror(
      zk=_zk,
      bzid=config.bzid,
      queue=Queue(),
      tce_shard_id=config.shard_id,
      num_tce_shard=config.num_tce_shard or 1,
      deploy_type=config.deploy_type)

    self.tfs_monitor = TFSMonitor(config=config)
    self.dispatcher = Dispatcher(self.zk)
    resource_reporter = ResourceReportHandler(
      self.zk, agent_address=f'{self.tfs_monitor.host}:{config.agent_port}')
    self.dispatcher.add_handler(resource_reporter)

    resource_reporter = StatusReportHandler(self.zk, self.tfs_monitor)
    self.dispatcher.add_handler(resource_reporter)

    model_loader = ModelLoaderHandler(self.zk, self.tfs_monitor)
    self.dispatcher.add_handler(model_loader)

    self.agent_service = AgentService(zk=self.zk, conf=config)

  def start_tfs(self):
    if self.config.deploy_type in {DeployType.MIXED, DeployType.ENTRY}:
      cmd, port = get_cmd_and_port(config=self.config,
                                  server_type=TFSServerType.ENTRY,
                                  config_file=self.config.model_config_file)
      with ServingLog('entry', self._tfs_log) as log_stdout:
        popen = Popen(cmd.split(),
                      shell=False,
                      stderr=STDOUT,
                      stdout=log_stdout,
                      env=os.environ)
        self._process.append(popen)
        logging.info(f'start_tfs entry cmd: {cmd}')
        logging.info(f'start entry at {self.tfs_monitor.host}:{port}')

    if self.config.deploy_type in {DeployType.MIXED, DeployType.PS}:
      cmd, port = get_cmd_and_port(config=self.config,
                                  server_type=TFSServerType.PS,
                                  config_file=self.config.model_config_file)
      with ServingLog('ps', self._tfs_log) as log_stdout:
        popen = Popen(cmd.split(),
                      shell=False,
                      stderr=STDOUT,
                      stdout=log_stdout,
                      env=os.environ)
        self._process.append(popen)
        logging.info(f'start_tfs ps cmd: {cmd}')
        logging.info(f'start ps at {self.tfs_monitor.host}:{port}')

  def start_proxy(self):
    if self.config.deploy_type == DeployType.PS:
      return
    
    cmd, port = get_cmd_and_port(config=self.config, conf_path=self._conf_path)
    with ServingLog('proxy', self._tfs_log) as log_stdout:
      popen = Popen(cmd.split(),
                    shell=False,
                    stderr=STDOUT,
                    stdout=log_stdout,
                    env=os.environ)
      self._process.append(popen)
      logging.info(f'start proxy cmd: {cmd}')
      logging.info(f'start ps at {self.tfs_monitor.host}:{port}')

  def signal_handler(self, signum, frame):
    logging.info(f"catch signal {signum}, frame {frame}")
    self._stop = True

  def start(self):
    self.start_tfs()
    self.start_proxy()

    self.zk.start()
    scheduler = Scheduler(self.zk)
    leader = Leader(self.dispatcher)
    self.zk.election(leader, sched=scheduler)

    self.tfs_monitor.connect()
    self.dispatcher.start()
    self.agent_service.start()

  def stop(self):
    try:
      self.agent_service.stop(True)
      self.dispatcher.stop()
      self.tfs_monitor.stop()
      self.zk.stop()
    except Exception as e:
      logging.warning(e)
    finally:
      for proc in self._process:
        try:
          if proc is not None and proc.stdout is not None:
            proc.stdout.close()
        except Exception as e:
          logging.info(e)
        finally:
          proc.kill()
  
  def wait_for_termination(self):
    while not self._stop:
      time.sleep(1)
      for proc in self._process:
        proc.poll()
        if proc.returncode is not None:
          self._stop = True
    
    self.stop()
    time.sleep(1) 
    os.kill(os.getpid(), signal.SIGKILL)

