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

from absl import app, flags
import os 

from monolith.agent_service import utils
from monolith.agent_service.client import FLAGS
from monolith.agent_service.utils import AgentConfig, TFSServerType
from monolith.native_training import env_utils


TFS_HOME = '/opt/tiger/monolith_serving'
PROXY_BINARY = f'{TFS_HOME}/bin/client'


def get_cmd(target, input_type, input_file):
  host, port = target.strip().split(':')
  
  if input_type == 'json':
    iw = utils.InstanceFormater.from_json(input_file)
    req_file = iw.to_pb()
  elif input_type == 'pbtext':
    iw = utils.InstanceFormater.from_pb_text(input_file)
    req_file = iw.to_pb()
  elif input_type == 'dump':
    iw = utils.InstanceFormater.from_dump(input_file)
    req_file = iw.to_pb()
  else:
    req_file = input_file
  
  assert req_file is not None
  
  cmd = f'{PROXY_BINARY} --host={host} --port={port} ' \
        f'--conf_file={TFS_HOME}/conf/client.conf ' \
        f'--log_conf={TFS_HOME}/conf/log4j.properties ' \
        f'--req_file={req_file}'
  
  return cmd 


def main(_):
  env_utils.setup_host_ip()
  cwd = os.getcwd()
  os.chdir(f'{TFS_HOME}/bin')
  config = AgentConfig.from_file(FLAGS.conf)
  if FLAGS.target is None:
    target = f'{os.environ.get("MY_HOST_IP", "localhost")}:{config.proxy_port}'
  else:
    target = FLAGS.target
  
  cmd = get_cmd(target, FLAGS.input_type, FLAGS.input_file)
  proc = subprocess.Popen(cmd.split(),
                          shell=False,
                          stdout=subprocess.PIPE,
                          env=os.environ)
  logging.info(f'pid of <{cmd}> is {proc.pid}')
  os.chdir(cwd)
  return proc

if __name__ == "__main__":
  app.run(main)
