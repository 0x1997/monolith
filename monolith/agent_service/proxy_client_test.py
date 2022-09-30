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
import json
import os
import tempfile
import unittest

from monolith.agent_service.proxy_client import get_cmd

prefix = '/opt/tiger/monolith_serving/bin/client --host=local --port=1234 --conf_file=/opt/tiger/monolith_serving/conf/client.conf --log_conf=/opt/tiger/monolith_serving/conf/log4j.properties --req_file='

class TFSClientTest(unittest.TestCase):
  
  def test_get_cmd_json(self):
    cmd = get_cmd('local:1234', 'json', 'monolith/agent_service/test_data/inst.json')
    self.assertTrue(cmd.startswith(prefix))
    
  def test_get_cmd_pbtext(self):
    cmd = get_cmd('local:1234', 'pbtext', 'monolith/agent_service/test_data/inst.pbtext')
    self.assertTrue(cmd.startswith(prefix))
    
  def test_get_cmd_dump(self):
    cmd = get_cmd('local:1234', 'dump', 'monolith/agent_service/test_data/inst.dump')
    self.assertTrue(cmd.startswith(prefix))

if __name__ == "__main__":
  unittest.main()
