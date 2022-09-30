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

import os
import unittest
from unittest import mock

from monolith.native_training import env_utils


class EnvUtilsTest(unittest.TestCase):

  @mock.patch.dict(os.environ, {"PYTHONPATH": "123"})
  def test_reset_to_default_py_env(self):
    with env_utils.reset_to_default_py_env():
      self.assertIn("pyutil", os.environ["PYTHONPATH"])
    self.assertEqual(os.environ["PYTHONPATH"], "123")


if __name__ == "__main__":
  unittest.main()
