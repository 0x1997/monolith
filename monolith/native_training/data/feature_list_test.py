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

from absl import flags
from typing import get_type_hints
from enum import Enum
import dataclasses
import threading

import tensorflow as tf

from monolith.native_training.data.feature_list import FeatureList


class FeatureListTest(tf.test.TestCase):

  def test_parse_cmcc(self):
    cmcc = FeatureList.parse(
        'monolith/native_training/data/test_data/cmcc.conf')
    self.assertTrue(len(cmcc) == 263)
    self.assertTrue('album_id-f_bhv_time_weekday' in cmcc)
    self.assertTrue(cmcc['f_album_id-f_bhv_time_weekday'].slot == 490)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()
