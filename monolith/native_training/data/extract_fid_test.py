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

import tensorflow as tf
from tensorflow.python.framework import load_library

from monolith.utils import get_libops_path

ragged_data_ops = load_library.load_op_library(
    get_libops_path('monolith/native_training/data/pb_data_ops.so'))


class ExtraFidTest(tf.test.TestCase):

  def test_parse_search(self):
    fid = ragged_data_ops.extract_fid(185, 4).numpy()
    self.assertTrue(fid == 1153447759131936)
    

if __name__ == "__main__":
  tf.test.main()
