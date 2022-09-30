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

from absl.testing import parameterized
import math
import tensorflow as tf
from tensorflow.python.framework.ops import name_from_scope_name

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from monolith.native_training.optimizers.rmsprop import RMSprop

_DATA_TYPES = [dtypes.half, dtypes.float32, dtypes.float64]


def rmsprop_v2_update(param, accum, g_t,
                      lr=0.01, decay=0.01, momentun=0.9, epsilon=1.0):
  grad = g_t + decay * param
  accum_t = momentun * accum + grad * grad
  param_t = param - lr * grad / (math.sqrt(accum_t) + epsilon)
  return param_t, accum_t


class RMSPropV1OptimizerTest(test.TestCase, parameterized.TestCase):
  def test_config(self):
    rmsprop = RMSprop(learning_rate=0.001,
                      initial_accumulator_value=0.1,
                      momentum=0.9,
                      weight_decay=0.01,
                      epsilon=1.0)
    config = rmsprop.get_config()
    RMSprop.from_config(config)

  def test_basic(self):
    v = tf.Variable([0.1, 0.1], name="var")
    loss = 0.12 * v
    opt = RMSprop(learning_rate=0.01,
                  initial_accumulator_value=0.1,
                  momentum=0.9,
                  weight_decay=0.01,
                  epsilon=1.0)
    update = opt.minimize(loss)
    eps = 1e-8
    with self.session() as sess:
      sess.run(tf.compat.v1.global_variables_initializer())
      sess.run(update)
      all_vars = tf.compat.v1.all_variables()
      vars_map = sess.run({var.name: var for var in all_vars})
      param_t, accum_t = rmsprop_v2_update(0.1, 0.1, 0.12)
      self.assertNear(vars_map['var:0'][0], param_t, eps)
      self.assertNear(vars_map['var/RMSpropOptimizer:0'][0], accum_t, eps)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  test.main()
