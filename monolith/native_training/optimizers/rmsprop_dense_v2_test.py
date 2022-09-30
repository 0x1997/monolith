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
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from monolith.native_training.optimizers.rmsprop import RMSpropV2 as RMSprop

_DATA_TYPES = [dtypes.half, dtypes.float32, dtypes.float64]


def rmsprop_v2_update_numpy(param, accum, g_t,
                            lr=0.001, decay=0.01, momentun=0.9, epsilon=1.0):
  grad = g_t + decay * param
  accum_t = momentun * accum + grad * grad
  param_t = param - lr * grad / (np.sqrt(accum_t) + epsilon)
  return param_t, accum_t


class RMSPropV2OptimizerTest(test.TestCase, parameterized.TestCase):
  def doTestBasic(self, use_callable_params=False):
    for dtype in _DATA_TYPES:
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)

      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

      var0 = variables.Variable(var0_np)
      var1 = variables.Variable(var1_np)
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)

      learning_rate = lambda: 3.0
      if not use_callable_params:
        learning_rate = learning_rate()

      rmsprop_v2 = RMSprop(learning_rate,
                           initial_accumulator_value=0.1,
                           momentum=0.9,
                           weight_decay=0.01,
                           epsilon=1.0)

      accum0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      accum1_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)

      if not context.executing_eagerly():
        ada_update = rmsprop_v2.apply_gradients(
          zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

      # Fetch params to validate initial values
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

      # Run 3 steps of adagrad
      for _ in range(3):
        if not context.executing_eagerly():
          self.evaluate(ada_update)
        else:
          rmsprop_v2.apply_gradients(zip([grads0, grads1], [var0, var1]))
        var0_np, accum0_np = rmsprop_v2_update_numpy(var0_np, accum0_np, grads0_np,
                                                     lr=3.0, decay=0.01)
        var1_np, accum1_np = rmsprop_v2_update_numpy(var1_np, accum1_np, grads1_np,
                                                     lr=3.0, decay=0.01)
        self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
        self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testBasic(self):
    self.doTestBasic()

  def test_config(self):
    rmsprop_v2 = RMSprop(learning_rate=0.001,
                         initial_accumulator_value=0.1,
                         momentum=0.9,
                         weight_decay=0.01,
                         epsilon=1.0)
    config = rmsprop_v2.get_config()
    RMSprop.from_config(config)


if __name__ == "__main__":
  test.main()
