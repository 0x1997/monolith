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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from typing import Union, Callable
import tensorflow as tf

import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.initializers import Constant

from tensorflow.python.framework import load_library

from monolith.utils import get_libops_path

training_ops = load_library.load_op_library(
    get_libops_path("monolith/native_training/optimizers/training_ops.so"))


class RmspropOptimizer(tf.compat.v1.train.Optimizer):
  """http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf"""

  def __init__(self,
               learning_rate=5e-6,
               beta1: float = 0.99,
               beta2: float = 0.999,
               epsilon: float = 1e-8,
               weight_decay: float = 0.0,
               use_locking: bool = False,
               use_v2: bool = False,
               name="Rmsprop"):
    super().__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon
    self._weight_decay = weight_decay
    self._use_v2 = use_v2
    # Created in Initialize.
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name + "/m")
      self._zeros_slot(v, "v", self._name + "/v")

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = tf.convert_to_tensor(learning_rate,
                                                      name="learning_rate")

  def _apply_dense(self, grad, var):
    raise NotImplementedError(
        "Please use tf.compat.v1.disable_eager_execution() instead of tf.compat.v1.disable_v2_behavior()"
    )

  def _resource_apply_dense(self, grad, var):
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    return training_ops.resource_apply_rmsprop(var.handle,
                                               m.handle,
                                               v.handle,
                                               tf.cast(
                                                   self._learning_rate_tensor,
                                                   grad.dtype.base_dtype),
                                               self._beta1,
                                               self._beta2,
                                               self._epsilon,
                                               self._weight_decay,
                                               grad,
                                               use_locking=self._use_locking,
                                               use_v2=self._use_v2)


class RMSprop(tf.compat.v1.train.Optimizer):

  def __init__(self,
               learning_rate: Union[float, Callable[[], float]],
               momentum: float = 0.0,
               weight_decay: float = 0.0,
               initial_accumulator_value: float = 0.1,
               epsilon: float = 1.0,
               use_v2: bool = True,
               use_locking: bool = False,
               name: str = "RMSpropOptimizer"):
    if initial_accumulator_value < 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(RMSprop, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._weight_decay = weight_decay
    self._initial_accumulator_value = initial_accumulator_value
    self._epsilon = epsilon
    self._use_v2 = use_v2

    # Created in Initialize.
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      dtype = v.dtype.base_dtype
      if v.get_shape().is_fully_defined():
        init = init_ops.constant_initializer(self._initial_accumulator_value,
                                             dtype=dtype)
      else:
        init = self._init_constant_op(v, dtype)
      self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                              "accumulator", self._name)

  def _init_constant_op(self, v, dtype):

    def init():
      # Use a Tensor instead of initializer if variable does not have
      # static shape.
      init_constant = gen_array_ops.fill(array_ops.shape(v),
                                         self._initial_accumulator_value)
      return math_ops.cast(init_constant, dtype)

    return init

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(learning_rate,
                                                       name="learning_rate")

  def _apply_dense(self, grad, var):
    grad2_acc = self.get_slot(var, "accumulator")
    grad_with_decay = grad + self._weight_decay * var

    if self._momentum:
      if self._use_v2:
        grad2_acc_t = grad2_acc * self._momentum + grad_with_decay * grad_with_decay
      else:
        grad2_acc_t = grad2_acc * self._momentum + grad_with_decay * grad_with_decay * (
            1 - self._momentum)
    else:
      # AdaGrad
      grad2_acc_t = grad2_acc + grad_with_decay * grad_with_decay

    lr_t = self._learning_rate_tensor
    grad2_acc_t = state_ops.assign(grad2_acc,
                                   grad2_acc_t,
                                   use_locking=self._use_locking)
    if self._momentum:
      var_t = var - tf.multiply(
          grad_with_decay, lr_t) / (math_ops.sqrt(grad2_acc_t) + self._epsilon)
    else:
      var_t = var - tf.multiply(
          grad_with_decay, lr_t) / math_ops.sqrt(grad2_acc_t + self._epsilon)
    return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var):
    return self._apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    raise NotImplementedError('_apply_sparse is not implemented')

  def _resource_apply_sparse(self, grad, var, indices):
    raise NotImplementedError('_resource_apply_sparse is not implemented')

  def get_config(self):
    return {
        "learning_rate": self._call_if_callable(self._learning_rate),
        "momentum": self._momentum,
        "weight_decay": self._weight_decay,
        "initial_accumulator_value": self._initial_accumulator_value,
        "epsilon": self._epsilon,
        "use_v2": self._use_v2,
        'use_locking': self._use_locking,
        "name": self._name
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'initial_accumulator_value' not in config:
      config['initial_accumulator_value'] = 0.0
    if 'lr' in config:
      config['learning_rate'] = config.pop('lr')
    return cls(**config)


class RMSpropV2(optimizer_v2.OptimizerV2):
  """RMPPropV2 optimizer: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

  grad = grad + weight_decay * weight
  grad_acc = momentum * grad_acc + grad * grad
  weight = weight - grad * lr / (std::sqrt(grad_acc) + 1)
  """

  _HAS_AGGREGATE_GRAD = True

  def __init__(self,
               learning_rate=0.001,
               momentum=0.0,
               weight_decay=0.0,
               initial_accumulator_value=0.0,
               epsilon=1.0,
               use_v2=True,
               name="RMSpropV2",
               **kwargs):
    super(RMSpropV2, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("weight_decay", weight_decay)
    self._set_hyper("decay", self._initial_decay)

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)
    self._epsilon = epsilon
    self._initial_accumulator_value = initial_accumulator_value
    self._use_v2 = use_v2

  def _create_slots(self, var_list):
    for var in var_list:
      self.add_slot(var,
                    "grad2_acc",
                    initializer=Constant(self._initial_accumulator_value))

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(RMSpropV2, self)._prepare_local(var_device, var_dtype, apply_state)

    weight_decay = array_ops.identity(self._get_hyper("weight_decay",
                                                      var_dtype))
    momentum = array_ops.identity(self._get_hyper("momentum", var_dtype))
    epsilon = ops.convert_to_tensor_v2_with_dispatch(self._epsilon, var_dtype)
    apply_state[(var_device, var_dtype)].update(
        dict(weight_decay=weight_decay, momentum=momentum, epsilon=epsilon))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    weight_decay = coefficients["weight_decay"]
    grad_with_decay = grad + weight_decay * var

    grad2_acc = self.get_slot(var, "grad2_acc")
    if self._momentum:
      momentum = coefficients["momentum"]
      if self._use_v2:
        grad2_acc_t = grad2_acc * momentum + grad_with_decay * grad_with_decay
      else:
        grad2_acc_t = grad2_acc * momentum + grad_with_decay * grad_with_decay * (
            1 - momentum)
    else:
      # AdaGrad
      grad2_acc_t = grad2_acc + grad_with_decay * grad_with_decay
    grad2_acc_t = state_ops.assign(grad2_acc,
                                   grad2_acc_t,
                                   use_locking=self._use_locking)

    lr_t = coefficients["lr_t"]
    epsilon = coefficients["epsilon"]
    if self._momentum:
      var_t = var - grad_with_decay * lr_t / (math_ops.sqrt(grad2_acc_t) +
                                              epsilon)
    else:
      var_t = var - grad_with_decay * lr_t / math_ops.sqrt(grad2_acc_t +
                                                           epsilon)
    return state_ops.assign(var, var_t, use_locking=self._use_locking).op

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    raise NotImplementedError('_resource_apply_sparse is not implemented')

  def set_weights(self, weights):
    params = self.weights
    # Override set_weights for backward compatibility of Keras V1 optimizer
    # since it does not include iteration at head of the weight list. Set
    # iteration to 0.
    if len(params) == len(weights) + 1:
      weights = [np.array(0)] + weights
    super(RMSprop, self).set_weights(weights)

  def get_config(self):
    config = super(RMSpropV2, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "weight_decay": self._serialize_hyperparameter("weight_decay"),
        "initial_accumulator_value": self._initial_accumulator_value,
        "epsilon": self._epsilon,
        "use_v2": self._use_v2
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'initial_accumulator_value' not in config:
      config['initial_accumulator_value'] = 0.0
    if 'lr' in config:
      config['learning_rate'] = config.pop('lr')
    return cls(**config)
