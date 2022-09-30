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

layer_ops_lib = load_library.load_op_library(
    get_libops_path('monolith/native_training/layers/layer_tf_ops.so'))


def ffm(left: tf.Tensor, right: tf.Tensor, dim_size: int, int_type: str = 'multiply') -> tf.Tensor:
  output = layer_ops_lib.FFM(left=left, right=right, dim_size=dim_size, int_type=int_type)
  return output


@tf.RegisterGradient('FFM')
def _ffm_grad(op, grad: tf.Tensor) -> tf.Tensor:
  left, right = op.inputs[0], op.inputs[1]
  dim_size = op.get_attr('dim_size')
  int_type = op.get_attr('int_type')

  (left_grad, right_grad) = layer_ops_lib.FFMGrad(grad=grad, left=left, right=right, 
                                                  dim_size=dim_size, int_type=int_type)
  return left_grad, right_grad
def bernoulli_gate(alpha: tf.Tensor, ste_type: str = None, 
                   use_logistic: bool = False, temperature: float = 1.0) -> tf.Tensor:
  if ste_type is None:
    ste_type = 'none'
  else:
    assert ste_type in {'softplus', 'clip', 'none'}
  
  (sampled, proba) = layer_ops_lib.BernoulliGate(alpha=alpha, ste_type=ste_type, 
                                                 use_logistic=use_logistic, temperature=temperature)
  return sampled


@tf.RegisterGradient('BernoulliGate')
def _bernoulli_gate_grad(op, sampled_grad: tf.Tensor, proba_grad: tf.Tensor) -> tf.Tensor:
  alpha, proba = op.inputs[0], op.outputs[1]
  ste_type = op.get_attr('ste_type')
  use_logistic = op.get_attr('use_logistic')
  temperature = op.get_attr('temperature')

  output = layer_ops_lib.BernoulliGateGrad(grad=sampled_grad, alpha=alpha, proba=proba, ste_type=ste_type,
                                           use_logistic=use_logistic, temperature=temperature)
  return output


def discrete_gate(alpha: tf.Tensor, is_one_hot: bool = False, use_gumbel: bool = False, temperature: float = 1.0):
  (sampled, proba) = layer_ops_lib.DiscreteGate(alpha=alpha, is_one_hot=is_one_hot, 
                                                use_gumbel=use_gumbel, temperature=temperature)
  return sampled if is_one_hot else proba


@tf.RegisterGradient('DiscreteGate')
def _discrete_gate_grad(op, sampled_grad, proba_grad):
  is_one_hot = op.get_attr('is_one_hot')
  grad = sampled_grad if is_one_hot else proba_grad
  sampled, proba = op.outputs[0], op.outputs[1]
  temperature = op.get_attr('temperature')

  output = layer_ops_lib.DiscreteGateGrad(grad=grad, sampled=sampled, proba=proba, 
                                          is_one_hot=is_one_hot, temperature=temperature)
  return output


def discrete_truncated_gate(alpha: tf.Tensor, threshold: float = 1.0, drop_first_dim: bool = True, use_gumbel: bool = False, temperature: float = 1.0):
  (sampled, proba) = layer_ops_lib.DiscreteTruncatedGate(alpha=alpha, threshold=threshold, drop_first_dim=drop_first_dim,
                                                         use_gumbel=use_gumbel, temperature=temperature)
  return sampled


@tf.RegisterGradient('DiscreteTruncatedGate')
def _discrete_truncated_gate_grad(op, sampled_grad, proba_grad):
  threshold = op.get_attr('threshold')
  sampled, proba = op.outputs[0], op.outputs[1]
  temperature = op.get_attr('temperature')
  drop_first_dim = op.get_attr('drop_first_dim')

  output = layer_ops_lib.DiscreteTruncatedGateGrad(grad=sampled_grad, sampled=sampled, proba=proba, 
                                                   threshold=threshold, drop_first_dim=drop_first_dim, temperature=temperature)
  return output
