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
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import test_util

from monolith.native_training.layers.layer_ops import ffm
from monolith.native_training.layers.layer_ops import bernoulli_gate, \
  discrete_gate, discrete_truncated_gate

tf.random.set_seed(0)


class LayerOpsTest(tf.test.TestCase):

  def test_ffm_mul(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      output_maybe_on_gpu = ffm(left=left, right=right, dim_size=4)
      if tf.test.is_gpu_available():
        self.assertEqual(output_maybe_on_gpu.device,
                         '/job:localhost/replica:0/task:0/device:GPU:0')
      with tf.device("/device:CPU:0"):
        output_on_cpu = ffm(left=left, right=right, dim_size=4)
        self.assertEqual(output_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
      self.assertTrue(output_maybe_on_gpu.shape == (8, 480))
      self.assertAllEqual(output_maybe_on_gpu, output_on_cpu)

  def test_ffm_mul_grad(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      with tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4)
        loss = tf.reduce_sum(out)
        left_grad_maybe_on_gpu, right_grad_maybe_on_gpu = g.gradient(
            loss, [left, right])
        self.assertTrue(left_grad_maybe_on_gpu.shape == (8, 40))
        self.assertTrue(right_grad_maybe_on_gpu.shape == (8, 48))

      with tf.device("/device:CPU:0"), tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4)
        loss = tf.reduce_sum(out)
        left_grad_on_cpu, right_grad_on_cpu = g.gradient(loss, [left, right])
        self.assertEqual(left_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertEqual(right_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertAllEqual(left_grad_maybe_on_gpu, left_grad_on_cpu)
        self.assertAllEqual(right_grad_maybe_on_gpu, right_grad_on_cpu)

  def test_ffm_dot(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      output_maybe_on_gpu = ffm(left=left,
                                right=right,
                                dim_size=4,
                                int_type='dot')
      if tf.test.is_gpu_available():
        self.assertEqual(output_maybe_on_gpu.device,
                         '/job:localhost/replica:0/task:0/device:GPU:0')
      with tf.device("/device:CPU:0"):
        output_on_cpu = ffm(left=left, right=right, dim_size=4, int_type='dot')
        self.assertEqual(output_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
      self.assertTrue(output_maybe_on_gpu.shape == (8, 120))
      self.assertAllEqual(output_maybe_on_gpu, output_on_cpu)

  def test_ffm_dot_grad(self):
    with test_util.use_gpu():
      left = tf.random.uniform(shape=(8, 10 * 4), minval=0, maxval=10)
      right = tf.random.uniform(shape=(8, 12 * 4), minval=0, maxval=10)
      with tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4, int_type='dot')
        loss = tf.reduce_sum(out)
        left_grad_maybe_on_gpu, right_grad_maybe_on_gpu = g.gradient(
            loss, [left, right])

        self.assertTrue(left_grad_maybe_on_gpu.shape == (8, 40))
        self.assertTrue(right_grad_maybe_on_gpu.shape == (8, 48))

      with tf.device("/device:CPU:0"), tf.GradientTape() as g:
        g.watch(left)
        g.watch(right)
        out = ffm(left=left, right=right, dim_size=4, int_type='dot')
        loss = tf.reduce_sum(out)
        left_grad_on_cpu, right_grad_on_cpu = g.gradient(loss, [left, right])
        self.assertEqual(left_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertEqual(right_grad_on_cpu.device,
                         '/job:localhost/replica:0/task:0/device:CPU:0')
        self.assertAllEqual(left_grad_maybe_on_gpu, left_grad_on_cpu)
        self.assertAllEqual(right_grad_maybe_on_gpu, right_grad_on_cpu)

  def test_bernoulli_gate(self):
    alpha = tf.constant([-0.5, 1.0, 0.8, -5.0])
    with tf.GradientTape() as g:
      g.watch(alpha)
      # check forward
      bernoulli = bernoulli_gate(alpha)
      sgmoid = tf.nn.sigmoid(alpha)

      # check for backward
      # sigmoid_loss = tf.reduce_sum(sgmoid * tf.constant([0, 1, 1, 0], dtype=tf.float32))
      # sigmoid_grad = g.gradient(sigmoid_loss, alpha)
      grad = [0., 0.19661193, 0.21390972, 0.]

      loss = tf.reduce_sum(bernoulli)
      var_grad = g.gradient(loss, alpha)
      for x, y in zip(var_grad.numpy(), grad):
        if y != 0:
          self.assertAlmostEqual(x, y, delta=1e-6)

  def test_discrete_gate_softmax(self):
    const = [6.0, 1.8, 3.8, -5.0]
    alpha = tf.constant(const)
    discrete = discrete_gate(alpha, temperature=1.5)
    out = list(discrete.numpy())
    softmax = np.exp([i / 1.5 for i in const])
    softmax /= np.sum(softmax)
    for x, y in zip(out, softmax):
      self.assertAlmostEqual(x, y)

  def test_discrete_gate_is_one_hot(self):
    const = [3.0, 1.8, 3.8, -5.0]
    alpha = tf.constant(const)
    discrete = discrete_gate(alpha, is_one_hot=True, use_gumbel=True)
    out = sum(discrete.numpy())
    self.assertAlmostEqual(out, 1.0)

  def test_discrete_gate(self):
    alpha = tf.constant([-0.5, 1.0, 0.8, -5.0])
    with tf.GradientTape() as g:
      g.watch(alpha)
      discrete = discrete_gate(alpha,
                               is_one_hot=False,
                               temperature=1.5,
                               use_gumbel=False)
      loss = tf.reduce_sum(discrete)
      self.assertAlmostEqual(loss.numpy(), 1, delta=1e-6)
      var_grad = g.gradient(loss, alpha)
      for x in var_grad.numpy():
        self.assertAlmostEqual(x, 0)

  def test_discrete_gate_grad(self):
    alpha = tf.constant([-0.5, 1.0, 0.8, -5.0])
    with tf.GradientTape() as g:
      g.watch(alpha)
      discrete = discrete_gate(alpha,
                               is_one_hot=True,
                               temperature=1.5,
                               use_gumbel=False)
      loss = tf.reduce_sum(discrete)

      var_grad = g.gradient(loss, alpha)
      print(var_grad, flush=True)

  def test_discrete_truncated_gate(self):
    alpha = tf.constant([1.5, 1.0, 0.8, -3.0, 2.5])
    with tf.GradientTape() as g:
      g.watch(alpha)
      gates = tf.nn.softmax(alpha / 1.5)
      discrete_truncated = discrete_truncated_gate(alpha,
                                                   threshold=0.9,
                                                   temperature=1.5,
                                                   use_gumbel=False)
      # check forward
      self.assertTrue(discrete_truncated.shape[0] == alpha.shape[0] - 1)
      for x, y in zip(gates.numpy()[1:], discrete_truncated.numpy()):
        self.assertTrue(y == 1 if x >= 0.14 else y == 0)

      # check backward
      # loss_gate = tf.reduce_sum(gates * tf.constant([0, 1, 1, 0, 1], dtype=tf.float32))
      # gate_grad = g.gradient(loss_gate, alpha)
      # print(">>>\n", gate_grad.numpy(), flush=True)
      grad = [-0.11643284, 0.02660953, 0.02328796, -0.00579685, 0.07233221]

      loss = tf.reduce_sum(discrete_truncated)
      var_grad = g.gradient(loss, alpha)
      for x, y in zip(var_grad.numpy(), grad):
        self.assertAlmostEqual(x, y, delta=1e-6)

  def test_discrete_truncated_gate_grad(self):
    alpha = tf.constant([1.5, 1.0, 0.8, -3.0, 2.5])
    with tf.GradientTape() as g:
      g.watch(alpha)
      discrete_truncated = discrete_truncated_gate(alpha,
                                                   threshold=0.9,
                                                   temperature=1.5,
                                                   use_gumbel=True,
                                                   drop_first_dim=False)
      loss = tf.reduce_sum(discrete_truncated)

      var_grad = g.gradient(loss, alpha)
      print(var_grad, flush=True)


if __name__ == '__main__':
  # tf.compat.v1.disable_eager_execution()
  tf.test.main()
