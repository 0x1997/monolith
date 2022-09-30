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
import time

import tensorflow as tf

from monolith.native_training.metric import metric_hook


class Tf2ProfilerHookTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logdir = os.path.join(os.environ["TEST_TMPDIR"], self._testMethodName)
    self.filepattern = os.path.join(self.logdir, "plugins/profile/*")
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.global_step = tf.compat.v1.train.get_or_create_global_step()
      self.train_op = tf.compat.v1.assign_add(self.global_step, 1)

  def _count_files(self):
    return len(tf.io.gfile.glob(self.filepattern))

  def test_steps(self):
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerHook(self.logdir, save_steps=10)
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        pass
    self.assertEqual(self._count_files(), 1)

  def test_multiple_steps(self):
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerHook(self.logdir, save_steps=10)
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        for _ in range(19):
          sess.run(self.train_op)
          # Since profiler directory is named by seconds, we need to make sure
          # two dumps are in the different folder.
          time.sleep(0.15)
      # Triggered at 0, 10, 19
      self.assertEqual(self._count_files(), 3)

  def test_already_profiled(self):
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerHook(self.logdir, save_steps=10)
      tf.profiler.experimental.start(self.logdir)
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        for i in range(15):
          sess.run(self.train_op)
      tf.profiler.experimental.stop()

  def test_secs(self):
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerHook(self.logdir, save_secs=1)
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        for _ in range(10):
          sess.run(self.train_op)
          # In total, we will sleep for 1.5s
          time.sleep(0.15)
      # At least we will 2 dumps (maybe more depending on how fast we run the program)
      self.assertGreaterEqual(self._count_files(), 2)


class Tf2ProfilerCaptureOnceHookTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.logdir = os.path.join(os.environ["TEST_TMPDIR"], self._testMethodName)
    self.filepattern = os.path.join(self.logdir, "plugins/profile/*")
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.global_step = tf.compat.v1.train.get_or_create_global_step()
      self.train_op = tf.compat.v1.assign_add(self.global_step, 1)

  def _count_files(self):
    return len(tf.io.gfile.glob(self.filepattern))

  def test_basic(self):
    # stop and save by .after_run in hook
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerCaptureOnceHook(self.logdir, capture_step_range=[10,18])
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        for _ in range(19):
          sess.run(self.train_op)
          time.sleep(0.15)
      self.assertEqual(self._count_files(), 1)
  
  def test_exceeded_range(self):
    # stop and save by .end in hook
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerCaptureOnceHook(self.logdir, capture_step_range=[10,21])
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        for _ in range(19):
          sess.run(self.train_op)
          time.sleep(0.15)
      self.assertEqual(self._count_files(), 1)

  def test_already_profiled(self):
    with self.graph.as_default():
      hook = metric_hook.Tf2ProfilerCaptureOnceHook(self.logdir, capture_step_range=[10,11])
      tf.profiler.experimental.start(self.logdir)
      with tf.compat.v1.train.SingularMonitoredSession(hooks=[hook]) as sess:
        for i in range(15):
          sess.run(self.train_op)
      tf.profiler.experimental.stop()


if __name__ == "__main__":
  tf.test.main()
