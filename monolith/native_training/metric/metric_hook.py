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

# Metrics codes are ported from Lagrange Lite: lagrange_lite/tensorflow/train.py
#coding:utf-8
import json
import numpy as np
import os
import tensorflow as tf
import time
from typing import Tuple

from absl import logging
from datetime import datetime
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

from monolith.native_training.alert import alert_manager
from monolith.native_training.alert import alert_pb2
from monolith.native_training.metric import cli
from monolith.native_training import utils
from monolith.native_training.metric.kafka_utils import KProducer



class ThroughputMetricHook(tf.estimator.SessionRunHook):
  """ Log accumulated steps and time elapsed per step. """

  def __init__(self,
               model_name,
               start_time_secs,
               cluster_type="stable",
               run_every_n_secs=30):

    self._model_name = model_name
    self._start_time_secs = start_time_secs
    self._cluster_type = cluster_type
    self._run_every_n_secs = run_every_n_secs
    self._is_first_step = True
    self._mcli = cli.get_cli(utils.get_metric_prefix())
    am = alert_manager.get_default_alert_manager()
    if am:
      proto = alert_pb2.AlertProto()
      proto.training_alert.prefix = utils.get_metric_prefix()
      am.add_rules(proto)

  def begin(self):
    self._global_step_tensor = tf.compat.v1.train.get_global_step()

  def before_run(self, run_context):
    if self._is_first_step is True:
      self._emit_step = run_context.session.run(self._global_step_tensor)
      self._emit_time = int(time.time())
      if self._start_time_secs is not None:
        tags = {
            "model_name": self._model_name,
            "cluster_type": self._cluster_type
        }
        run_start_elapsed_time = self._emit_time - self._start_time_secs
        logging.info("Run start took {}s.".format(run_start_elapsed_time))
        self._mcli.emit_timer("run_start_elapsed_time.all",
                              run_start_elapsed_time, tags)
      self._is_first_step = False
    return session_run_hook.SessionRunArgs({
        "global_step": self._global_step_tensor,
    })

  def after_run(self, run_context, run_values):
    end_time = int(time.time())
    elapsed_time = end_time - self._emit_time
    if elapsed_time >= self._run_every_n_secs:
      global_step = run_values.results["global_step"]
      step_inerval = global_step - self._emit_step
      tags = {
          "model_name": self._model_name,
          "cluster_type": self._cluster_type
      }
      self._mcli.emit_counter("run_steps.all", step_inerval, tags)
      self._mcli.emit_timer("run_steps_elapsed_time.all",
                            elapsed_time / step_inerval, tags)
      self._emit_step = global_step
      self._emit_time = end_time


class StepLossMetricHook(tf.estimator.SessionRunHook):
  """ Log loss of each step. """

  def __init__(self, loss_tensor):
    self._loss_tensor = loss_tensor
    self._mcli = cli.get_cli(utils.get_metric_prefix())

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._loss_tensor)

  def after_run(self, run_context, run_value):
    self._mcli.emit_store("step_loss", run_value.results)


class CustomMetricHook(tf.estimator.SessionRunHook):
  """ Log group of customed metircs for a batch. """

  def __init__(self, metric_tensors):
    for name in metric_tensors:
      tensor = metric_tensors[name]
      if len(tensor.shape.dims) > 0:
        raise ValueError("The metric tensor should be a scalar!")
      if tensor.dtype.base_dtype not in (tf.float32, tf.int32):
        raise ValueError(
            "The dtype of a metric tensor should be either tf.float or tf.int32!"
        )
    if len(metric_tensors) == 0:
      raise ValueError("At least one metric tensor should be offered!")
    self._metric_tensors = metric_tensors
    self._mcli = cli.get_cli(utils.get_metric_prefix())

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._metric_tensors)

  def after_run(self, run_context, run_value):
    metric_values = run_value.results
    for name in metric_values:
      self._mcli.emit_store(name, float(metric_values[name]))


class Tf2ProfilerHook(tf.estimator.SessionRunHook):
  """ Using TF2 profiler in esitmator """

  def __init__(self,
               logdir: str,
               save_steps: int = None,
               save_secs: int = None,
               options: tf.profiler.experimental.ProfilerOptions = None):
    """Only one of save_steps and save_secs should be provided."""
    self._logdir = logdir
    self._options = options
    self._timer = tf.estimator.SecondOrStepTimer(every_steps=save_steps,
                                                 every_secs=save_secs)
    self._global_step_tensor = tf.compat.v1.train.get_global_step()

    self._profiling = False

  def begin(self):
    self._start_profiling()

  def before_run(self, run_context):
    del run_context
    return tf.estimator.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values: tf.estimator.SessionRunValues):
    del run_context
    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      self._stop_profiling()
      self._timer.update_last_triggered_step(global_step)
      self._start_profiling()

  def end(self, sess):
    del sess
    self._stop_profiling()

  def _start_profiling(self):
    try:
      tf.profiler.experimental.start(self._logdir, self._options)
      self._profiling = True
    except tf.errors.AlreadyExistsError:
      # User profiles by themselves. OK to ignore here.
      pass

  def _stop_profiling(self):
    try:
      if self._profiling:
        self._profiling = False
        tf.profiler.experimental.stop()
    except tf.errors.UnavailableError:
      # Maybe user terminates profiling, ignore here.
      pass


class Tf2ProfilerCaptureOnceHook(tf.estimator.SessionRunHook):
  """Using TF2 profiler in esitmator to capture only once."""

  def __init__(self,
               logdir: str,
               capture_step_range: Tuple[int, int],
               options: tf.profiler.experimental.ProfilerOptions = None):
    """Capture the profiler between (start, end) step of capture_step_range."""
    self._logdir = logdir
    self._start_step, self._end_step = capture_step_range
    self._options = options

    self._profiling = False

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use Tf2ProfilerCaptureOnceHook.")

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values: tf.estimator.SessionRunValues):
    current_step = run_values.results
    if self._start_step is None:
      self._start_step = current_step + 10
      self._end_step = self._start_step + 10

    if not self._profiling and current_step >= self._start_step - 1 and current_step < self._end_step - 1:
      self._start_profiling()
    if self._profiling and current_step >= self._end_step - 1:
      self._stop_profiling()

  def end(self, sess):
    if self._profiling:
      self._stop_profiling()

  def _start_profiling(self):
    try:
      tf.profiler.experimental.start(self._logdir, self._options)
      self._profiling = True
    except tf.errors.AlreadyExistsError:
      # User profiles by themselves. OK to ignore here.
      pass

  def _stop_profiling(self):
    try:
      if self._profiling:
        self._profiling = False
        tf.profiler.experimental.stop()
    except tf.errors.UnavailableError:
      # Maybe user terminates profiling, ignore here.
      pass


class ByteCCLTelemetryHook(tf.estimator.SessionRunHook):
  """Log telemetry information at regular intervals"""

  def __init__(self, interval: int):
    """Log telemetry information at regular intervals"""
    self._interval = interval
    self._last_step = 0
    logging.info(f"Created ByteCCL telemetry hook, interval={interval}")

  def begin(self):
    self._global_step_tensor = training_util._get_or_create_global_step_read()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use ByteCCLTelemetryHook")

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values: tf.estimator.SessionRunValues):
    current_step = run_values.results
    if current_step > self._last_step + self._interval:
      self._log_telemetry()
      self._last_step = current_step

  def end(self, sess):
    pass

  def _log_telemetry(self):
    import byteps.tensorflow as bps
    if bps.rank() == 0:
      telemetry = bps.get_telemetry()
      # sample a few operations and show them
      samples = []
      num_allreduce_ops = 0
      for name, mean, stdev, count in telemetry:
        name = str(name)
        is_alltoall = 'alltoall' in name.lower()
        if is_alltoall or ('PushPull' in name and num_allreduce_ops < 3):
          num_allreduce_ops += 1
          entry = f'name: {name} mean(ms): {mean:.2f} stdev(ms): {stdev:.2f} count: {count}'
          samples.append(entry)
      if samples:
        logging.info(f'Communication telemetry: {samples} ...')


class NVProfilerCaptureOnceHook(Tf2ProfilerCaptureOnceHook):

  def __init__(self, capture_step_range: Tuple[int, int]):
    super().__init__(None, capture_step_range)
    import ctypes
    self._libcudart = ctypes.cdll.LoadLibrary("libcudart.so")  # linux

  def _start_profiling(self):
    # http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    self._libcudart.cudaProfilerStart()
    self._profiling = True

  def _stop_profiling(self):
    if self._profiling:
      self._profiling = False
      self._libcudart.cudaProfilerStop()


class KafkaMetricHook(tf.estimator.SessionRunHook):
  """ Log group of customed metircs for a batch. """
  __instance = None 

  def __new__(cls, *args, **kwargs):
    if cls.__instance is None:
      cls.__instance = super().__new__(cls)
      cls.__instance._kproducer = None
      cls.__instance._init_kafka()

    return cls.__instance

  @classmethod
  def _init_kafka(cls):
    brokers = os.getenv('KAFKA_BROKER_LIST', None)
    topic = os.getenv('KAFKA_TOPIC_NAME', None)
    if brokers is None or topic is None:
      logging.info('KafkaMetricHook init kafka failed, brokers: {}, topic: {}'.format(brokers, topic))
      return

    cls.__instance._kproducer = KProducer(brokers, topic)
    logging.info('KafkaMetricHook init kafka success, brokers: {}, topic: {}'.format(brokers, topic))

  def __init__(self, deep_insight_op = None):
    if deep_insight_op is None:
      collection = tf.compat.v1.get_collection(key='deep_insight_op')
      if collection:
        if isinstance(collection, (list, tuple)):
          deep_insight_op = collection[0]
        else:
          deep_insight_op = collection
    self._metric_tensors = {'deep_insight_op': deep_insight_op}

  def before_run(self, run_context):
    return tf.estimator.SessionRunArgs(self._metric_tensors)

  def after_run(self, run_context, run_value):
    if self._kproducer:
      metric_values = run_value.results
      msgs = metric_values.get('deep_insight_op')
      if msgs is not None and len(msgs) > 0:
        self._kproducer.send(msgs)

  def end(self, session):
    if self._kproducer:
      self._kproducer.close()
      logging.info('KafkaMetricHook end, flush msg, success: {}, failed: {}'.\
        format(self._kproducer.success(), self._kproducer.failed()))
      self._kproducer = None

