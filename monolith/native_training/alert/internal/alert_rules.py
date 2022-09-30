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

import abc
import time

from absl import logging
from typing import NamedTuple

from monolith.monitoring import metric_openapi
from monolith.native_training.alert import alert_pb2
from monolith.native_training.alert.internal import messager


class AlertInterface(abc.ABC):

  @abc.abstractmethod
  def watch(self):
    pass


class KafkaAlert(AlertInterface):

  def __init__(self, config: alert_pb2.KafkaAlertProto, m: messager.Messager,
               open_api: metric_openapi.MetricOpenApi):
    self._config = config
    self._m = m
    self._open_api = open_api
    self._lag_metrics = []
    topics = self._config.topic.split(",")
    if not topics or not self._config.group:
      raise ValueError(
          f"topic or group can't be emprty. Got {config.topic} and {config.group}"
      )
    for topic in topics:
      self._lag_metrics.append(f"inf.bmq.data.{topic}.{config.group}.lag.size")
    logging.info(f"Watching %s", self._lag_metrics)

  def watch(self):
    for metric in self._lag_metrics:
      self._watch_lag_metric(metric)

  def _watch_lag_metric(self, metric: str):
    now = time.time()
    j = self._open_api.query(
        metric_openapi.format_query(now - 12 * 3600, now, metric))

    if not j:
      self._m.send(
          f"Alert: the training {self._config.group} metric cannot be found. Metric: {metric}."
      )
      return
    dps = j[0]["dps"]
    if len(dps) < 2:
      return
    min_lag = min(dps.values())
    if min_lag < 10.0:
      return
    # In this case, probably we are chasing lag
    lags = list(dps.items())
    v = (lags[0][1] - lags[-1][1]) / (int(lags[-1][0]) - int(lags[0][0]))
    if v * 48 * 3600 > lags[-1][1]:
      # Can eliminate lag in the next 48 hours.
      return
    self._m.send(
        f"Alert: the training {self._config.group} lag size is always > 10.0 in the past 24 hours."
    )


class TrainingAlert(AlertInterface):

  def __init__(self, config: alert_pb2.TrainingAlertProto, m: messager.Messager,
               open_api: metric_openapi.MetricOpenApi):
    self._m = m
    self._open_api = open_api
    self._config = config
    if not self._config.prefix:
      raise ValueError("prefix can't be empty.")
    self._global_step_metric = self._config.prefix + ".run_steps.all"
    logging.info("Watching %s", self._global_step_metric)

  def watch(self):
    self._watch_global_step()

  def _watch_global_step(self):
    now = time.time()
    j = self._open_api.query(
        metric_openapi.format_query(now - 6 * 3600, now,
                                    self._global_step_metric))
    if not j or not j[0]["dps"]:
      self._m.send(
          f"Alert: unable to catch the global step metric. Metric: {self._global_step_metric}"
      )
