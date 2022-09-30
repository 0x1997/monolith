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

import unittest
from unittest import mock

from monolith.native_training.alert.internal import alert_rules
from monolith.native_training.alert import alert_pb2


class KafkaAlertTest(unittest.TestCase):

  def test_basic(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = [{
        "dps": {
            "100": 0.0,
            "200": 0.0,
        }
    }]

    a = alert_rules.KafkaAlert(
        alert_pb2.KafkaAlertProto(topic="topic", group="group"), m, open_api)
    a.watch()
    m.send.assert_not_called()

  def test_metric_missing(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = []

    a = alert_rules.KafkaAlert(
        alert_pb2.KafkaAlertProto(topic="topic", group="group"), m, open_api)
    a.watch()
    m.send.assert_called()

  def test_lag_alert(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = [{
        "dps": {
            "100": 10.0,
            "200": 10.0,
            "300": 10.0,
        }
    }]

    a = alert_rules.KafkaAlert(
        alert_pb2.KafkaAlertProto(topic="topic", group="group"), m, open_api)
    a.watch()
    m.send.assert_called()

  def test_lag_in_elimination(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = [{
        "dps": {
            "100": 90000.0,
            "200": 80000.0,
            "300": 70000.0,
        }
    }]

    a = alert_rules.KafkaAlert(
        alert_pb2.KafkaAlertProto(topic="topic", group="group"), m, open_api)
    a.watch()
    m.send.assert_not_called()

  def test_multi_topic(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = [{
        "dps": {
            "100": 0.0,
            "200": 0.0,
        }
    }]

    a = alert_rules.KafkaAlert(
        alert_pb2.KafkaAlertProto(topic="topic1,topic2", group="group"), m,
        open_api)
    a.watch()
    m.send.assert_not_called()
    self.assertEqual(open_api.query.call_count, 2)


class TrainingAlertTest(unittest.TestCase):

  def test_basic(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = [{
        "dps": {
            "100": 100.0,
            "200": 200.0,
        }
    }]
    a = alert_rules.TrainingAlert(alert_pb2.TrainingAlertProto(prefix="metric"),
                                  m, open_api)
    a.watch()
    m.send.assert_not_called()

  def test_alert(self):
    m = mock.MagicMock()
    open_api = mock.MagicMock()
    open_api.query.return_value = [{"dps": {}}]
    a = alert_rules.TrainingAlert(alert_pb2.TrainingAlertProto(prefix="metric"),
                                  m, open_api)
    a.watch()
    m.send.assert_called()


if __name__ == "__main__":
  unittest.main()
