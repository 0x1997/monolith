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

import time

from absl import logging
import json
import tensorflow as tf

import monolith.native_training.metric.deep_insight_ops as ops


class DeepInsightOpsTest(tf.test.TestCase):

  def test_basic(self):
    # prepare test data
    now = int(time.time())
    bytedance_born = 1331481600
    # 2012/03/12
    uids_list = [0, 19, 1005]
    req_times_list = [bytedance_born + i for i in range(0, 3)]
    labels_list = [0.1, 0.2, 0.3]
    preds_list = [0.4, 0.5, 0.6]
    sample_rates_list = [0.7, 0.8, 0.9]
    sample_ratio = 0.01

    deep_insight_client = ops.deep_insight_client(enable_metrics_counter=False)

    model_name = "deep_insight_test_{}".format(now)
    target = "ctr_head"
    logging.info("model_name: {}, target: {}".format(model_name, target))

    uids = tf.constant(uids_list, dtype=tf.int64)
    req_times = tf.constant(req_times_list, dtype=tf.int64)
    labels = tf.constant(labels_list, dtype=tf.float32)
    preds = tf.constant(preds_list, dtype=tf.float32)
    sample_rates = tf.constant(sample_rates_list, dtype=tf.float32)

    msgs = ops.write_deep_insight(
        deep_insight_client_tensor=deep_insight_client,
        uids=uids,
        req_times=req_times,
        labels=labels,
        preds=preds,
        sample_rates=sample_rates,
        model_name=model_name,
        target=target,
        sample_ratio=sample_ratio,
        return_msgs=True,
        use_zero_train_time=True)

    with self.session() as sess:
      msgs_values = sess.run(msgs)

    for i, msg in enumerate(msgs_values):
      if uids_list[i] % 1000 >= 10:
        continue
      parsed = json.loads(msg)
      self.assertAllClose(parsed["sample_rate"]["ctr_head"],
                          sample_rates_list[i])
      self.assertAllClose(parsed["predict"]["ctr_head"], preds_list[i])
      self.assertAllClose(parsed["label"]["ctr_head"], labels_list[i])

  def test_write_deep_insight_v2(self):
    # prepare test data
    bytedance_born = 1331481600
    # 2012/03/12
    req_times_list = [bytedance_born + i for i in range(0, 2)]
    labels_list = [[0.1, 0.15], [0.2, 0.25], [0.3, 0.35]]
    preds_list = [[0.4, 0.45], [0.5, 0.55], [0.6, 0.65]]
    sample_rates_list = [[0.7, 0.75], [0.8, 0.85], [0.9, 0.95]]
    sample_ratio = 1.

    deep_insight_client = ops.deep_insight_client(enable_metrics_counter=False)

    model_name = "deep_insight_v2_test"
    targets = ["a_head", "b_head", "c_head"]
    logging.info("model_name: {}, targets: {}".format(model_name, targets))

    # uids = tf.constant(uids_list, dtype=tf.int64)
    req_times = tf.constant(req_times_list, dtype=tf.int64)
    labels = tf.constant(labels_list, dtype=tf.float32)
    preds = tf.constant(preds_list, dtype=tf.float32)
    sample_rates = tf.constant(sample_rates_list, dtype=tf.float32)
    extra_int_fields = [123, 456]
    extra_float_fields = [0.333, 0.666]
    extra_str_fields = ["foo", "bar"]
    extra_fields_values = [
        tf.constant(extra_int_fields, dtype=tf.int64),
        tf.constant(extra_float_fields, dtype=tf.float32),
        tf.constant(extra_str_fields, dtype=tf.string)
    ]
    extra_fields_keys = ["uid", "float_field", "string_field"]

    msgs = ops.write_deep_insight_v2(
        deep_insight_client_tensor=deep_insight_client,
        req_times=req_times,
        labels=labels,
        preds=preds,
        sample_rates=sample_rates,
        model_name=model_name,
        sample_ratio=sample_ratio,
        extra_fields_values=extra_fields_values,
        extra_fields_keys=extra_fields_keys,
        targets=targets,
        return_msgs=True,
        use_zero_train_time=True)

    with self.session() as sess:
      msgs_values = sess.run(msgs)

    msgs_values = [m.decode("utf-8") for m in msgs_values]

    for i, msg in enumerate(msgs_values):
      parsed = json.loads(msg)
      print(print(json.dumps(parsed, indent=4)))
      self.assertAllClose(parsed["sample_rate"]["a_head"],
                          sample_rates_list[0][i])
      self.assertAllClose(parsed["predict"]["b_head"], preds_list[1][i])
      self.assertAllClose(parsed["label"]["c_head"], labels_list[2][i])
      self.assertEqual(parsed["req_time"], req_times_list[i])
      self.assertAllClose(parsed["extra_float"]["float_field"], extra_float_fields[i])
      self.assertEqual(parsed["extra_str"]["string_field"], extra_str_fields[i])


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.test.main()