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

"""Monolith Sail API Library Test."""

from absl import app
from absl import flags
from absl import logging
from google.protobuf import text_format
import os

import tensorflow.compat.v1 as tf
from fountain import Config, DataFlow, FeatureConfig, Types, utils
import fountain_pb2

FLAGS = flags.FLAGS


class FountainLibTest(tf.test.TestCase):

  def testBasic(self):
    config_str = r"""
    graph {
      proto_reader {
        kafka_dump_prefix: true
        has_sort_id: true
        data_source {
          dtype: PROTO_INSTANCE
          feature_config {
          }
          thread_num: 1
          max_queue_size: 1
        }
        parallel: 4
      }
      processor {
        name: "matrix_process_0"
        type: "matrix_process"
        parallel: 1
        input {
          dtype: PROTO_INSTANCE
        }
      }
      processor {
        name: "shuffle_0"
        type: "shuffle"
        parallel: 1
        input {
          from: "matrix_process_0"
          dtype: PROTO_INSTANCE
        }
        param {
          key: "shuffle_size"
          value: "0.0"
        }
      }
      processor {
        name: "batch_0"
        type: "batch"
        parallel: 1
        input {
          from: "shuffle_0"
          dtype: PROTO_INSTANCE
        }
        param {
          key: "batch_size"
          value: "256"
        }
      }
    }
    fetch {
      name: "batch_0"
      pool {
        from: "batch_0"
        dtype: PROTO_BATCH
      }
    }
    """
    gold_config = fountain_pb2.FountainConfig()
    text_format.Merge(config_str, gold_config)

    config = Config()
    config.reader.parallel = 4
    config.reader.has_sort_id = True
    config.reader.kafka_dump_prefix = True
    config.reader.kafka_dump = False
    data = DataFlow(data_source="", dtype=Types.PROTO_INSTANCE, config=config).\
        matrix_process().\
        shuffle(shuffle_size=0.).\
        batch(batch_size=256)
    config.add_fetch("batch_0", data_flow=data)

    self.assertEqual(config.config_proto(), gold_config)

  def testFeatureConfig(self):
    config = Config()
    config.reader.filename = ''

    config1 = FeatureConfig(
        ignore_model_meta_feature=True, extra_feature="f_uid,f_aid")
    config2 = FeatureConfig(
        ignore_model_meta_feature=False, extra_feature=['f_uid'])
    data_flow1 = DataFlow(
        data_source="", dtype=Types.EXAMPLE,
        feature_config=config1).to_training_instance()
    data_flow2 = DataFlow(
        data_source="", dtype=Types.EXAMPLE,
        feature_config=config2).to_training_instance()
    data_flow3 = DataFlow(
        data_source="",
        dtype=Types.EXAMPLE,
        feature_config=config2,
        storage='kafka').to_training_instance()

    config.add_fetch(name="data_flow1", data_flow=data_flow1)
    config.add_fetch(name="data_flow2", data_flow=data_flow2)
    config.add_fetch(name="data_flow3", data_flow=data_flow3)
    # print(config.config_proto())

  def testMultiConfig(self):
    # batch config
    batch_config = Config("batch_train")

    toutiao_example = DataFlow(
        data_source="/ad/ctr/toutiao_example",
        dtype=Types.EXAMPLE,
        config=batch_config)
    toutiao_instance = toutiao_example.\
        ad_instance_sample_rate(neg_sample_rate=0.25, merge_ins_sample_rate=1).\
        to_training_instance()

    xigua_instance = DataFlow(
        data_source="/ad/ctr/xigua_instance",
        dtype=Types.PROTO_INSTANCE,
        config=batch_config)
    pipixia_instance = DataFlow(
        data_source="/ad/ctr/pipixia_instance",
        dtype=Types.PROTO_INSTANCE,
        config=batch_config)

    xigua_pipixia_merge = xigua_instance.merge(
        pipixia_instance).matrix_process()

    ctr_merge = xigua_pipixia_merge.to_training_instance().merge(
        toutiao_instance)
    ctr_batch = ctr_merge.shuffle(shuffle_size=1000).batch()

    batch_config.add_fetch(name="ctr_batch", data_flow=ctr_batch)

    # stream config
    stream_config = Config("stream_train")

    toutiao_example = stream_config.add_data_flow(
        data_source="/ad/ctr/toutiao_example", dtype=Types.EXAMPLE_PB)
    toutiao_instance = toutiao_example.to_training_instance().\
        ad_instance_sample_rate(neg_sample_rate=0.25, merge_ins_sample_rate=1). \
        fill_ads_pos_fid().\
        fill_gpos_from_fid().\
        fill_pt_from_fid().\
        fill_rt_from_fid()

    xigua_instance = stream_config.add_data_flow(
        data_source="/ad/ctr/xigua_instance", dtype=Types.PROTO_INSTANCE)
    pipixia_instance = stream_config.add_data_flow(
        data_source="/ad/ctr/pipixia_instance", dtype=Types.PROTO_INSTANCE)

    xigua_pipixia_merge = xigua_instance.merge(
        pipixia_instance).matrix_process()

    ctr_merge = xigua_pipixia_merge.to_training_instance().merge(
        toutiao_instance)
    ctr_batch = ctr_merge.shuffle(shuffle_size=1000).batch(batch_size=200)

    stream_config.add_fetch(name="ctr_batch", data_flow=ctr_batch)

    # Merged multi-config
    utils.serialize_configs([batch_config, stream_config])


if __name__ == '__main__':
  FLAGS.alsologtostderr = True
  app.run(tf.test.main)