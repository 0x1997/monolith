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

from absl import logging
import os
import getpass
import random
import numpy as np
import tensorflow as tf
from struct import pack, unpack
from datetime import datetime, timedelta

from idl.matrix.proto.proto_parser_pb2 import Instance
from monolith.native_training.data.parsers import parse_instances
from monolith.native_training.data.datasets import PBDataset, PbType

uids = [674432, 9754221, 7665435, 98797865, 778754432]
item_ids = [8767554565, 574220985, 65548979, 5358521231]
actions = [1, 2]
device_types = ['pc', 'mobile', 'cloud']
slots = [1, 200, 5, 7, 9]
NUM_INSTANCE = 4096
home_dir = os.environ.get('HOME') or os.getcwd()
MODEL_DIR = os.path.join(home_dir, 'model_dir', 'multi_flow')

class MultiFlowTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    mask = (1 << 54) - 1
    start = int(datetime.now().timestamp())
    stop = int((datetime.now() + timedelta(days=1)).timestamp())
    if not tf.io.gfile.exists(MODEL_DIR):
      tf.io.gfile.makedirs(MODEL_DIR)
    ofile = os.path.join(MODEL_DIR, 'data.pb')
    print(ofile, flush=True)
    if not tf.io.gfile.exists(ofile):
      with tf.io.gfile.GFile(ofile, 'wb') as ostream:
        for _ in range(NUM_INSTANCE):
          inst = Instance()
          for slot in slots:
            h = random.randrange(start, stop)
            fid = (slot << 54) | (h & mask) 
            inst.fid.append(fid)
          
          line_id = inst.line_id
          line_id.uid = random.choice(uids)
          line_id.item_id = random.choice(item_ids)
          line_id.req_time = random.randrange(start, stop)
          line_id.device_type = random.choice(device_types)
          line_id.actions.append(random.choice(actions))

          lgx_header = cls.mk_kgx_header(dataflow=line_id.device_type)
          data = inst.SerializeToString()

          ostream.write(file_content=lgx_header)
          ostream.write(file_content=pack(f'<Q', len(data)))
          ostream.write(file_content=data)

  @classmethod
  def tearDownClass(cls):
    if not tf.io.gfile.exists(MODEL_DIR):
      tf.io.gfile.rmtree(MODEL_DIR)

  @classmethod
  def mk_kgx_header(cls, dataflow: str):
    # calc java hash code
    seed, h = 31, 0
    for c in dataflow:
      h = np.int32(seed * h) + ord(c)
    
    dfhc = int(np.uint32(h)).to_bytes(4, 'little')
    return pack('4Bi', 0, dfhc[0], dfhc[1], dfhc[2], 0)
  
  def test_data_flow(self):
    ofile = os.path.join(MODEL_DIR, 'data.pb')
    dataset = PBDataset(file_name=ofile, 
                        lagrangex_header=True, 
                        input_pb_type=PbType.INSTANCE, 
                        output_pb_type=PbType.INSTANCE)
    pc = dataset.split_flow(data_flow=device_types, index=0, variant_type='instance')
    mobile = dataset.split_flow(data_flow=device_types, index=1, variant_type='instance')
    cloud = dataset.split_flow(data_flow=device_types, index=2, variant_type='instance')

    dataset = pc.merge_flow(dataset_to_merge=[mobile, cloud], variant_type='instance')

    def map_fn(tensor: tf.Tensor):
      features = parse_instances(tensor, fidv1_features=slots, 
                      extra_features=['uid', 'item_id', 'req_time', 'device_type', 'actions'],
                      extra_feature_shapes=[1, 1, 1, 1, 1])
      return features
    
    dataset = dataset.batch(batch_size=512, drop_remainder=True).map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    cnt = 0
    for feat in dataset:
      cnt += 1
    self.assertEqual(cnt, 8)

  def check_data(self):
    fname = '/data01/home/wangcaihua/getdata/data.pb' 
    dataflows = ['/f100/jarvis_instance/preclk', '/f100/push/house_instance/fast_join_window_v2']

    stats_data = {unpack('<Q', self.mk_kgx_header(source))[0]: {'name': source, 'pos': 0, 'neg': 0}
                  for source in dataflows}
    print(stats_data, flush=True)
    # 4068439040: {'name': '/f100/jarvis_instance/preclk', 'pos': 103544, 'neg': 1071141}
    # 2008899072: {'name': '/f100/push/house_instance/fast_join_window_v2', 'pos': 1629, 'neg': 153895}
    with open(fname, 'rb') as istream:
      while True:
        try:
          lg_header = unpack('<Q', istream.read(8))[0]
          size = unpack('<Q', istream.read(8))[0]
          if size > 0:
            inst = Instance()
            inst.ParseFromString(istream.read(size))
            if inst.label[0] > 0.5:
              stats_data[lg_header]['pos'] += 1
            else:
              stats_data[lg_header]['neg'] += 1
        except:
          print(stats_data, flush=True)
          break

  def count(self):
    fname = '/data01/home/wangcaihua/getdata/data.pb' 
    device_types = ['/f100/jarvis_instance/preclk', '/f100/push/house_instance/fast_join_window_v2']
    dataset = PBDataset(file_name=fname,
                        kafka_dump=True,
                        kafka_dump_prefix=False,
                        lagrangex_header=True,
                        input_pb_type=PbType.INSTANCE,
                        output_pb_type=PbType.INSTANCE)

    # 分流
    feed_data = dataset.split_flow(data_flow=device_types, index=0, variant_type='instance')
    push_data = dataset.split_flow(data_flow=device_types, index=1, variant_type='instance')
    dataset = feed_data.merge_flow(dataset_to_merge=[push_data], variant_type='instance')

    def map_fn(ts):
      return parse_instances(ts, fidv1_features=[1, 200], 
                             dense_features=['label'], 
                             dense_feature_shapes=[1], 
                             dense_feature_types=[tf.float32])
    dataset = dataset.batch(batch_size=512, drop_remainder=False).map(map_fn, tf.data.AUTOTUNE)

    cnt, pos, neg = 0, 0, 0
    for inst in dataset:
      label = inst['label']
      pos += np.sum(label > 0)
      neg += np.sum(label < 0)
      cnt += label.shape[0]
    self.assertEqual(cnt, 1330209)
    self.assertEqual(pos, 105173)
    self.assertEqual(neg, 1225036)

  def count_with_negative_gen(self):
    fname = '/data01/home/wangcaihua/getdata/data.pb' 
    device_types = ['/f100/jarvis_instance/preclk', '/f100/push/house_instance/fast_join_window_v2']
    dataset = PBDataset(file_name=fname,
                        kafka_dump=True,
                        kafka_dump_prefix=False,
                        lagrangex_header=True,
                        input_pb_type=PbType.INSTANCE,
                        output_pb_type=PbType.INSTANCE)

    # 分流
    feed_data = dataset.split_flow(data_flow=device_types, index=0, variant_type='instance')
    push_data = dataset.split_flow(data_flow=device_types, index=1, variant_type='instance')

    push_data = push_data.negative_gen_deprecated(neg_num=10,
                          per_channel_sample=True,
                          channel_feature_name=507,
                          group_feature_names=[2,501,502,503,504,505,506,507,508,509,510,511,512,513,514,515,518,519,520,521,522,523,524,525,526,527,528,529,530,531,532,533,534,535,543,547,548,549,550],
                          max_group_num_per_channel=5000, 
                          label_field='label', label_index=0, negative_label=-1,
                          variant_type='instance')
    dataset = feed_data.merge_flow(dataset_to_merge=[push_data], variant_type='instance')

    def map_fn(ts):
      return parse_instances(ts, fidv1_features=[1, 200], 
                             dense_features=['label'], 
                             dense_feature_shapes=[1], 
                             dense_feature_types=[tf.float32],
                             extra_features=['data_source_name'],
                             extra_feature_shapes=[1])
    dataset = dataset.batch(batch_size=512, drop_remainder=False).map(map_fn, tf.data.AUTOTUNE)

    cnt, pos, neg, preclk, push = 0, 0, 0, 0, 0
    preclk_pos, preclk_neg, push_pos, push_neg = 0,0,0,0
    for inst in dataset:
      label = inst['label']
      data_source_name = inst['data_source_name']
      is_preclk = data_source_name == 'data_source4068439040'
      num_preclk = np.sum(is_preclk)

      is_pos = label > 0
      num_pos = np.sum(is_pos)

      preclk += num_preclk
      push += label.shape[0] - num_preclk
      pos += num_pos
      neg += label.shape[0] - num_pos

      preclk_pos += np.sum(np.logical_and(is_preclk, is_pos))
      preclk_neg += np.sum(np.logical_and(is_preclk, np.logical_not(is_pos)))
      push_pos += np.sum(np.logical_and(np.logical_not(is_preclk), is_pos))
      push_neg += np.sum(np.logical_and(np.logical_not(is_preclk), np.logical_not(is_pos)))

      cnt += label.shape[0]
    self.assertEqual(cnt, 1330209)
    self.assertEqual(pos, 105173)
    self.assertEqual(neg, 1225036)
    self.assertEqual(push_pos, 1629)
    self.assertEqual(push_neg, 153895)
    print(cnt, pos, neg, preclk, push, flush=True)
    print(preclk_pos, preclk_neg, push_pos, push_neg, flush=True)


if __name__ == "__main__":
  tf.test.main()
