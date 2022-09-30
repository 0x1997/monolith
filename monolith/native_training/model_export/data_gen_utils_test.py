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
import getpass
import tensorflow as tf
from struct import pack, unpack
from idl.matrix.proto.line_id_pb2 import LineId
from idl.matrix.proto.feature_pb2 import Feature as IFeature
from idl.matrix.proto.example_pb2 import Feature as EFeature
from monolith.native_training.data.feature_list import FeatureList, Feature as FeatureConf
from monolith.native_training.data.datasets import PBDataset, PbType
from monolith.native_training.data.parsers import parse_example_batch, parse_examples, parse_instances
import monolith.native_training.model_export.data_gen_utils as utils
from monolith.native_training.utils import add_to_collections


feature_list_file = "monolith/native_training/data/test_data/cmcc.conf"
direct_feats = ['f_album_id', 'f_rating', 'f_alias_bes']

recent_feats = [
    'f_user_lt_play_album_id_recent',
    'f_user_st_1d_play_album_name_recent',
    'f_user_st_20d_play_bhv_time_hour_recent',
]

cp_feats = [
    'f_user_st_3d_play_album_id_cp',
    'f_user_st_1d_play_album_name_terms_cp',
    'f_user_st_1d_play_bhv_time_hour_cp',
]

combine_feats = [
    'f_user_lt_play_album_name_terms_cp-f_bhv_time_hour',
    'f_user_lt_play_album_name_terms_cp-f_bhv_time_monthday',
    'f_channel_id-f_stars'
]

feature_names = direct_feats + recent_feats + cp_feats + combine_feats


class DataGenUtilsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.feature_list = FeatureList.parse(feature_list_file)

  def test_gen_fids_v1(self):
    fids = utils.gen_fids_v1(123, size=3)
    self.assertTrue(len(fids)== 3)
    self.assertTrue(all(isinstance(fid, int) and (fid >> 54) == 123 for fid in fids))

  def test_gen_fids_v2(self):
    fids = utils.gen_fids_v2(123, size=3)
    self.assertTrue(len(fids)== 3)
    self.assertTrue(all(isinstance(fid, int) and (fid >> 48) == 123 for fid in fids))

  def test_fill_names_features(self):
    feature = EFeature()
    feature_name = 'feature_name'
    extra = FeatureConf(feature_name, slot=2, method='DirectString')
    meta = utils.FeatureMeta(name=feature_name, slot=2, extra=extra)
    utils.fill_features(feature, meta=meta)
    values = feature.fid_v2_list.value
    self.assertTrue(len(values) == 1 and (values[0] >> 48) == 2)

  def test_fill_features(self):
    feature = IFeature()
    meta = utils.FeatureMeta(name='feature_name', slot=2)
    utils.fill_features(feature, meta=meta)
    self.assertTrue(len(feature.fid) >= 1)

  def test_fill_line_id(self):
    line_id = LineId()
    extra_features=['uid', 'item_id', 'actions', 'req_id', 'page_count', 'abtest_label']
    extra_feature_shapes=[1, 1, 3, 1, 1, 3]

    features = [utils.FeatureMeta(name=name, shape=shape)
                for name , shape in zip(extra_features, extra_feature_shapes)]
    utils.fill_line_id(line_id, features, hash_len=54)
    self.assertEqual(len(line_id.actions), 3)
    self.assertEqual(len(line_id.abtest_label), 3)
    print(line_id, flush=True)

  def test_gen_instance(self):
    fidv1_features = [1, 200, 3, 5, 9, 203, 205]
    instance = utils.gen_instance(fidv1_features=fidv1_features,
                                  dense_features=[utils.FeatureMeta('name1', slot=1025), 
                                                  utils.FeatureMeta('name2', shape=2, dtype=tf.float32)],
                                  extra_features=[utils.FeatureMeta('actions', shape=2), utils.FeatureMeta('uid')])
    self.assertTrue(len(instance.fid) <= len(fidv1_features) * 3)
    self.assertEqual(len(instance.line_id.actions), 2)

  def test_gen_example(self):
    example = utils.gen_example(sparse_features=feature_names,
                                feature_list=self.feature_list)
    for named_feature in example.named_feature:
      if named_feature.name in direct_feats:
        try:
          self.assertTrue(len(named_feature.feature.fid_v2_list.value) == 1)
        except:
          self.assertTrue(named_feature.name == 'f_alias_bes')
          self.assertTrue(len(named_feature.feature.fid_v2_list.value) >= 0)
      else:
        self.assertTrue(named_feature.name in feature_names)
        self.assertTrue(len(named_feature.feature.fid_v2_list.value) >= 0)

  def test_gen_example_batch(self):
    example_batch = utils.gen_example_batch(sparse_features=feature_names,
                                            feature_list=self.feature_list,
                                            batch_size=8)
    for named_feature_list in example_batch.named_feature_list:
      self.assertTrue(len(named_feature_list.feature) == 8)
      self.assertTrue(named_feature_list.name in feature_names or
                      named_feature_list.name == '__LINE_ID__' or
                      named_feature_list.name == '__LABEL__')

      for feat in named_feature_list.feature:
        if named_feature_list.name == '__LINE_ID__':
          line_id = LineId()
          line_id.ParseFromString(feat.bytes_list.value[0])
          self.assertTrue(len(line_id.actions) == 1)
          self.assertTrue(0 <= line_id.actions[0] <= 10)
          continue
        if named_feature_list.name == '__LABEL__':
          self.assertTrue(len(feat.float_list.value) == 1)
          self.assertTrue(feat.float_list.value[0] in [0, 1])
          continue

        if named_feature_list.name in direct_feats:
          try:
            self.assertTrue(len(feat.fid_v2_list.value) == 1)
          except:
            self.assertTrue(named_feature_list.name == 'f_alias_bes')
            self.assertTrue(len(feat.fid_v2_list.value) >= 0)
        else:
          self.assertTrue(named_feature_list.name in feature_names)
          self.assertTrue(len(feat.fid_v2_list.value) >= 0)

  def test_gen_prediction_log(self):
    args = utils.ParserArgs(model_name='entry',
                            sparse_features=feature_names,
                            max_records=40,
                            feature_list=self.feature_list,
                            batch_size=8,
                            variant_type='example_batch')
    prediction_logs = utils.gen_prediction_log(args)
    self.assertTrue(len([log for log in prediction_logs]) > 0)

  def test_gen_warmup_args(self):
    extra_features = ['sample_rate', 'req_time', 'actions']
    extra_feature_shapes = [1, 1, 1]

    add_to_collections('sparse_features', feature_names)
    add_to_collections('extra_features', extra_features)
    add_to_collections('extra_feature_shapes', extra_feature_shapes)
    add_to_collections('variant_type', 'example')

    warmup_args = utils.ParserArgs()
    self.assertTrue(warmup_args.variant_type == 'example')
    self.assertListEqual(warmup_args.sparse_features, feature_names)
    self.assertListEqual(warmup_args.extra_features, extra_features)
    self.assertListEqual(warmup_args.extra_feature_shapes, extra_feature_shapes)

  def test_gen_warmup_file(self):
    extra_features = ['sample_rate', 'req_time', 'actions']
    extra_feature_shapes = [1, 1, 1]

    add_to_collections('sparse_features', feature_names)
    add_to_collections('extra_features', extra_features)
    add_to_collections('extra_feature_shapes', extra_feature_shapes)
    add_to_collections('variant_type', 'example')

    warmup_file = "{}/tmp/{}/warmup_files/tf_serving_warmup_requests".format(
        os.getenv("HOME"), getpass.getuser())
    utils.gen_warmup_file(warmup_file)

  def test_gen_random_instance_file(self):
    args = utils.ParserArgs(
        fidv1_features=[1, 200, 5, 6, 8, 234, 567],
        extra_features=['sample_rate', 'req_time', 'actions'],
        extra_feature_shapes=[1, 1, 1],
        variant_type='instance')
    data_file = "instance_files.pb"
    sort_id=True
    kafka_dump=True
    utils.gen_random_data_file(data_file, args, sort_id=sort_id,
                               kafka_dump=kafka_dump,
                               actions=[1, 2])
    dataset = PBDataset(data_file,
                        has_sort_id=sort_id,
                        kafka_dump=kafka_dump,
                        lagrangex_header=False,
                        input_pb_type=PbType.INSTANCE,
                        output_pb_type=PbType.INSTANCE)
    dataset = dataset.batch(batch_size=64, drop_remainder=True)

    def map_fn(tensor):
      return parse_instances(tensor=tensor,
                             fidv1_features=args.fidv1_features,
                             extra_features=args.extra_features,
                             extra_feature_shapes=args.extra_feature_shapes)
    dataset = dataset.map(map_func=map_fn)

    cnt = 0
    for batch in dataset:
      cnt += 1
    self.assertEqual(cnt, 128)
    tf.io.gfile.remove(data_file)

  def test_gen_random_example_file(self):
    args = utils.ParserArgs(
        sparse_features=feature_names,
        extra_features=['sample_rate', 'req_time', 'actions'],
        extra_feature_shapes=[1, 1, 1],
        feature_list=self.feature_list,
        variant_type='example')
    data_file = "example_files.pb"
    sort_id=True
    kafka_dump=True
    utils.gen_random_data_file(data_file, args,
                               sort_id=sort_id, kafka_dump=kafka_dump,
                               actions=[1, 2])
    dataset = PBDataset(data_file,
                        has_sort_id=sort_id,
                        kafka_dump=kafka_dump,
                        input_pb_type=PbType.EXAMPLE,
                        output_pb_type=PbType.EXAMPLE)
    dataset = dataset.batch(batch_size=64, drop_remainder=True)

    def map_fn(tensor):
      return parse_examples(tensor=tensor,
                            sparse_features=args.sparse_features,
                            extra_features=args.extra_features,
                            extra_feature_shapes=args.extra_feature_shapes)
    dataset = dataset.map(map_func=map_fn)

    cnt = 0
    for batch in dataset:
      cnt += 1
    self.assertEqual(cnt, 128)
    tf.io.gfile.remove(data_file)

  def test_gen_random_example_batch_file(self):
    args = utils.ParserArgs(
        sparse_features=feature_names,
        extra_features=['sample_rate', 'req_time', 'actions'],
        extra_feature_shapes=[1, 1, 1],
        feature_list=self.feature_list,
        variant_type='example_batch')
    data_file = "example_batch_files.pb"
    sort_id=True
    kafka_dump=False
    utils.gen_random_data_file(data_file, args,
                               sort_id=sort_id, kafka_dump=kafka_dump,
                               actions=[1, 2])
    dataset = PBDataset(data_file,
                        has_sort_id=sort_id,
                        kafka_dump=kafka_dump,
                        input_pb_type=PbType.EXAMPLEBATCH, 
                        output_pb_type=PbType.EXAMPLE)
    dataset = dataset.batch(batch_size=64, drop_remainder=True)

    def map_fn(tensor):
      return parse_examples(tensor=tensor,
                            sparse_features=args.sparse_features,
                            extra_features=args.extra_features,
                            extra_feature_shapes=args.extra_feature_shapes)
    dataset = dataset.map(map_func=map_fn)

    cnt = 0
    for batch in dataset:
      cnt += 1
    self.assertEqual(cnt, 128)
    tf.io.gfile.remove(data_file)

if __name__ == "__main__":
  tf.test.main()
