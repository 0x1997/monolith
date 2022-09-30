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

import os, time
from copy import deepcopy
from absl import app, logging, flags
from typing import Dict, List, Tuple, get_type_hints

import tensorflow as tf

from monolith.entry import *
from monolith.estimator import EstimatorSpec, Estimator, RunConfig
from monolith.native_training.model_dump.dump_utils import DumpUtils
from monolith.native_training.hooks import ckpt_hooks
from monolith.base_model import MonolithModel, get_sigmoid_loss_and_pred
from monolith.data import PBDataset, PbType, parse_examples
from monolith.model_export.export_context import ExportMode
from monolith.native_training import env_utils
from monolith.native_training.data.parsers import get_default_parser_ctx
from monolith.native_training.model_export.export_context import get_current_export_ctx

FLAGS = flags.FLAGS


class DummpedModel(MonolithModel):

  def __init__(self, params = None, model_dump_path: str = None):
    self.is_dummed = True
    self.dump_utils = DumpUtils()
    self.dump_utils.enable = False
    model_dump_path = model_dump_path or os.path.join(os.path.dirname(__file__), 'model_dump')
    if not tf.io.gfile.exists(model_dump_path):
      model_dump_path = "monolith/native_training/model_dump/test_data/model_dump"
    self.dump_utils.load(model_dump_path)
    model_params = self.dump_utils.restore_params()
    params = model_params.pop('p')
    params.cls = MonolithModel
    super(DummpedModel, self).__init__(params)
    for key, value in model_params.items():
      setattr(self, key, value)
    self.file_name = ''

  def __deepcopy__(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for name, value in self.__dict__.items():
      if name == 'dump_utils':
        result.__dict__[name] = value
      else:
        result.__dict__[name] = deepcopy(value)
    return result

  def input_fn(self, mode) -> 'DatasetV2':
    proto_model = self.dump_utils.get_proto_model(mode)
    graph_helper = self.dump_utils.get_graph_helper(mode)
    ckpt_hooks.disable_iterator_save_restore()
    get_default_parser_ctx().parser_type = proto_model.input_fn.parser_type
    result = graph_helper.import_input_fn(input_conf=proto_model.input_fn, 
                                          file_name=self.file_name)

    pool_tensor_name = proto_model.input_fn.item_pool
    if pool_tensor_name:
      graph = tf.compat.v1.get_default_graph()
      pool = graph.get_tensor_by_name(pool_tensor_name)
      tf.compat.v1.add_to_collection(POOL_KEY, pool)
    return result

  def model_fn(
      self, features: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys):
    proto_model = self.dump_utils.get_proto_model(mode)
    graph_helper = self.dump_utils.get_graph_helper(mode)
    graph = tf.compat.v1.get_default_graph()

    # ragged features
    for fc in proto_model.features:
      expire_time = fc.expire_time if fc.expire_time else None
      shared_name = fc.shared_name if fc.shared_name else None
      self.create_embedding_feature_column(feature_name=fc.feature_name, 
                                           occurrence_threshold=fc.occurrence_threshold, 
                                           expire_time=expire_time,
                                           max_seq_length=fc.max_seq_length,
                                           shared_name=shared_name)

    input_map = {}
    for es_args in proto_model.emb_slices:
      slice_dim = es_args.slice_dim if es_args.slice_dim else None
      initializer = es_args.initializer if es_args.initializer else None
      optimizer = es_args.optimizer if es_args.optimizer else None
      compressor = es_args.compressor if es_args.compressor else None
      group_out_type = es_args.group_out_type if es_args.group_out_type else 'add_n'
      out_type = es_args.out_type if es_args.out_type else None
      embeddings = self.lookup_embedding_slice(features=eval(es_args.features), 
                                               slice_name=es_args.slice_name,
                                               slice_dim=slice_dim,
                                               initializer=initializer,
                                               optimizer=optimizer,
                                               compressor=compressor,
                                               group_out_type=group_out_type,
                                               out_type=out_type)
      for name, ts in zip(es_args.output_tensor_names, embeddings):
        input_map[name] = ts

    # non ragged features
    for name, ts_name in proto_model.model_fn.non_ragged_features.items():
      assert name in features
      if features[name] is not None:
        if ts_name != features[name].name:
          logging.warning(f"{name}: The tensor name {ts_name} in dumped file and " \
            f"that from tensor {features[name].name} are not match!")
        input_map[ts_name] = features[name]
    
    # label 
    label_ts_name = proto_model.model_fn.label
    if label_ts_name is not None and len(label_ts_name) > 0 and label_ts_name not in input_map:
      ts = features.get('label')
      if ts is None:
        try:
          ts = graph.get_tensor_by_name(label_ts_name) 
          input_map[label_ts_name] = ts
        except Exception as e:
          logging.warning(str(e))
      else:
        input_map[label_ts_name] = ts

    model_params = graph_helper.import_model_fn(input_map, proto_model)
    label, loss, pred, head_name, extra_output = model_params

    # add extra losses
    for loss_name in proto_model.model_fn.extra_losses:
      self.add_loss(graph.get_tensor_by_name(loss_name))

    if extra_output:
      for name, outputs in extra_output.items():
        self.add_extra_output(name, outputs)

    export_ctx = get_current_export_ctx()
    if export_ctx:
      for signature in proto_model.signature:
        name = signature.name
        inputs = {ip_key: graph.get_tensor_by_name(value)
          for ip_key, value in signature.inputs.items()}
        outputs = {op_key: graph.get_tensor_by_name(value)
          for op_key, value in signature.outputs.items()}
        export_ctx.add_signature(graph, name, inputs, outputs)

    optimizer = graph_helper.get_optimizer(proto_model)
    return EstimatorSpec(label=label,
                          pred=pred,
                          head_name=head_name,
                          loss=loss,
                          optimizer=optimizer)

  def serving_input_receiver_fn(self):
    proto_model = self.dump_utils.get_proto_model(tf.estimator.ModeKeys.PREDICT)
    graph_helper = self.dump_utils.get_graph_helper(tf.estimator.ModeKeys.PREDICT)
    get_default_parser_ctx().parser_type = proto_model.serving_input_receiver_fn.parser_type
    features, receiver_tensors = graph_helper.import_receiver_fn(
      proto_model.serving_input_receiver_fn)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def main(_):
  env_utils.setup_hdfs_env()
  model = DummpedModel()
  
  config_json = DumpUtils().get_config()
  est_config = RunConfig.from_json(config_json)

  # use cmd line params
  for name, _ in get_type_hints(RunConfig).items():
    default_value = getattr(RunConfig, name)
    if hasattr(FLAGS, name):
      cmd_value = getattr(FLAGS, name)
      if default_value != cmd_value:
        from_dump_value = getattr(est_config, name)
        if from_dump_value != cmd_value:
          setattr(est_config, name, cmd_value)
  est_config.is_local = False
  if not FLAGS.feature_list:
    feature_list = os.path.join(os.getcwd(), 'feature_list.conf')
    if tf.io.gfile.exists(feature_list):
      FLAGS.feature_list = feature_list
    else:
      feature_list = os.path.join(os.path.dirname(__file__), 'feature_list.conf')
      if tf.io.gfile.exists(feature_list):
        FLAGS.feature_list = feature_list
  model.feature_list = FLAGS.feature_list
  
  estimator = Estimator(model, est_config)
  if FLAGS.mode == tf.estimator.ModeKeys.EVAL:
    estimator.evaluate()
  elif FLAGS.mode == tf.estimator.ModeKeys.TRAIN:
    estimator.train()


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  app.run(main)
