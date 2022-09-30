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

from posixpath import split
from monolith.native_training.distributed_serving_ops import remote_predict
from monolith.native_training.utils import with_params
from absl import logging, flags
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from functools import partial
import os, math, time
import hashlib
from typing import Tuple, Dict, Iterable, Union, Optional
import numpy as np

import tensorflow as tf
from tensorflow.estimator.export import ServingInputReceiver
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY

from monolith.core import hyperparams
from monolith.native_training.entry import *
from monolith.native_training.feature import *
from monolith.core.base_layer import get_layer_loss
from monolith.core.hyperparams import update_params

from monolith.native_training import distribution_ops
from monolith.native_training import file_ops
from monolith.native_training import hash_table_ops
from monolith.native_training.native_task_context import get
import monolith.native_training.feature_utils as feature_utils
from monolith.native_training.estimator import EstimatorSpec
from monolith.native_training.embedding_combiners import FirstN
from monolith.native_training.layers import LogitCorrection
from monolith.native_training.native_task import NativeTask, NativeContext
from monolith.native_training.metric import utils as metric_utils
from monolith.native_training.model_export import export_context
from monolith.native_training.model_export.export_context import is_exporting, is_exporting_distributed
from monolith.native_training.data.feature_list import FeatureList, get_feature_name_and_slot
from monolith.native_training.monolith_export import monolith_export
from monolith.native_training.runtime.hash_table import \
    embedding_hash_table_pb2
from monolith.native_training.data.utils import get_slot_feature_name, enable_to_env
from monolith.native_training.utils import add_to_collections
from monolith.native_training.model_dump.dump_utils import DumpUtils
from monolith.native_training.dense_reload_utils import CustomRestoreListener, CustomRestoreListenerKey
from monolith.native_training.layers.utils import dim_size
from monolith.native_training.metric.metric_hook import KafkaMetricHook
from idl.matrix.proto.example_pb2 import OutConfig, OutType, TensorShape
from monolith.native_training.data.datasets import POOL_KEY


FLAGS = flags.FLAGS
dump_utils = DumpUtils(enable=False)

@monolith_export
def get_sigmoid_loss_and_pred(
    name,
    logits,
    label,
    batch_size: int,
    sample_rate: Union[tf.Tensor, float] = 1.0,
    sample_bias: bool = False,
    mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN,
    instance_weight: tf.Tensor = None,
    logit_clip_threshold: Optional[float] = None,
    predict_before_correction: bool = True):
  """对二分类, 基于sigmoid计算loss和predict

  由于负例采样, fast_emit等原因, 需要对logit进进较正, 在get_sigmoid_loss_and_pred会透明地进行

  Args:
    name (:obj:`str`): 名称
    logits (:obj:`tf.Tensor`): 样本logits(无偏logit), 可用于直接predict, 但是不能用于直接计算loss
    label (:obj:`tf.Tensor`): 样本标签
    batch_size (:obj:`int`): 批大小
    sample_rate (:obj:`tf.Tensor`): 负例采样的采样率
    sample_bias (:obj:`bool`): 是否有开启fast_emit
    mode (:obj:`str`): 运行模式, 可以是train/eval/predict等

  """

  logits = tf.reshape(logits, shape=(-1,))
  batch_size = dim_size(logits, 0)
  if mode != tf.estimator.ModeKeys.PREDICT:
    if sample_rate is not None and isinstance(sample_rate, float):
      sample_rate = tf.fill(dims=(batch_size,), value=sample_rate)
    if sample_rate is None:
      sample_rate = tf.fill(dims=(batch_size,), value=1.0)
    src = LogitCorrection(activation=None,
                          sample_bias=sample_bias,
                          name='sample_rate_correction')
    logits_biased = src((logits, sample_rate))
    if predict_before_correction:
      pred = tf.nn.sigmoid(logits, name='{name}_sigmoid_pred'.format(name=name))
    else:
      pred = tf.nn.sigmoid(logits_biased,
                           name='{name}_sigmoid_pred'.format(name=name))

    if logit_clip_threshold is not None:
      assert 0 < logit_clip_threshold < 1
      threshold = math.log((1 - logit_clip_threshold) / logit_clip_threshold)
      logits_biased = tf.clip_by_value(logits_biased, -threshold, threshold)

    if instance_weight is None:
      loss = tf.reduce_sum(
          tf.nn.sigmoid_cross_entropy_with_logits(
              labels=tf.reshape(label, shape=(-1,)),
              logits=logits_biased,
              name='{name}_sigmoid_loss'.format(name=name)))
    else:
      instance_weight = tf.reshape(instance_weight, shape=(-1,))
      loss = tf.reduce_sum(
          tf.multiply(
              tf.nn.sigmoid_cross_entropy_with_logits(
                  labels=tf.reshape(label, shape=(-1,)),
                  logits=logits_biased,
                  name='{name}_sigmoid_loss'.format(name=name)),
              instance_weight))
  else:
    loss = None
    pred = tf.nn.sigmoid(logits, name='{name}_sigmoid_pred'.format(name=name))

  return loss, pred


@monolith_export
def get_softmax_loss_and_pred(name, logits, label, mode):
  """对多分类, 基于softmax计算loss和predict

  Args:
    name (:obj:`str`): 名称
    logits (:obj:`tf.Tensor`): 样本logits
    label (:obj:`tf.Tensor`): 样本标签
    mode (:obj:`str`): 运行模式, 可以是train/eval/predict等

  """

  pred = tf.argmax(tf.nn.softmax(logits,
                                 name='{name}_softmax_pred'.format(name=name)),
                   axis=1)
  if mode != tf.estimator.ModeKeys.PREDICT:
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=label,
        logits=logits,
        name='{name}_softmax_loss'.format(name=name))
  else:
    loss = None

  return loss, pred


class DeepRoughSortBaseModel(object):
  # TODO: auto determine it by candidate number
  def split_shard(self) -> bool:
    return False

  @abstractmethod
  def user_fn(self, features: Dict[str, tf.Tensor],
              mode: tf.estimator.ModeKeys) -> Dict[str, tf.Tensor]:
    raise NotImplementedError("user_fn not implemented")

  @abstractmethod
  def item_fn(self, features: Dict[str, tf.Tensor],
              mode: tf.estimator.ModeKeys):
    raise NotImplementedError("group_fn not implemented")

  @abstractmethod
  def pred_fn(
      self, features: Dict[str, tf.Tensor], user_features: Dict[str, tf.Tensor],
      item_bias: tf.Tensor, item_vec: tf.Tensor, mode: tf.estimator.ModeKeys
  ) -> Union[EstimatorSpec, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    raise NotImplementedError("pred_fn not implemented")

  def _entry_distributed_pred(self, cur_export_ctx,
                              user_features_for_placeholder,
                              head_names: List[str]):
    ps_num = get().num_ps
    item_ids = tf.compat.v1.placeholder(tf.int64, shape=[None], name="item_ids")
    user_feature_names = list(user_features_for_placeholder)
    user_feature_tensors = [
        tf.compat.v1.placeholder(
            user_features_for_placeholder[name].dtype,
            shape=user_features_for_placeholder[name].shape,
            name=name) for name in user_feature_names
    ]
    input_placeholders = dict(zip(user_feature_names, user_feature_tensors))
    input_placeholders["item_ids"] = item_ids
    indices = tf.math.floormod(item_ids, ps_num)
    split_ids = distribution_ops.split_by_indices(indices, item_ids, ps_num)
    for head_name in head_names:
      ps_responses = [None] * ps_num
      for i in range(ps_num):
        ps_pred_result = remote_predict(
            input_tensor_alias=user_feature_names + ["item_ids"],
            input_tensors=user_feature_tensors + [split_ids[i]],
            output_tensor_alias=["preds"],
            task=i,
            old_model_name=f"ps_item_embedding_{i}",
            model_name=f"{get().model_name or ''}:ps_item_embedding_{i}",
            model_version=-1,
            max_rpc_deadline_millis=3000,
            output_types=[tf.float32],
            signature_name=head_name)[0]
        ps_responses[i] = tf.reshape(ps_pred_result, [-1, 1])
      preds = distribution_ops.map_id_to_embedding(split_ids, ps_responses,
                                                   item_ids)
      flat_preds = tf.reshape(preds, [-1])
      signature_name = "distributed_pred" if head_name == "ps_pred" else head_name
      cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(),
                                   signature_name, input_placeholders,
                                   {"preds": flat_preds})

  # predict on entry
  def _entry_pred(self, cur_export_ctx, user_features_for_placeholder,
                  cache_table, mode):
    item_ids = tf.compat.v1.placeholder(tf.int64, shape=[None], name="item_ids")
    user_feature_names = list(user_features_for_placeholder)
    user_feature_tensors = [
        tf.compat.v1.placeholder(
            user_features_for_placeholder[name].dtype,
            shape=user_features_for_placeholder[name].shape,
            name=name) for name in user_feature_names
    ]

    # broadcast the user feature to the dim of item_ids
    # for example
    # user_feature = [[1,2,3,4,5]], shape(1,5)
    # item_ids = [9,8,7], shape(3)
    # then tile_args = [3,1], tile_res = [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]], shape(3,5)
    user_feature_tensors_tiled = []
    for tensor in user_feature_tensors:
      tile_args = tf.concat(
          [tf.shape(item_ids)[:1],
           np.ones(len(tensor.shape) - 1)], axis=0)
      user_feature_tensors_tiled.append(tf.tile(tensor, tile_args))

    input_placeholders = dict(zip(user_feature_names, user_feature_tensors))
    input_placeholders["item_ids"] = item_ids

    embeddings = cache_table.lookup(item_ids)
    item_bias = embeddings[:, 0]
    item_vec = embeddings[:, 1:]
    local_spec = self.pred_fn(
        {},  # no extra features for serving
        dict(zip(user_feature_names, user_feature_tensors_tiled)),
        item_bias,
        item_vec,
        mode)
    if isinstance(local_spec, EstimatorSpec):
      preds = local_spec.pred
    elif isinstance(local_spec, (tuple, list)):
      preds = local_spec[2]
    else:
      raise Exception("EstimatorSpec Error!")

    flags = tf.reshape(tf.reduce_sum(tf.math.abs(embeddings), axis=1), [-1])
    if isinstance(preds, dict):
      for name, pred in preds.items():
        assert isinstance(pred, tf.Tensor)
        flat_pred = tf.reshape(pred, [-1])
        flat_pred = tf.where(flags > 0, flat_pred,
                             tf.ones_like(flat_pred, dtype=tf.float32) * -10000)
        cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(), name,
                                     input_placeholders, {"preds": flat_pred})
    else:
      assert isinstance(preds, tf.Tensor)
      flat_preds = tf.reshape(preds, [-1])
      flat_pred = tf.where(flags > 0, flat_preds,
                           tf.ones_like(flat_preds, dtype=tf.float32) * -10000)
      cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(),
                                   "entry_pred", input_placeholders,
                                   {"preds": flat_preds})

  # predict on PS
  def _ps_pred(self, cur_export_ctx, user_features_for_placeholder, cache_table,
               mode):
    item_ids = tf.compat.v1.placeholder(tf.int64, shape=[None], name="item_ids")
    user_feature_names = list(user_features_for_placeholder)
    user_feature_tensors = [
        tf.compat.v1.placeholder(
            user_features_for_placeholder[name].dtype,
            shape=user_features_for_placeholder[name].shape,
            name=name) for name in user_feature_names
    ]

    # broadcast the user feature to the dim of item_ids
    user_feature_tensors_tiled = []
    for tensor in user_feature_tensors:
      tile_args = tf.concat(
          [tf.shape(item_ids)[:1],
           np.ones(len(tensor.shape) - 1)], axis=0)
      user_feature_tensors_tiled.append(tf.tile(tensor, tile_args))

    input_placeholders = dict(zip(user_feature_names, user_feature_tensors))
    input_placeholders["item_ids"] = item_ids

    embeddings = cache_table.lookup(item_ids)
    item_bias = embeddings[:, 0]
    item_vec = embeddings[:, 1:]
    local_spec = self.pred_fn(
        {},  # no extra features for serving
        dict(zip(user_feature_names, user_feature_tensors_tiled)),
        item_bias,
        item_vec,
        mode)
    if isinstance(local_spec, EstimatorSpec):
      ps_pred = local_spec.pred
    elif isinstance(local_spec, (tuple, list)):
      ps_pred = local_spec[2]
    else:
      raise Exception("EstimatorSpec Error!")

    flags = tf.reshape(tf.reduce_sum(tf.math.abs(embeddings), axis=1), [-1])
    if isinstance(ps_pred, dict):
      for name, pred in ps_pred.items():
        assert isinstance(pred, tf.Tensor)
        pred = tf.reshape(pred, [-1])
        pred = tf.where(flags > 0, pred,
                        tf.ones_like(pred, dtype=tf.float32) * -10000)
        cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(), name,
                                     input_placeholders, {"preds": pred})
    else:
      assert isinstance(ps_pred, tf.Tensor)
      ps_pred = tf.reshape(ps_pred, [-1])
      ps_pred = tf.where(flags > 0, ps_pred,
                         tf.ones_like(ps_pred, dtype=tf.float32) * -10000)
      cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(), "ps_pred",
                                   input_placeholders, {"preds": ps_pred})
    cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(),
                                 "lookup_item_embedding",
                                 {"item_ids": item_ids},
                                 {"embeddings": embeddings})
    if isinstance(ps_pred, dict):
      return list(ps_pred.keys())
    else:
      return ['ps_pred']

  # interface for lookup embedding, useful for debug/trace
  def _entry_lookup_embedding(self, cur_export_ctx, cache_table):
    item_ids = tf.compat.v1.placeholder(tf.int64, shape=[None], name="item_ids")
    input_placeholders = {}
    input_placeholders["item_ids"] = item_ids

    embeddings = cache_table.lookup(item_ids)

    cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(),
                                 "lookup_embedding", input_placeholders,
                                 {"embeddings": embeddings})

  # interface for lookup embedding(same with the above)
  # just route the request to PS
  def _entry_call_ps_lookup_embedding(self, cur_export_ctx, dim):
    ps_num = get().num_ps
    item_ids = tf.compat.v1.placeholder(tf.int64, shape=[None], name="item_ids")
    input_placeholders = {}
    input_placeholders["item_ids"] = item_ids
    indices = tf.math.floormod(item_ids, ps_num)
    split_ids = distribution_ops.split_by_indices(indices, item_ids, ps_num)
    ps_responses = [None] * ps_num
    for i in range(ps_num):
      ps_pred_result = remote_predict(["item_ids"], [split_ids[i]],
                                      ["embeddings"],
                                      task=i,
                                      old_model_name=f"ps_{i}",
                                      model_name=f"{get().model_name or ''}:ps_{i}",
                                      model_version=-1,
                                      max_rpc_deadline_millis=3000,
                                      output_types=[tf.float32],
                                      signature_name="lookup_item_embedding")[0]
      ps_responses[i] = tf.reshape(ps_pred_result, [-1, dim])
    outputs = distribution_ops.map_id_to_embedding(split_ids, ps_responses,
                                                   item_ids)
    cur_export_ctx.add_signature(tf.compat.v1.get_default_graph(),
                                 "lookup_embedding", input_placeholders,
                                 {"embeddings": outputs})

  def _entry_metadata(self, cur_export_ctx):
    ps_num = get().num_ps
    cur_export_ctx.add_signature(
        tf.compat.v1.get_default_graph(), "metadata", {},
        {"ps_num": tf.constant(ps_num, dtype=tf.int32)})

  def _model_fn(
      self, features: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if not is_exporting_distributed():
      user_features = self.user_fn(features, mode)
      item_bias, item_vec = self.item_fn(features, mode)
      return self.pred_fn(features, user_features, item_bias, item_vec, mode)
    else:
      cur_ctx = export_context.get_current_export_ctx()
      ps_num = get().num_ps
      user_features = self.user_fn(features, mode)
      item_bias, item_vec = self.item_fn(features, mode)
      # entry signature 1: fetch user tensors
      self.add_extra_output(name="fetch_user", outputs=user_features)
      # entry signature 2: fetch item tensor
      self.add_extra_output(name="fetch_item",
                            outputs={
                                "item_bias": item_bias,
                                "item_vec": item_vec
                            })

      def create_hash_table(dim):
        table_config = embedding_hash_table_pb2.EmbeddingHashTableConfig()
        table_config.cuckoo.SetInParent()
        segment = table_config.entry_config.segments.add()
        segment.dim_size = dim
        segment.opt_config.sgd.SetInParent()
        segment.init_config.zeros.SetInParent()
        segment.comp_config.fp32.SetInParent()
        config_instance = entry.HashTableConfigInstance(table_config, [1.0])
        table = hash_table_ops.hash_table_from_config(
            config_instance, name_suffix="cached_item_embeddings")
        # disable embedding sharing
        table.export_share_embedding = False
        return table

      # entry signature 3: metadata
      self._entry_metadata(cur_ctx)

      # entry signature 4: pred on PS or Entry
      if self.split_shard():
        heads = None
        # ps graph: call pred_fn with user_features and item_ids
        for i in range(ps_num):
          with cur_ctx.dense_sub_graph(
              f"ps_item_embedding_{i}").as_default() as g:
            cache_table = create_hash_table(item_vec.shape[-1] + 1)
            # ps signature 1: pred
            cur_heads = self._ps_pred(cur_ctx, user_features, cache_table, mode)
            if heads is None:
              heads = cur_heads
            else:
              assert len(cur_heads) == len(heads)
              assert len(set(cur_heads) - set(heads)) == 0
              assert len(set(heads) - set(cur_heads)) == 0
            # add a suffix to indicate the exporting is half done
            g.export_suffix = "_dense"
        # entry graph: route the request to PS
        assert heads is not None
        self._entry_distributed_pred(cur_ctx, user_features, heads)
        self._entry_call_ps_lookup_embedding(cur_ctx, item_vec.shape[-1] + 1)
      else:
        with cur_ctx.dense_sub_graph(
            f"entry_item_embedding_0").as_default() as g:
          cache_table = create_hash_table(item_vec.shape[-1] + 1)
          # entry signature 1: pred
          self._entry_pred(cur_ctx, user_features, cache_table, mode)
          self._entry_lookup_embedding(cur_ctx, cache_table)
          g.export_suffix = "_dense"
      # return dummy predict value
      return None, None, tf.constant([0.0], dtype=tf.float32)


@monolith_export
class MonolithBaseModel(NativeTask, ABC):
  """模型开发的基类"""

  @classmethod
  def params(cls):
    p = super(MonolithBaseModel, cls).params()
    p.define("output_path", None, "The output path of predict/eval")
    p.define("output_fields", None, "The output fields")
    p.define("delimiter", '\t', "The delimiter of output file")
    p.define('dense_weight_decay', 0.001, 'dense_weight_decay')
    p.define("clip_norm", 250.0, "float, clip_norm")
    p.define('file_name', '', 'the test input file name')
    p.define('enable_grads_and_vars_summary', False,
             'enable_grads_and_vars_summary')

    # others
    p.define('default_occurrence_threshold', 5, 'int')

    return p

  def __init__(self, params):
    super(MonolithBaseModel, self).__init__(params)
    enable_to_env()
    self.fs_dict = {}
    self.fc_dict = {}
    # feature_name -> slice_name -> FeatureSlice(feature_slot, start, end)
    self.slice_dict = {}
    self._layout_dict = {}
    self.valid_label_threshold = 0
    self._occurrence_threshold = {}

  def __getattr__(self, name):
    if "p" in self.__dict__:
      if hasattr(self.p, name):
        return getattr(self.p, name)
      elif name == 'batch_size':
        if self.p.mode == tf.estimator.ModeKeys.EVAL:
          return self.p.eval.per_replica_batch_size
        else:
          return self.p.train.per_replica_batch_size

    if (hasattr(type(self), name) and
        isinstance(getattr(type(self), name), property)):
      return getattr(type(self), name).fget(self)
    else:
      return super(MonolithBaseModel, self).__getattr__(name)

  def __setattr__(self, key, value):
    if 'p' in self.__dict__:
      if hasattr(self.p, key):
        setattr(self.p, key, value)
        return value
      elif key == 'batch_size':
        self.p.eval.per_replica_batch_size = value
        self.p.train.per_replica_batch_size = value
        return value

    super(MonolithBaseModel, self).__setattr__(key, value)
    return value

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

  def _get_file_ops(self, features, pred):
    assert self.p.output_fields is not None
    output_path = os.path.join(self.p.output_path,
                               f"part-{get().worker_index:05d}")
    op_file = file_ops.WritableFile(output_path)
    op_fields = [features[field] for field in self.p.output_fields.split(',')]
    if isinstance(pred, (tuple, list)):
      op_fields.extend(pred)
    else:
      op_fields.append(pred)
    fmt = self.p.delimiter.join(["{}"] * len(op_fields)) + "\n"
    result = tf.map_fn(fn=lambda t: tf.strings.format(fmt, t),
                       elems=tuple(op_fields),
                       fn_output_signature=tf.string)
    write_op = op_file.append(tf.strings.reduce_join(result))
    return op_file, write_op

  def _get_real_mode(self, mode: tf.estimator.ModeKeys):
    if mode == tf.estimator.ModeKeys.PREDICT:
      return mode
    elif mode == tf.estimator.ModeKeys.TRAIN:
      return self.mode
    else:
      raise ValueError('model error!')

  def is_fused_layout(self) -> bool:
    return self.ctx.layout_factory is not None
  
  def instantiate(self):
    """实例化对像"""
    return self

  def add_loss(self, losses):
    """用于追加辅助loss, 如layer loss等

    Args:
      losses (:obj:`List[tf.Tensor]`): 辅助loss列表

    """

    if losses:
      if isinstance(losses, (list, tuple)):
        self.losses.extend(losses)
      else:
        self.losses.append(losses)

  @property
  def losses(self):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__losses'):
      return getattr(graph, '__losses')
    else:
      setattr(graph, '__losses', [])
      return graph.__losses

  @losses.setter
  def losses(self, losses):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__losses'):
      graph.__losses = losses
    else:
      setattr(graph, '__losses', losses)

  @property
  def _global_step(self):
    return tf.compat.v1.train.get_or_create_global_step()

  @property
  def _training_hooks(self):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__training_hooks'):
      return getattr(graph, '__training_hooks')
    else:
      setattr(graph, '__training_hooks', [])
      return graph.__training_hooks

  @_training_hooks.setter
  def _training_hooks(self, hooks):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__training_hooks'):
      graph.__training_hooks = hooks
    else:
      setattr(graph, '__training_hooks', hooks)

  def clean(self):
    # update fs_dict, fc_dict, slice_dict
    self.fs_dict = {}
    self.fc_dict = {}
    self.slice_dict = {}  # slot_id -> Dict[slot_id, slice]
    self.valid_label_threshold = 0
    self._occurrence_threshold = {}

  def create_input_fn(self, mode):
    """生成input_fn"""
    return partial(self.input_fn, mode)

  def create_model_fn(self):
    """生成model_fn"""
    self.clean()

    def model_fn_internal(
        features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys,
        config: tf.estimator.RunConfig) -> tf.estimator.EstimatorSpec:

      real_mode = self._get_real_mode(mode)
      local_spec = self.model_fn(features, real_mode)

      # get label, loss, pred and head_name from model_fn result
      if isinstance(local_spec, EstimatorSpec):
        label, loss, pred = local_spec.label, local_spec.loss, local_spec.pred
        if isinstance(pred, dict):
          assert label is None or isinstance(label, dict)
          head_name, pred = list(zip(*pred.items()))
        else:
          head_name = local_spec.head_name or self.metrics.deep_insight_target.split(
              ',')
        is_classification = local_spec.classification
      elif isinstance(local_spec, (tuple, list)):
        label, loss, pred = local_spec
        if isinstance(pred, dict):
          assert label is None or isinstance(label, dict)
          head_name, pred = list(zip(*pred.items()))
        else:
          head_name = self.metrics.deep_insight_target
        assert head_name is not None
        is_classification = True
        logging.warning(
            'if this is not a classification task, pls. return EstimatorSpec in model_fn and specify it'
        )
      else:
        raise Exception("EstimatorSpec Error!")

      # check label/pred/head_name
      if isinstance(pred, (list, tuple, dict)):
        assert isinstance(head_name, (list, tuple))
        assert isinstance(pred, (list, tuple))
        if label is not None:
          assert len(head_name) == len(label)
          assert len(label) == len(pred)
      else:
        if isinstance(head_name, (list, tuple)):
          assert len(head_name) == 1
          head_name = head_name[0]
        assert isinstance(head_name, str)
        if label is not None:
          assert isinstance(label, tf.Tensor)
        if isinstance(pred, (list, tuple)):
          assert len(pred) == 1
          pred = pred[0]
          assert isinstance(pred, tf.Tensor)

      dump_utils.add_model_fn(self, mode, features, label, loss, pred, head_name)

      if self.losses:
        loss = loss + tf.add_n(self.losses)

      if real_mode == tf.estimator.ModeKeys.PREDICT:
        if isinstance(pred, (list, tuple)):
          assert isinstance(head_name,
                            (list, tuple)) and len(pred) == len(head_name)
          predictions = dict(zip(head_name, pred))
        else:
          predictions = pred

        if is_exporting() or self.p.output_path is None:
          spec = tf.estimator.EstimatorSpec(real_mode,
                                            predictions=predictions,
                                            training_hooks=self._training_hooks)
        else:
          op_file, write_op = self._get_file_ops(features, pred)
          close_hook = file_ops.FileCloseHook([op_file])
          with tf.control_dependencies(control_inputs=[write_op]):
            if isinstance(pred, dict):
              predictions = {k: tf.identity(v) for k, v in predictions.items()}
            else:
              predictions = tf.identity(predictions)
            spec = tf.estimator.EstimatorSpec(mode,
                                              training_hooks=[close_hook] +
                                              self._training_hooks,
                                              predictions=predictions)
        if is_exporting() and self._export_outputs:
          self._export_outputs.update(spec.export_outputs)
          return spec._replace(export_outputs=self._export_outputs)
        else:
          return spec

      train_ops = []
      targets, labels_list, preds_list = [], [], []
      if isinstance(pred, (list, tuple, dict)):
        assert isinstance(label,
                          (list, tuple, dict)) and len(pred) == len(label)
        assert isinstance(head_name,
                          (list, tuple)) and len(pred) == len(head_name)
        if isinstance(is_classification, (tuple, list, dict)):
          assert len(pred) == len(is_classification)
        else:
          is_classification = [is_classification] * len(pred)
        print("Debug. is_classification: ", is_classification)

        for i, name in enumerate(head_name):
          label_tensor = label[i] if isinstance(label,
                                                (list, tuple)) else label[name]
          pred_tensor = pred[i] if isinstance(pred,
                                              (list, tuple)) else pred[name]
          head_classification = is_classification[i] if isinstance(
              is_classification, (list, tuple)) else is_classification[name]

          targets.append(name)
          labels_list.append(label_tensor)
          preds_list.append(pred_tensor)

          mask = tf.greater_equal(label_tensor, self.valid_label_threshold)
          l = tf.boolean_mask(label_tensor, mask)
          p = tf.boolean_mask(pred_tensor, mask)

          if head_classification:
            auc_per_core, auc_update_op = tf.compat.v1.metrics.auc(
                labels=l, predictions=p, name=name)
            tf.compat.v1.summary.scalar("{}_auc".format(name), auc_per_core)
            train_ops.append(auc_update_op)
          else:
            mean_squared_error, mse_update_op = tf.compat.v1.metrics.mean_squared_error(
                labels=l, predictions=p, name=name)
            tf.compat.v1.summary.scalar("{}_mse".format(name),
                                        mean_squared_error)
            train_ops.append(mse_update_op)
      else:
        targets.append(head_name)
        labels_list.append(label)
        preds_list.append(pred)

        if is_classification:
          auc_per_core, auc_update_op = tf.compat.v1.metrics.auc(
              labels=label, predictions=pred, name=head_name)
          tf.compat.v1.summary.scalar(f"{head_name}_auc", auc_per_core)
          train_ops.append(auc_update_op)
        else:
          mean_squared_error, mse_update_op = tf.compat.v1.metrics.mean_squared_error(
              labels=label, predictions=pred, name=head_name)
          tf.compat.v1.summary.scalar("{}_mse".format(head_name),
                                      mean_squared_error)
          train_ops.append(mse_update_op)

      enable_metrics = self.metrics.enable_kafka_metrics or self.metrics.enable_deep_insight
      if enable_metrics and self.metrics.deep_insight_sample_ratio > 0:
        model_name = self.metrics.deep_insight_name
        sample_ratio = self.metrics.deep_insight_sample_ratio
        extra_fields_keys = self.metrics.extra_fields_keys

        deep_insight_op = metric_utils.write_deep_insight(
            features=features,
            sample_ratio=self.metrics.deep_insight_sample_ratio,
            labels=label,
            preds=pred,
            model_name=model_name or "model_name",
            target=self.metrics.deep_insight_target,
            targets=targets,
            labels_list=labels_list,
            preds_list=preds_list,
            extra_fields_keys=extra_fields_keys,
            enable_kafka_metrics=self.metrics.enable_kafka_metrics)
        logging.info("model_name: {}, target: {}.".format(
            model_name, self.metrics.deep_insight_target))
        train_ops.append(deep_insight_op)
        tf.compat.v1.add_to_collection("deep_insight_op", deep_insight_op)
        if self.metrics.enable_kafka_metrics:
          self.add_training_hook(KafkaMetricHook(deep_insight_op))
        logging.info("model_name: {}, target {}".format(model_name, head_name))

      if real_mode == tf.estimator.ModeKeys.EVAL:
        if is_exporting() or self.output_path is None:
          if isinstance(pred, (list, tuple)):
            train_ops.extend(pred)
          else:
            train_ops.append(pred)
          return tf.estimator.EstimatorSpec(mode,
                                            loss=loss,
                                            train_op=tf.group(train_ops),
                                            training_hooks=self._training_hooks)
        else:
          op_file, write_op = self._get_file_ops(features, pred)
          close_hook = file_ops.FileCloseHook([op_file])
          with tf.control_dependencies(control_inputs=[write_op]):
            if isinstance(pred, (list, tuple)):
              train_ops.extend([tf.identity(p) for p in pred])
            else:
              train_ops.append(tf.identity(pred))
            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=tf.group(train_ops),
                                              training_hooks=[close_hook] +
                                              self._training_hooks)
      else:  # training
        if hasattr(local_spec, 'optimizer'):
          dense_optimizer = local_spec.optimizer
        elif hasattr(self, '_default_dense_optimizer'):
          dense_optimizer = self._default_dense_optimizer
        else:
          raise Exception("dense_optimizer not found!")
        dump_utils.add_optimizer(dense_optimizer)

        if self.is_fused_layout():
          train_ops.append(feature_utils.apply_gradients(
            self.ctx, dense_optimizer, loss,
            clip_type=feature_utils.GradClipType.ClipByGlobalNorm,
            clip_norm=self.clip_norm,
            dense_weight_decay=self.dense_weight_decay,
            global_step=self._global_step))
        else:
          train_ops.append(
              feature_utils.apply_gradients_with_var_optimizer(
                  self.ctx,
                  self.fc_dict.values(),
                  dense_optimizer,
                  loss,
                  clip_type=feature_utils.GradClipType.ClipByGlobalNorm,
                  clip_norm=self.clip_norm,
                  dense_weight_decay=self.dense_weight_decay,
                  global_step=self._global_step,
                  grads_and_vars_summary=self.enable_grads_and_vars_summary))

        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=tf.group(train_ops),
                                          training_hooks=self._training_hooks)

    return model_fn_internal

  def create_serving_input_receiver_fn(self):
    """生在Serving数据流, serving_input_receiver_fn"""
    return dump_utils.record_receiver(self.serving_input_receiver_fn)

  @abstractmethod
  def input_fn(self, mode: tf.estimator.ModeKeys) -> DatasetV2:
    """抽象方法, 定义数据流

    Args:
      mode (:obj:`str`): 训练模式, train/eval/predict等

    Returns:
      DatasetV2, TF数据集

    """

    raise NotImplementedError('input_fn() not Implemented')

  @abstractmethod
  def model_fn(
      self, features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys
  ) -> Union[EstimatorSpec, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    """抽象方法, 定义模型

    Args:
      features (:obj:`Dict[str, tf.Tensor]`): 特征
      mode (:obj:`str`): 训练模式, train/eval/predict等

    Returns:
      Union[EstimatorSpec, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]], 可以是tuple, 包括(loss, label, predict),
                                                                    也可以是EstimatorSpec
    """

    raise NotImplementedError('generate_model() not Implemented')

  @abstractmethod
  def serving_input_receiver_fn(self) -> ServingInputReceiver:
    """Serving数据流, 训练数据流与Serving数据流或能不一样

    Returns:
      ServingInputReceiver

    """

    raise NotImplementedError('serving_input_receiver_fn() not Implemented')

  @property
  def _export_outputs(self):
    graph = tf.compat.v1.get_default_graph()
    if hasattr(graph, '__export_outputs'):
      return getattr(graph, '__export_outputs')
    else:
      setattr(graph, '__export_outputs', {})
      return graph.__export_outputs

  def add_extra_output(self, name: str, outputs: Union[tf.Tensor,
                                                       Dict[str, tf.Tensor]]):
    """如果有出多输出, 可以用add_extra_output, 每个输出会成为Serving中的一个Signature

    Args:
      name (:obj:`str`): 签名的名称
      outputs (:obj:`Union[tf.Tensor, Dict[str, tf.Tensor]]`): 输出, 可以是一个Tensor, 也可以是一个Dict[str, tf.Tensor]

    """

    add_to_collections('signature_name', name)
    if is_exporting():
      exported_outputs = self._export_outputs
      if name not in exported_outputs:
        exported_outputs[name] = tf.estimator.export.PredictOutput(outputs)
      else:
        raise KeyError("key {name} exists!".format(name))

  def add_training_hook(self, hook):
    if isinstance(hook, KafkaMetricHook):
      if any(isinstance(h, KafkaMetricHook) for h in self._training_hooks):
        return
    self._training_hooks.append(hook)

  def add_layout(self, name: str, slice_list: list, out_type: str, shape_list: list):
    if out_type == 'concat':
      out_conf = OutConfig(out_type=OutType.CONCAT)
    elif out_type == 'stack':
      out_conf = OutConfig(out_type=OutType.STACK)
    elif out_type == 'addn':
      out_conf = OutConfig(out_type=OutType.ADDN)
    else:
      out_conf = OutConfig(out_type=OutType.NONE)

    for slice_conf in slice_list:
      slice_config = out_conf.slice_configs.add()
      if len(slice_conf.feature_slot.get_feature_columns()) > 1:
        raise Exception("There are multi feature columns on a slot, not support yet!")
      slice_config.feature_name = slice_conf.feature_slot.name
      slice_config.start = slice_conf.start
      slice_config.end = slice_conf.end

    for shape in shape_list:
      shape_dims = out_conf.shape.add()
      for i, dim in enumerate(shape):
        if i == 0:
          shape_dims.dims.append(-1)
        else:
          if isinstance(dim, int):
            shape_dims.dims.append(dim)
          else:
            assert hasattr(dim, 'value')
            shape_dims.dims.append(dim.value)

    self._layout_dict[name] = out_conf

  @property
  def layout_dict(self):
    return self._layout_dict

  @layout_dict.setter
  def layout_dict(self, layouts):
    self._layout_dict = layouts


@monolith_export
class MonolithModel(MonolithBaseModel):
  '''模型开发的基类

  Args:
      params (:obj:`Params`): 配置参数, 默认为None
  '''

  @classmethod
  def params(cls):
    p = super(MonolithModel, cls).params()
    p.define("feature_list", None, "The feature_list conf file.")
    return p

  def __init__(self, params=None):
    params = params or type(self).params()
    super(MonolithModel, self).__init__(params)
    dump_utils.enable = FLAGS.enable_model_dump

  def _get_fs_conf(self, shared_name: str, slot: int, occurrence_threshold: int,
                   expire_time: int) -> FeatureSlotConfig:
    return FeatureSlotConfig(name=shared_name,
                             has_bias=False,
                             slot_id=slot,
                             occurrence_threshold=occurrence_threshold,
                             expire_time=expire_time)

  def _embedding_slice_lookup(self, fc: Union[str,
                                              FeatureColumn], slice_name: str,
                              slice_dim: int, initializer: Initializer,
                              optimizer: Optimizer, compressor: Compressor,
                              learning_rate_fn,
                              slice_list: list) -> FeatureSlice:
    assert not self.is_fused_layout()
    if isinstance(fc, str):
      fc = self.fc_dict[fc]

    feature_slot = fc.feature_slot
    feature_name = fc.feature_name

    if feature_name in self.slice_dict:
      if slice_name in self.slice_dict[feature_name]:
        fc_slice = self.slice_dict[feature_name][slice_name]
      else:
        fc_slice = feature_slot.add_feature_slice(slice_dim, initializer,
                                                  optimizer, compressor,
                                                  learning_rate_fn)
        self.slice_dict[feature_name][slice_name] = fc_slice
    else:
      fc_slice = feature_slot.add_feature_slice(slice_dim, initializer,
                                                optimizer, compressor,
                                                learning_rate_fn)
      self.slice_dict[feature_name] = {slice_name: fc_slice}

    slice_list.append(fc_slice)
    return fc.embedding_lookup(fc_slice)

  @dump_utils.record_feature
  def create_embedding_feature_column(self,
                                      feature_name,
                                      occurrence_threshold: int = None,
                                      expire_time: int = 36500,
                                      max_seq_length: int = 0,
                                      shared_name: str = None) -> FeatureColumn:
    """创建嵌入特征列(embedding feature column)

    Args:
      feature_name (:obj:`Any`): 特征列的名字
      occurrence_threshold (:obj:`int`): 用于低频特征过滤, 如果出现次数小于`occurrence_threshold`, 则这个特征将大概率不会进入模型
      expire_time (:obj:`int`): 特征过期时间, 如果一个特征在`expire_time`之内没有更新了, 则这个特征可能从hash表中移除
      max_seq_length (:obj:`int`): 如果设为0, 表示非序列特征, 如果设为正数, 则表示序列特征的长度
      shared_name (:obj:`str`): 共享embedding. 如果本feature与另一个feature共享embedding, 则可以将被共享feature设为`shared_name`

    Returns:
     FeatureColumn, 特征列

    """

    feature_name, slot = get_feature_name_and_slot(feature_name)

    if feature_name in self.fc_dict:
      return self.fc_dict[feature_name]
    else:
      if shared_name is not None and len(shared_name) > 0:
        if shared_name in self.fs_dict:
          fs = self.fs_dict[shared_name]
        elif shared_name in self.fc_dict:
          fs = self.fc_dict[shared_name].feature_slot
        else:
          try:
            feature_list = FeatureList.parse()
            shared_slot = feature_list[shared_name].slot
            shared_fs = self.ctx.create_feature_slot(
                self._get_fs_conf(shared_name, shared_slot,
                                  occurrence_threshold, expire_time))
            self.fs_dict[shared_name] = shared_fs
            fs = shared_fs
          except:
            raise Exception(
                f"{feature_name} shared embedding with {shared_name}, so {shared_name} should create first!"
            )
      else:
        fs = self.ctx.create_feature_slot(
            self._get_fs_conf(feature_name, slot, occurrence_threshold,
                              expire_time))
      if max_seq_length > 0:
        combiner = FeatureColumn.first_n(max_seq_length)
      else:
        combiner = FeatureColumn.reduce_sum()
      fc = FeatureColumn(fs, feature_name, combiner=combiner)
      self.fc_dict[feature_name] = fc
      return fc

  @dump_utils.record_slice
  def lookup_embedding_slice(self,
                             features,
                             slice_name,
                             slice_dim=None,
                             initializer: Initializer = None,
                             optimizer: Optimizer = None,
                             compressor: Compressor = None,
                             learning_rate_fn=None,
                             group_out_type: str = 'add_n',
                             out_type: str = None) -> tf.Tensor:
    """Monolith中embedding是分切片的, 每个切片可以有独立的初始化器, 优化器, 压缩器, 学习率等. 切片的引入使Embedding更加强大. 如某些情况
    下要共享Embedding, 另一些情况下要独立Embedding, 与一些域交叉要用一种Embedding, 与另一些域交叉用另一种Embedding等. 切片的引入可以方便
    解上以上问题. 切片与完整Embedding的关系由Monolith自动维护, 对用户透明.

    Args:
      slice_name (:obj:`str`): 切片名称
      features (:obj:`List[str], Dict[str, int]`): 支持三种形式
        1) 特征名列表, 此时每个切片的长度相同, 由`slice_dim`确定, 不能为None
        2) 特征 (特征名, 切片长度) 列表, 此时每个切片的长度可以不同, 全局的`slice_dim`必须为None
        3) 特征字典, 特征名 -> 切片长度, 此时每个切片的长度可以不同, 全局的`slice_dim`必须为None
      slice_dim (:obj:`int`): 切片长度
      initializer (:obj:`Initializer`): 切片的初始化器, Monolith中的初始化器,  不能是TF中的
      optimizer (:obj:`Optimizer`): 切片的优化器, Monolith中的优化器,  不能是TF中的
      compressor (:obj:`Compressor`): 切片的压缩器, 用于在Servering模型加载时将模型压缩
      learning_rate_fn (:obj:`tf.Tensor`): 切片的学习率

    """
    concat = ",".join(sorted(map(str, features)))
    layout_name = f'{slice_name}_{hashlib.md5(concat.encode()).hexdigest()}'
    if self.is_fused_layout():
      if isinstance(features, (list, tuple)) and isinstance(slice_dim, int):
        if all(isinstance(ele, (tuple, list)) for ele in features):
          raise ValueError("group pool is not support when fused_layout")
      return self.ctx.layout_factory.get_layout(layout_name)

    feature_embeddings, slice_list = [], []
    if isinstance(features, dict):
      for fc_name, sdim in features.items():
        fc_name, _ = get_feature_name_and_slot(fc_name)
        feature_embeddings.append(
            self._embedding_slice_lookup(fc_name, slice_name, sdim, initializer,
                                         optimizer, compressor,
                                         learning_rate_fn, slice_list))
    elif isinstance(features, (list, tuple)) and isinstance(slice_dim, int):
      if all(isinstance(ele, (str, int, FeatureColumn)) for ele in features):
        # a list of feature with fixed dim
        for fc_name in features:
          fc_name, _ = get_feature_name_and_slot(fc_name)
          feature_embeddings.append(
              self._embedding_slice_lookup(fc_name, slice_name, slice_dim,
                                           initializer, optimizer, compressor,
                                           learning_rate_fn, slice_list))
      elif all(isinstance(ele, (tuple, list)) for ele in features):
        assert group_out_type in {'concat', 'add_n'}
        for group_name in features:
          assert all(isinstance(ele, int) for ele in group_name)
          local_embeddings = []
          for fc_name in group_name:
            fc_name, _ = get_feature_name_and_slot(fc_name)
            local_embeddings.append(
                self._embedding_slice_lookup(fc_name, slice_name, slice_dim,
                                            initializer, optimizer, compressor,
                                            learning_rate_fn, slice_list))
          if group_out_type == 'add_n':
            feature_embeddings.append(tf.add_n(local_embeddings))
          else:
            feature_embeddings.append(tf.concat(local_embeddings, axis=1))
      else:
        raise ValueError("ValueError for features")
    elif isinstance(features, (list, tuple)):
      if all([
          isinstance(ele, (tuple, list)) and len(ele) == 2 for ele in features
      ]):
        for fc_name, sdim in features:
          fc_name, _ = get_feature_name_and_slot(fc_name)
          feature_embeddings.append(
              self._embedding_slice_lookup(fc_name, slice_name, sdim,
                                           initializer, optimizer, compressor,
                                           learning_rate_fn, slice_list))
      else:
        raise ValueError("ValueError for features")
    else:
      raise ValueError("ValueError for features")

    if out_type is None:
      shape_list = [emb.shape for emb in feature_embeddings]
      self.add_layout(layout_name, slice_list, out_type, shape_list)
      return feature_embeddings
    else:
      assert out_type in {'concat', 'stack', 'add_n', 'addn'}
      if out_type == 'concat':
        out = tf.concat(feature_embeddings, axis=1, name=layout_name)
        self.add_layout(layout_name, slice_list, out_type, shape_list=[out.shape])
        return out
      elif out_type == 'stack':
        out = tf.stack(feature_embeddings, axis=1, name=layout_name)
        self.add_layout(layout_name, slice_list, out_type, shape_list=[out.shape])
        return out
      else:
        out = tf.add_n(feature_embeddings, name=layout_name)
        self.add_layout(layout_name, slice_list, 'addn', shape_list=[out.shape])
        return out


class NativeModelV2(MonolithBaseModel):

  @classmethod
  def params(cls):
    p = super(NativeModelV2, cls).params()
    # for bias opt
    p.define('bias_opt_learning_rate', 0.01, 'float')
    p.define('bias_opt_beta', 1.0, 'float')
    p.define('bias_l1_regularization', 1, 'float')
    p.define('bias_l2_regularization', 1.0, 'float')

    # for vector opt
    p.define('vec_opt_learning_rate', 0.02, 'float')
    p.define('vec_opt_beta', 1.0, 'float')
    p.define('vec_opt_weight_decay_factor', 0.001, 'float')
    p.define('vec_opt_init_factor', 0.0625, 'float')

    # hessian sketch
    p.train.define("hessian_sketch", hyperparams.Params(),
                   "Hessian sketch parameters.")
    p.train.hessian_sketch.define('compression_times', 1,
                                  'Hessian sketch compression times')

    # QAT
    p.train.define("qat", hyperparams.Params(), "QAT parameters.")
    p.train.qat.define('enable', False, 'Enable QAT')
    p.train.qat.define('fixed_range', 1.0, 'Fixed range')

    # HashNet
    p.train.define("hash_net", hyperparams.Params(), "HashNet parameters.")
    p.train.hash_net.define('enable', False, 'Enable HashNet.')
    p.train.hash_net.define('step_size', 200,
                            'Step size for scheduling beta of tanh.')
    p.train.hash_net.define('amplitude', 0.1, 'Amplitude of tanh function.')

    return p

  def __init__(self, params=None):
    params = params or NativeModelV2.params()
    super(NativeModelV2, self).__init__(params)
    dump_utils.enable = False

  @property
  def _default_vec_compressor(self):
    assert not (self.p.train.qat.enable and self.p.train.hash_net.enable
               ), "Please do NOT enable QAT and HashNet simultaneously!"
    if self.p.train.qat.enable:
      return FixedR8Compressor(self.p.train.qat.fixed_range)
    elif self.p.train.hash_net.enable:
      return OneBitCompressor(self.p.train.hash_net.step_size,
                              self.p.train.hash_net.amplitude)
    else:
      return Fp16Compressor()

  @property
  def _default_dense_optimizer(self):
    return tf.compat.v1.train.AdagradOptimizer(learning_rate=0.01,
                                               initial_accumulator_value=0.001)

  def _get_or_create_fs(self, slot, is_bias=False) -> FeatureSlot:
    if "_default_bias_optimizer" not in self.__dict__:
      self.instantiate()

    if slot in self.fs_dict:
      return self.fs_dict[slot]

    if slot in self._occurrence_threshold:
      occurrence_threshold = self._occurrence_threshold[slot]
    else:
      occurrence_threshold = self.default_occurrence_threshold

    if is_bias:
      fs = self.ctx.create_feature_slot(
          FeatureSlotConfig(
              name=str(slot),
              has_bias=True,
              bias_optimizer=self._default_bias_optimizer,
              bias_initializer=self._default_bias_initializer,
              bias_compressor=self._default_bias_compressor,
              default_vec_optimizer=self._default_vec_optimizer,
              default_vec_initializer=self._default_vec_initializer,
              default_vec_compressor=self._default_vec_compressor,
              slot_id=slot,
              occurrence_threshold=occurrence_threshold))
    else:
      fs = self.ctx.create_feature_slot(
          FeatureSlotConfig(
              name=str(slot),
              has_bias=True,
              bias_optimizer=self._default_bias_optimizer,
              bias_initializer=self._default_bias_initializer,
              bias_compressor=self._default_bias_compressor,
              default_vec_optimizer=self._default_vec_optimizer,
              default_vec_initializer=self._default_vec_initializer,
              default_vec_compressor=self._default_vec_compressor,
              slot_id=slot,
              occurrence_threshold=occurrence_threshold))

    self.fs_dict[slot] = fs
    return fs

  def _get_or_create_fc(self,
                        slot: int,
                        name: str = None,
                        is_bias: bool = False,
                        max_seq_length: int = 0) -> FeatureColumn:
    name = name or 'fc_slot_{}'.format(slot)
    if name not in self.fc_dict:
      fs = self._get_or_create_fs(slot, is_bias=is_bias)
      if max_seq_length > 0:
        combiner = FeatureColumn.first_n(max_seq_length)
      else:
        combiner = FeatureColumn.reduce_sum()
      fc = FeatureColumn(fs, name, combiner=combiner)
      self.fc_dict[name] = fc
      return fc
    else:
      return self.fc_dict[name]

  def _get_or_create_slice(self,
                           fc: FeatureColumn,
                           slice_name: str,
                           dim: int,
                           initializer: Initializer = None,
                           optimizer: Optimizer = None,
                           compressor: Compressor = None,
                           learning_rate_fn=None) -> FeatureSlice:
    feature_slot = fc.feature_slot
    slot = feature_slot.slot

    if slot in self.slice_dict:
      if slice_name in self.slice_dict[slot]:
        fc_slice = self.slice_dict[slot][slice_name]
      else:
        fc_slice = feature_slot.add_feature_slice(
            dim,
            initializer=initializer,
            optimizer=optimizer,
            compressor=compressor,
            learning_rate_fn=learning_rate_fn)
        self.slice_dict[slot][slice_name] = fc_slice
    else:
      fc_slice = feature_slot.add_feature_slice(
          dim,
          initializer=initializer,
          optimizer=optimizer,
          compressor=compressor,
          learning_rate_fn=learning_rate_fn)
      self.slice_dict[slot] = {slice_name: fc_slice}

    return fc_slice

  def instantiate(self):
    self._default_bias_optimizer = FtrlOptimizer(
        learning_rate=self.p.bias_opt_learning_rate,
        initial_accumulator_value=1e-6,
        beta=self.p.bias_opt_beta,
        l1_regularization=self.p.bias_l1_regularization,
        l2_regularization=self.p.bias_l2_regularization)

    self._default_bias_initializer = ZerosInitializer()

    self._default_vec_optimizer = AdagradOptimizer(
        learning_rate=self.p.vec_opt_learning_rate,
        weight_decay_factor=self.p.vec_opt_weight_decay_factor,
        initial_accumulator_value=self.p.vec_opt_beta,
        hessian_compression_times=self.p.train.hessian_sketch.compression_times)

    self._default_vec_initializer = RandomUniformInitializer(
        -self.p.vec_opt_init_factor, self.p.vec_opt_init_factor)
    logging.info('default_vec_initializer: %s', self._default_vec_initializer)

    self._default_bias_compressor = Fp32Compressor()

    return self

  def set_occurrence_threshold(self, occurrence_threshold: Dict[int, int]):
    self._occurrence_threshold.update(occurrence_threshold)

  def embedding_lookup(self,
                       slice_name,
                       slots,
                       dim=None,
                       out_type=None,
                       axis=1,
                       keep_list=False,
                       max_seq_length: int = 0,
                       initializer: Initializer = None,
                       optimizer: Optimizer = None,
                       compressor: Compressor = None,
                       learning_rate_fn=None) -> tf.Tensor:
    """
    Args:
        slice_name: slice_name for this alloc
        slots: only allowed the three following input,
            - a list/dict of slot with fixed dim: for example [slot1, slot2, ...] , in this condition the `dim` must
              set to a int
            - a list of slot with varies dim: for example [(slot1, dim1), (slot2, dim2), ...], in this condition
              the `dim` must set to a `None`
            - a list of slot groups with fixed dim: for example [(slot1, slot2, ...), (slot10, slot11, ...), ...],
              in this condition the `dim` must set to a int. the embedding of each group will be added
        dim (Option[int]): fix dim of embedding if not None
        out_type (Option[str]): 'stack', 'concat', or None
        axis (int):
        keep_list (bool):
    """

    slot_embeddings, slot_fcs = [], []
    if isinstance(slots, dict):
      for fc_name, info in slots.items():
        slot, emb_dim = None, None
        if isinstance(info, int):
          slot = info
          assert dim is not None
          emb_dim = dim
        elif isinstance(info, dict):
          slot = info['slot']
          if 'dim' in info:
            emb_dim = info['dim']
          elif 'vec_dim' in info:
            emb_dim = info['vec_dim']
          else:
            assert dim is not None
            emb_dim = dim

        fc = self._get_or_create_fc(slot,
                                    fc_name,
                                    max_seq_length=max_seq_length)
        slot_fcs.append(fc)
        fc_slice = self._get_or_create_slice(fc, slice_name, emb_dim,
                                             initializer, optimizer, compressor,
                                             learning_rate_fn)
        slot_embeddings.append(fc.embedding_lookup(fc_slice))
    elif isinstance(slots, (list, tuple)) and isinstance(dim, int):
      if all([isinstance(ele, int) for ele in slots]):
        # a list of slot with fixed dim
        for sid in slots:
          fc = self._get_or_create_fc(sid, max_seq_length=max_seq_length)
          slot_fcs.append(fc)
          fc_slice = self._get_or_create_slice(fc, slice_name, dim, initializer,
                                               optimizer, compressor,
                                               learning_rate_fn)
          slot_embeddings.append(fc.embedding_lookup(fc_slice))
      elif all([isinstance(ele, (tuple, list)) for ele in slots]):
        # a list of slot groups with fixed dim
        for group in slots:
          assert all([isinstance(ele, int) for ele in group])
          group_embeddings = []
          for sid in group:
            fc = self._get_or_create_fc(sid, max_seq_length=max_seq_length)
            slot_fcs.append(fc)
            fc_slice = self._get_or_create_slice(fc, slice_name, dim,
                                                 initializer, optimizer,
                                                 compressor, learning_rate_fn)
            group_embeddings.append(fc.embedding_lookup(fc_slice))
          slot_embeddings.append(tf.add_n(group_embeddings))
      else:
        raise ValueError("ValueError for features")
    elif isinstance(slots, (list, tuple)):
      if all(
          [isinstance(ele, (tuple, list)) and len(ele) == 2 for ele in slots]):
        for sid, sdim in slots:
          fc = self._get_or_create_fc(sid, max_seq_length=max_seq_length)
          slot_fcs.append(fc)
          fc_slice = self._get_or_create_slice(fc, slice_name, sdim,
                                               initializer, optimizer,
                                               compressor, learning_rate_fn)
          slot_embeddings.append(fc.embedding_lookup(fc_slice))
      else:
        raise ValueError("ValueError for features")
    else:
      raise ValueError("ValueError for features")

    if any([type(fc.combiner) == FirstN for fc in slot_fcs]):
      assert all([type(fc.combiner) == FirstN for fc in slot_fcs])

    if len(slot_embeddings) == 1 and not keep_list:
      target = slot_embeddings[0]
    elif out_type == 'stack':
      if isinstance(dim, int):
        # [batch, solt_dim, emb_dim]
        target = tf.stack(slot_embeddings, axis=axis)
      else:
        raise ValueError("cannot output stacked of varies dim")
    elif out_type == 'concat':
      # [batch, solt_dim * emb_dim]
      target = tf.concat(slot_embeddings, axis=axis)
    else:
      # [batch, emb_dim] * solt_dim/num_group
      target = slot_embeddings

    return target

  def bias_lookup(self,
                  bias_slots,
                  cal_bias_sum=True,
                  keepdims=True) -> tf.Tensor:
    # multi-hot don't support bias
    assert len(bias_slots) == len(set(bias_slots))

    for slot in bias_slots:
      if slot not in self.fs_dict:
        self._get_or_create_fc(slot=slot, is_bias=True)

    bias_list = []
    for sid in bias_slots:
      for fc in self.fs_dict[sid].get_feature_columns():
        bias_list.append(fc.get_bias())
        break

    bias_nn = tf.concat(bias_list,
                        axis=1,
                        name='concatenate_tensor_from_{}_bias'.format(
                            len(bias_list)))

    if cal_bias_sum:
      bias_sum = tf.reduce_sum(bias_nn, axis=1, keepdims=keepdims)  # [batch, 1]
      return bias_nn, bias_sum
    else:
      return bias_nn

  def clean(self):
    # update fs_dict, fc_dict, slice_dict
    self.fs_dict = {}
    self.fc_dict = {}
    self.slice_dict = {}  # slot_id -> Dict[slot_id, slice]
    self.multi_head_preds = {}

  def create_input_fn(self, mode):
    return partial(self.input_fn, mode)

  def create_model_fn(self):
    self.clean()

    def model_fn_internal(
        features: Dict[str, tf.Tensor], mode: tf.estimator.ModeKeys,
        config: tf.estimator.RunConfig) -> tf.estimator.EstimatorSpec:

      real_mode = self._get_real_mode(mode)
      label, loss, pred = self.model_fn(features, real_mode)
      if self.losses:
        loss = loss + tf.add_n(self.losses)

      if real_mode == tf.estimator.ModeKeys.PREDICT:
        if is_exporting() or self.p.output_path is None:
          spec = tf.estimator.EstimatorSpec(real_mode,
                                            predictions=pred,
                                            training_hooks=self._training_hooks)
        else:
          op_file, write_op = self._get_file_ops(
              features,
              pred.values() if isinstance(pred, dict) else pred)
          hooks = [file_ops.FileCloseHook([op_file])]
          with tf.control_dependencies(control_inputs=[write_op]):
            if isinstance(pred, dict):
              pred = {k: tf.identity(pred[k]) for k in pred}
            else:
              pred = tf.identity(pred)
            spec = tf.estimator.EstimatorSpec(mode,
                                              training_hooks=hooks +
                                              self._training_hooks,
                                              predictions=pred)

        if is_exporting() and self._export_outputs:
          self._export_outputs.update(spec.export_outputs)
          spec = spec._replace(export_outputs=self._export_outputs)

        return spec

      train_ops = []
      enable_metrics = self.metrics.enable_kafka_metrics or self.metrics.enable_deep_insight
      if enable_metrics and self.metrics.deep_insight_sample_ratio > 0:
        model_name = self.p.metrics.deep_insight_name
        assert not (isinstance(label, dict) ^ isinstance(pred, dict))
        targets, labels_list, preds_list = None, None, None
        if isinstance(label, dict):
          label_keys, pred_keys = sorted(label.keys()), sorted(pred.keys())
          assert label_keys == pred_keys, 'label.key() = {}, pred.keys() = {}'.format(
              label_keys, pred_keys)
          labels_list = [label[k] for k in label_keys]
          preds_list = [pred[k] for k in pred_keys]
          targets = label_keys
          logging.info("model_name: {}, targets: {}.".format(
              model_name, targets))

        deep_insight_op = metric_utils.write_deep_insight(
            features=features,
            sample_ratio=self.p.metrics.deep_insight_sample_ratio,
            labels=label,
            preds=pred,
            model_name=model_name or "model_name",
            target=self.p.metrics.deep_insight_target,
            targets=targets,
            labels_list=labels_list,
            preds_list=preds_list,
            enable_kafka_metrics=self.metrics.enable_kafka_metrics)

        tf.compat.v1.add_to_collection("deep_insight_op", deep_insight_op)
        if self.metrics.enable_kafka_metrics:
          self.add_training_hook(KafkaMetricHook(deep_insight_op))
          logging.info('add KafkaMetricHook success')
        
        logging.info("model_name: {}, target: {}.".format(
            model_name, self.p.metrics.deep_insight_target))
        train_ops.append(deep_insight_op)

      if real_mode == tf.estimator.ModeKeys.EVAL:
        if is_exporting() or self.p.output_path is None:
          train_ops.append(tf.identity(pred))
          return tf.estimator.EstimatorSpec(mode,
                                            loss=loss,
                                            train_op=tf.group(train_ops),
                                            training_hooks=self._training_hooks)
        else:
          op_file, write_op = self._get_file_ops(features, pred)
          hooks = [file_ops.FileCloseHook([op_file])]
          with tf.control_dependencies(control_inputs=[write_op]):
            train_ops.append(tf.identity(pred))
            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=tf.group(train_ops),
                                              training_hooks=hooks +
                                              self._training_hooks)

      else:  # training
        dense_optimizer = self._default_dense_optimizer

        train_ops.append(
            feature_utils.apply_gradients_with_var_optimizer(
                self.ctx,
                self.fc_dict.values(),
                dense_optimizer,
                loss,
                clip_type=feature_utils.GradClipType.ClipByGlobalNorm,
                clip_norm=self.p.clip_norm,
                dense_weight_decay=self.p.dense_weight_decay,
                global_step=self._global_step,
                grads_and_vars_summary=self.enable_grads_and_vars_summary))

        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=tf.group(train_ops),
                                          training_hooks=self._training_hooks)

    return model_fn_internal

  def create_serving_input_receiver_fn(self):
    return self.serving_input_receiver_fn

  def get_sigmoid_loss_and_pred(
      self,
      name,
      logits,
      label,
      sample_rate: tf.Tensor = None,
      sample_bias: float = 0.,
      mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN):
    return get_sigmoid_loss_and_pred(name, logits, label, self.batch_size,
                                     sample_rate, sample_bias, mode)

  @staticmethod
  def get_softmax_loss_and_pred(name, logits, label, mode):
    return get_softmax_loss_and_pred(name, logits, label, mode)


class NativeDeepRoughSortModel(NativeModelV2, DeepRoughSortBaseModel):

  def __init__(self, params=None):
    super().__init__(params)

  def model_fn(
      self, features: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return self._model_fn(features, mode)


@monolith_export
class MonolithDeepRoughSortModel(MonolithModel, DeepRoughSortBaseModel):

  def __init__(self, params=None):
    super().__init__(params)

  def model_fn(
      self, features: Dict[str, tf.Tensor],
      mode: tf.estimator.ModeKeys) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return self._model_fn(features, mode)
