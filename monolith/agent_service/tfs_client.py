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

from absl import app, flags, logging
from datetime import datetime
import grpc
import json
import random
import socket
import os
import sys
from struct import unpack
from typing import List
from google.protobuf import text_format
from idl.matrix.proto.proto_parser_pb2 import Instance

from tensorflow_serving.apis.get_model_metadata_pb2 import GetModelMetadataRequest
from tensorflow_serving.apis.get_model_status_pb2 import GetModelStatusRequest
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis.model_management_pb2 import ReloadConfigRequest
from tensorflow_serving.config.model_server_config_pb2 import ModelServerConfig
from monolith.agent_service import utils
from monolith.agent_service.client import FLAGS
from monolith.native_training.data.feature_list import FeatureList
from idl.matrix.proto.example_pb2 import Example, ExampleBatch, FeatureListType
from idl.matrix.proto.line_id_pb2 import LineId
from monolith.native_training import env_utils


_NUM_SLOTS = 6
_VOCAB_SIZES = [5, 5, 5, 5, 5, 5]

flags.DEFINE_string("signature_name", "serving_default", "signature name")
flags.DEFINE_string("feature_list", None, "feature_list for prediction")
flags.DEFINE_enum("file_type", 'pb', ['pb', 'pbtxt'], "The input file type")
flags.DEFINE_integer("batch_size", 8, "batch_size for prediction")
flags.DEFINE_bool("lagrangex_header", False, "wheather has lagrangex_header")
flags.DEFINE_bool("has_sort_id", False, "wheather has sort_id")
flags.DEFINE_bool("kafka_dump", False, "wheather has kafka_dump")
flags.DEFINE_bool("kafka_dump_prefix", False, "wheather has kafka_dump_prefix")


SKIP_LIST = {'-', '_lt_', '_st_', '_lt', '_st', '_cp_', '_recent_', '_cp', '_recent'}


def read_header(stream):
  int_size = 8
  if FLAGS.lagrangex_header:
    stream.read(int_size)
  else:
    aggregate_page_sortid_size = 0

    if FLAGS.kafka_dump_prefix:
      size = unpack("<Q", stream.read(int_size))[0]
      if size == 0:
        size = unpack("<Q", stream.read(int_size))[0]
      else:
        aggregate_page_sortid_size = size
    
    if FLAGS.has_sort_id:
      if aggregate_page_sortid_size == 0:
        size = unpack("<Q", stream.read(int_size))[0]
      else:
        aggregate_page_sortid_size = size
      stream.read(size)
    if FLAGS.kafka_dump:
      stream.read(int_size)


def read_data(stream):
  read_header(stream)
  size = unpack("<Q", stream.read(8))[0]
  return stream.read(size)


def generate_random_instance(slots: List[int] = None,
                             vocab_sizes: List[int] = _VOCAB_SIZES):
  max_vocab = max(vocab_sizes)
  if slots is None:
    slots = list(range(1, len(_VOCAB_SIZES) + 1))
  
  fids = [(slot << 54) | (i * max_vocab + random.randint(1, vocab_sizes[i]))
          for i, slot in enumerate(slots) for _ in range(vocab_sizes[i])]
  
  instance = Instance()
  instance.fid.extend(fids)
  return instance.SerializeToString()


def generate_random_example_batch(feature_list: FeatureList, batch_size: int = 256) -> str:
  eb = ExampleBatch()
  eb.batch_size = batch_size

  for feature in feature_list:
    flag = False
    for s in SKIP_LIST:
      if s in feature.name:
        flag = True
        break

    if flag:
      continue

    if not ("_id" in feature.name or "_name" in feature.name):
      continue

    named_feature_list = eb.named_feature_list.add()
    named_feature_list.name = feature.name
    for _ in range(batch_size):
      _feature = named_feature_list.feature.add()
      if feature.method.lower().startswith('vectortop') and feature.args is not None:
        if len(feature.args) > 0 and feature.args[0].isnumeric():
          num = int(feature.args[0])
          if num > 0:
            num = random.randint(1, num)

          fids = [(feature.slot << 48) | random.randint(1, sys.maxsize - 1) for _ in range(num)]
          _feature.fid_v2_list.value.extend(fids)
      else:
        fid = (feature.slot << 48) | random.randint(1, (1 << 48) - 1)
        _feature.fid_v2_list.value.append(fid)

  named_feature_list = eb.named_feature_list.add()
  named_feature_list.name = '__LINE_ID__'
  for _ in range(batch_size):
    _feature = named_feature_list.feature.add()
    line_id = LineId()
    line_id.sample_rate = 0.001
    line_id.req_time = int(datetime.now().timestamp() - random.randint(1, 1000))
    line_id.actions.extend([random.randint(1, 3), random.randint(3, 5)])
    _feature.bytes_list.value.append(line_id.SerializeToString())

  return eb.SerializeToString()


def get_instance_proto(input_file: str = None, batch_size: int = 256):
  if input_file is None:
    instances = [generate_random_instance() for _ in range(batch_size)]
  else:
    assert os.path.exists(input_file)
    with open(input_file, 'rb') as stream:
      instances = []
      for _ in range(batch_size):
        inst = Instance()
        inst.ParseFromString(read_data(stream))
        instances.append(inst.SerializeToString())
  return utils.make_tensor_proto(instances)


def get_example_batch_proto(input_file: str = None, feature_list: FeatureList = None, 
                            batch_size: int = 256, file_type: str = 'pb'):
  if input_file is None:
    example_batch = generate_random_example_batch(feature_list, batch_size)
  else:
    assert os.path.exists(input_file)

    eb = ExampleBatch()
    if file_type == 'pb':
      with open(input_file, 'rb') as stream:
        eb.ParseFromString(read_data(stream))
    else:
      with open(input_file, 'r') as stream:
        txt = stream.read()
        text_format.Parse(txt, eb)
    
    example_batch = eb.SerializeToString()
  return utils.make_tensor_proto([example_batch])


def get_example_batch_to_instance(input_file: str, file_type: str):
  assert os.path.exists(input_file)

  eb = ExampleBatch()
  if file_type == 'pb':
    with open(input_file, 'rb') as stream:
      eb.ParseFromString(read_data(stream))
  else:
    with open(input_file, 'r') as stream:
      txt = stream.read()
      text_format.Parse(txt, eb)

  inst_list = []
  mask = (1 << 48) - 1 
  for i in range(eb.batch_size):
    inst = Instance()
    for named_feature_list in eb.named_feature_list:
      if named_feature_list.type == FeatureListType.SHARED:
        efeat = named_feature_list.feature[0]
      else:
        efeat = named_feature_list.feature[i]
      
      if named_feature_list.name == '__LABEL__':
        inst.label.extend(efeat.float_list.value)
      elif named_feature_list.name == '__LINE_ID__':
        inst.line_id.ParseFromString(efeat.bytes_list.value[0])
      elif len(efeat.fid_v1_list.value) > 0:
        ifeat = inst.feature.add()
        ifeat.name = named_feature_list.name
        fid = efeat.fid_v1_list.value[0]
        slot_id = fid >> 54
        fid_v2 = [(slot_id << 48) | (mask & v) for v in efeat.fid_v1_list.value]
        ifeat.fid.extend(fid_v2)
      elif len(efeat.fid_v2_list.value) > 0:
        ifeat = inst.feature.add()
        ifeat.name = named_feature_list.name
        ifeat.fid.extend(efeat.fid_v2_list.value)
      elif len(efeat.float_list.value) > 0:
        ifeat = inst.feature.add()
        ifeat.name = named_feature_list.name
        ifeat.float_value.extend(efeat.float_list.value)
      elif len(efeat.double_list.value) > 0:
        ifeat = inst.feature.add()
        ifeat.name = named_feature_list.name
        ifeat.float_value.extend(efeat.double_list.value)
      elif len(efeat.int64_list.value) > 0:
        ifeat = inst.feature.add()
        ifeat.name = named_feature_list.name
        ifeat.int64_value.extend(efeat.int64_list.value)
      elif len(efeat.bytes_list.value) > 0:
        ifeat = inst.feature.add()
        ifeat.name = named_feature_list.name
        ifeat.bytes_value.extend(efeat.bytes_list.value)
      else:
        pass
    
    inst_list.append(inst.SerializeToString())
  
  return utils.make_tensor_proto(inst_list)


def main(_):
  env_utils.setup_host_ip()
  agent_conf = utils.AgentConfig.from_file(FLAGS.conf)
  host = os.environ.get("MY_HOST_IP", socket.gethostbyname(socket.gethostname()))
  
  model_name = FLAGS.model_name
  if model_name is None:
    model_name = utils.TFSServerType.ENTRY if agent_conf.deploy_type != utils.DeployType.PS else 'ps_{}'.format(agent_conf.shard_id)
  
  if agent_conf.deploy_type == utils.DeployType.PS:
    target = FLAGS.target or f"{host}:{agent_conf.tfs_ps_port}"
  else:
    target = FLAGS.target or f"{host}:{agent_conf.tfs_entry_port}"
  
  channel = grpc.insecure_channel(target)
  
  if FLAGS.cmd_type == 'status':
    stub = ModelServiceStub(channel)
    request = GetModelStatusRequest()
    request.model_spec.CopyFrom(utils.gen_model_spec(model_name,
                                                     signature_name=FLAGS.signature_name))
    print(stub.GetModelStatus(request))
  elif FLAGS.cmd_type == 'meta':
    stub = PredictionServiceStub(channel)
    request = GetModelMetadataRequest()
    request.model_spec.CopyFrom(utils.gen_model_spec(model_name,
                                                     signature_name=FLAGS.signature_name))
    request.metadata_field.extend(['base_path', 'num_versions', 'signature_name'])
  
    response = stub.GetModelMetadata(request)
    print(response)
  elif FLAGS.cmd_type == 'load':
    request = ReloadConfigRequest()
    model_configs = ModelServerConfig()
    with open(FLAGS.input_file, 'r') as stream:
      txt = stream.read()
      text_format.Parse(txt, model_configs)
    
    request.config.CopyFrom(model_configs)
    stub = ModelServiceStub(channel)
    response = stub.HandleReloadConfigRequest(request)
    logging.info(f'{model_configs} load done!')

    return response.status
  else:
    # url = f"http://{target}/v1/models/{model_name}:{FLAGS.signature_name}"
    # cmd = ['curl', '-d', f"'{FLAGS.inputs}'", '-X', 'POST', url]
    # output = subprocess.check_output(cmd, shell=True)
    # print(output)

    stub = PredictionServiceStub(channel)
    request = PredictRequest()
    request.model_spec.CopyFrom(utils.gen_model_spec(model_name,
                                                     signature_name=FLAGS.signature_name))
    
    if FLAGS.input_type == 'instance':
      tensor_proto = get_instance_proto(FLAGS.input_file, FLAGS.batch_size)
      request.inputs["instances"].CopyFrom(tensor_proto)
    elif FLAGS.input_type == 'example_batch':
      feature_list = FeatureList.parse(FLAGS.feature_list)
      tensor_proto = get_example_batch_proto(FLAGS.input_file, feature_list, FLAGS.batch_size, FLAGS.file_type)
      request.inputs["example_batch"].CopyFrom(tensor_proto)
    else:
      tensor_proto = get_example_batch_to_instance(FLAGS.input_file, FLAGS.file_type)
      request.inputs["instances"].CopyFrom(tensor_proto)

    response = stub.Predict(request, 30)
    print(response)


if __name__ == "__main__":
  app.run(main)
