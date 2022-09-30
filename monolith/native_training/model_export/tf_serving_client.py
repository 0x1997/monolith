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

from absl import app
from absl import flags
from absl import logging

import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from monolith.native_training.model_export import demo_predictor


def load_emb_table():
  channel = grpc.insecure_channel("127.0.0.1:8710")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = "entry"
  request.model_spec.signature_name = "load_cache_table"
  request.inputs["table_path"].CopyFrom(
          tf.make_tensor_proto("/home/yijie.zhu/tmp/hashtables/MonolithHashTable"))
  result = stub.Predict(request, 30)
  print(result)

def fetch_item():
  channel = grpc.insecure_channel("127.0.0.1:8710")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = "entry"
  request.model_spec.signature_name = "fetch_item"
  request.inputs["instances"].CopyFrom(
          tf.make_tensor_proto(
              demo_predictor.random_generate_instances(1)))
  result = stub.Predict(request, 30)
  print(result)

def fetch_user():
  channel = grpc.insecure_channel("127.0.0.1:8710")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = "entry"
  request.model_spec.signature_name = "fetch_user"
  request.inputs["instances"].CopyFrom(
          tf.make_tensor_proto(
              demo_predictor.random_generate_instances(1)))
  result = stub.Predict(request, 30)
  print("fetch_user: ", result)
  pred_request = predict_pb2.PredictRequest()
  pred_request.model_spec.name = "entry"
  pred_request.model_spec.signature_name = "distributed_pred"

  for k in result.outputs:
    pred_request.inputs[k].CopyFrom(result.outputs[k])
  pred_request.inputs["item_ids"].CopyFrom(tf.make_tensor_proto(tf.range(10, dtype=tf.int64)))
  result = stub.Predict(pred_request, 30)
  print("distributed_pred: ", result)


if __name__ == "__main__":
  # load_emb_table()
  fetch_item()
  fetch_user()