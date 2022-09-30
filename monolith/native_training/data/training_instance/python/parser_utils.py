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

import tensorflow as tf
from collections import deque
from typing import Callable

from monolith.native_training import ragged_utils

_extra_parse_steps = deque([])


def add_extra_parse_step(parse_fn: Callable):
  _extra_parse_steps.append(parse_fn)


class RaggedEncodingHelper:
  """Helper methods to precompute ragged encodings in input_fn, as a workaround
  
  Fundamentally, we should modify TensorFlow Dataset structure handler to compute
  provided encoding tensor in RowParition of a RaggedTensor.
  """

  suffix_value_rowids = ":value_rowids"

  @staticmethod
  def expand(name_to_ragged_ids):
    """Expand the RaggedTensor format in dict to precompute encodings within data iterator."""
    d = {}
    for k, v in name_to_ragged_ids.items():
      if isinstance(v, tf.RaggedTensor):
        d[k + RaggedEncodingHelper.
          suffix_value_rowids] = ragged_utils.fused_value_rowids(v)
    name_to_ragged_ids.update(d)
    return name_to_ragged_ids

  @staticmethod
  def contract(name_to_ragged_ids):
    """Contract to recover RaggedTensor-only dict after computed."""
    o_keys = list(name_to_ragged_ids.keys())
    for k in o_keys:
      if RaggedEncodingHelper.suffix_value_rowids in k:
        name = k[:-len(RaggedEncodingHelper.suffix_value_rowids)]
        assert name_to_ragged_ids[
            name]._row_partition._value_rowids is None, "Shouldn't override the exisiting tensor."
        name_to_ragged_ids[
            name]._row_partition._value_rowids = name_to_ragged_ids.pop(k)
    return name_to_ragged_ids


def advanced_parse(features):
  while _extra_parse_steps:
    fn = _extra_parse_steps.popleft()
    features = fn(features)

  return features