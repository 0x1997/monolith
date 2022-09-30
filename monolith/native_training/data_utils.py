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

import datetime
from typing import List, Tuple

from absl import logging
import tensorflow as tf

from monolith.native_training.primus import input_generator


def _expand_paths(paths: List[str]):
  '''Expand `{one_piece,other_piece}` in paths.'''
  expanded = []
  for path in paths:
    lp = path.find('{')
    if lp >= 0:
      rp = path.find('}', lp)
      head = path[:lp]
      tail = path[rp + 1:]
      for body in path[lp + 1:rp].split(','):
        expanded.append(head + body + tail)
    else:
      expanded.append(path)
  return expanded


def _validate_date(date_text: str) -> Tuple[bool, str]:
  try:
    date_text = datetime.datetime.strptime(date_text,
                                           "%Y-%m-%d").strftime("%Y%m%d")
  except ValueError:
    try:
      datetime.datetime.strptime(date_text, "%Y%m%d")
    except:
      return False, date_text
  return True, date_text


def _get_sort_key(path: str) -> Tuple[str, str]:
  values = path.split("/")
  date_key = ""
  file_key = ""
  for value in values[::-1]:
    valid, date_key = _validate_date(value)
    if valid:
      break
    file_key = value + "/" + file_key
  return date_key, file_key


def _expand_glob_paths(paths_list: List[str]) -> List[str]:
  return [y for x in paths_list for y in tf.io.gfile.glob(x)]


def _sort_path_by_date_and_part(paths_list: List[str]) -> List[str]:

  return sorted(paths_list, key=_get_sort_key, reverse=False)


def process_paths(path_or_paths, expand_glob=False) -> List[str]:
  if not isinstance(path_or_paths, (tuple, list)):
    paths = [path_or_paths]
  else:
    paths = path_or_paths
  paths = _expand_paths(paths)
  if expand_glob:
    paths = _expand_glob_paths(paths)
    paths = _sort_path_by_date_and_part(paths)
  return paths


# This is a quick hack to fix the makelist too slow issue.
def process_path_with_sharding(path, shard_index, shard_cnt,
                               file_cnt) -> List[str]:
  input_paths = []
  path_list = _expand_paths([path])
  logging.info("We should see path: {}".format(path_list))
  for path in path_list:
    index = shard_index
    while index < file_cnt:
      handled_path = path + "{:05}".format(index) + ".pb.snappy"
      input_paths.append(handled_path)
      index += shard_cnt
  logging.info("We should see path: {}".format(input_paths))
  expanded_paths = _expand_glob_paths(input_paths)
  paths = _sort_path_by_date_and_part(expanded_paths)
  logging.info("The data for shard {} has {} files".format(
      shard_index, len(paths)))
  return paths


primus_style_input = input_generator.generate_input_by_wildcard
