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

"""A wrapper that set necessary env and just forward flags to the cpu_runner."""
import argparse
import copy
import os
import subprocess
from subprocess import PIPE, Popen
import sys

from monolith import utils
from monolith.common.python import mem_profiling
from monolith.native_training import env_utils
from monolith.native_training import net_utils
from monolith.native_training import yarn_runtime

parser = argparse.ArgumentParser()
parser.add_argument("--kafka_dump_prefix",
                    type=str,
                    default=None,
                    help='kafka_dump_prefix')
parser.add_argument("--kafka_dump", type=str, default=None, help='kafka_dump')
parser.add_argument("--has_sort_id", type=str, default=None, help='has_sort_id')
parser.add_argument("--heap_profiling",
                    type=bool,
                    default=False,
                    help='heap_profiling')
parser.add_argument(
    "--has_fids",
    default=None,
    help="The instance of interest should contain at least one of the given "
    "fids, or it will be dropped.")
parser.add_argument(
    "--has_actions",
    default=None,
    help="The instance of interest should contain at least one of the given "
    "actions, or it will be dropped.")
parser.add_argument(
    "--filter_fids",
    default=None,
    help="The instance will be dropped if it contains any one of the given "
    "fids.")
parser.add_argument(
    "--select_fids",
    default=None,
    help="The instance of interest should contain all of the given fids, or it "
    "will be dropped.")
parser.add_argument("--req_time_min",
                    default=None,
                    help="The instance of interest should satisfy "
                    "line_id.req_time >= req_time_min, or it will be dropped.")
args, unknown = parser.parse_known_args()

env_utils.setup_hdfs_env()
# Google has BNS, however, we don't have. So we need to
# handle the worker failure by ourselves.
os.environ["GRPC_FAIL_FAST"] = "use_caller"
if args.heap_profiling:
  mem_profiling.setup_heap_profile(heap_profile_inuse_interval=104857600,
                                   heap_profile_allocation_interval=1073741824,
                                   heap_profile_time_interval=0,
                                   sample_ratio=0.3,
                                   heap_profile_mmap=False)
else:
  mem_profiling.enable_tcmalloc()
argv = copy.deepcopy(sys.argv)
argv[0] = utils.get_libops_path("monolith/native_training/cpu_runner")
cpu_runner_cmd = [argv[0]] + unknown

# 100M
BUFFER_SIZE = 100 * 1024 * 1024

instance_processor_args = [
    args.has_fids, args.has_actions, args.filter_fids, args.select_fids,
    args.req_time_min
]

args_dict = vars(args)

if all(arg is None for arg in instance_processor_args):
  cpu_runner_cmd += [
      '--{}={}'.format(key, args_dict[key])
      for key in args_dict
      if key in ['kafka_dump_prefix', 'kafka_dump', 'has_sort_id'] and
      args_dict[key] is not None
  ]
  print('A single CPU runner subprocess started...', flush=True)
  print(' '.join(cpu_runner_cmd), flush=True)
  os.execv(cpu_runner_cmd[0], cpu_runner_cmd)
else:
  instance_processor_cmd = utils.get_libops_path(
      "monolith/native_training/data/training_instance/instance_processor")
  instance_processor_cmd = [instance_processor_cmd] + [
      '--{}={}'.format(key, args_dict[key])
      for key in args_dict
      if args_dict[key] is not None
  ]
  print('Subprocess #1: instance processor subprocess started...', flush=True)
  print(' '.join(instance_processor_cmd), flush=True)
  instance_processor_process = Popen(instance_processor_cmd,
                                     stdout=PIPE,
                                     bufsize=BUFFER_SIZE)

  print('Subprocess #2: CPU runner subprocess started...')
  print(' '.join(cpu_runner_cmd), flush=True)
  cpu_runner_process = Popen(cpu_runner_cmd,
                             stdin=instance_processor_process.stdout,
                             stdout=sys.stdout,
                             bufsize=BUFFER_SIZE)
  cpu_runner_process.wait()
