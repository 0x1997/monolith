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

"""A warpper that set necessary env and just forward flags to the cpu_runner."""
import copy
import os
import subprocess
import sys
import time

from absl import logging

from monolith.native_training import env_utils

os.environ["MONOLITH_WITH_HOROVOD"] = "True"
test_locally = os.getenv("TEST_LOCALLY")
# This is a wrapper process, which spawns a child process for the real training.
if test_locally == None:
  env_utils.setup_hdfs_env()
argv = copy.deepcopy(sys.argv)
logging.info("The input cmd is: %s", argv)
# When we use sync training, MPIRUN drives the job, and thus we do not need
# to start the job here.
worker_id = int(os.environ.get("ORACLE_ID"))
test_no_mpi = os.getenv("TEST_NO_MPI")
if worker_id != 0:
  # TODO(zouxuan): add consul registration.
  # We give a huge sleep duration.
  time.sleep(2**30)
else:
  argv[0] = "monolith/native_training/cpu_runner"
  # TODO(zouxuan): add -H options
  # TODO(zouxuan): add profiling options
  mpirun_args = "mpirun -x HOROVOD_CYCLE_TIME=0.1 -x HOROVOD_AUTOTUNE=1 -x HOROVOD_MPI_THREADS_DISABLE=1 --report-bindings -bind-to socket -np 1 --map-by ppr:1:socket --mca pml ucx --mca btl ^vader,openib,uct -x UCX_NET_DEVICES=mlx5_0:1"
  mpirun_args = mpirun_args.split(" ")
  if test_no_mpi == None:
    argv = mpirun_args + argv
  logging.info("The running cmd is: %s", argv)
  subprocess.run(argv, check=True)
