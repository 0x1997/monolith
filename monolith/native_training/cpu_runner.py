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

import hashlib
import json
import os
import time
import importlib

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from monolith.agent_service.backends import ZKBackend
from monolith.agent_service.utils import AgentConfig
from monolith.agent_service.replica_manager import ReplicaWatcher
from monolith.core import model_registry
from monolith.native_training import cpu_training
from monolith.native_training import env_utils
from monolith.native_training.service_discovery import ServiceDiscoveryType, \
  ConsulServiceDiscovery, TfConfigServiceDiscovery, ZKServiceDiscovery
from monolith.native_training.zk_utils import default_zk_servers
from monolith.native_training import gflags_utils
from monolith.native_training.runner_utils import RunnerConfig
from monolith.native_training import utils
from monolith.native_training import yarn_runtime
from monolith.native_training.zk_utils import MonolithKazooClient

FLAGS = flags.FLAGS

_PSM_PREFIX = "data.aml.monolith_native_training"


def main(_):
  env_utils.setup_host_ip()
  tf.compat.v1.disable_eager_execution()
  logging.set_verbosity(logging.INFO)

  if FLAGS.use_estimator:
    model_py = importlib.import_module(FLAGS.task)
    return model_py.main(_)

  dct_config = RunnerConfig()

  os.environ["TF_GRPC_WORKER_CACHE_THREADS"] = str(
      dct_config.tf_grpc_worker_cache_threads)
  os.environ["MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER"] = str(
      dct_config.monolith_grpc_worker_service_handler_multiplier)
  utils.set_metric_prefix("monolith.training.{}".format(
      dct_config.deep_insight_name))
  logging.info("Environment vars: %s", os.environ)
  logging.info("Flags: %s", flags.FLAGS.flag_values_dict())
  params = model_registry.GetParams(dct_config.task)
  params.metrics.deep_insight_name = dct_config.deep_insight_name
  if params.train.use_fountain and bool(dct_config.fountain_zk_host) and bool(
      dct_config.fountain_model_name):
    logging.info("Override Fountain Params:{}; {}".format(
        dct_config.fountain_model_name, dct_config.fountain_zk_host))
    params.train.fountain_zk_host = dct_config.fountain_zk_host
    params.train.fountain_model_name = dct_config.fountain_model_name
    # TODO: unify the two different above and below params_override logics for fountain
  if dct_config.params_override:
    logging.info("Override: {}".format(dct_config.params_override))
    params_override_dict = json.loads(dct_config.params_override)
    params.set(**params_override_dict)
  logging.info("Model params: {}".format(params))

  psm = env_utils.generate_psm_from_uuid(dct_config.uuid)
  logging.info("PSM: %s", psm)

  zk_servers = dct_config.zk_server or os.environ.get('zk_servers',
                                                      default_zk_servers())
  kazoo_client, sync_backend = None, None
  if dct_config.bzid:
    if dct_config.unified_serving:
      sync_backend = ZKBackend(dct_config.bzid, zk_servers=zk_servers)
    else:
      assert dct_config.base_name, "Base name cannot be none while realtime training."
      kazoo_client = MonolithKazooClient(hosts=zk_servers)
      kazoo_client.start()
      agent_config = AgentConfig(bzid=dct_config.bzid,
                                 base_name=dct_config.base_name,
                                 deploy_type='ps',
                                 num_ps=dct_config.num_ps,
                                 dc_aware=dct_config.dc_aware)
      replica_watcher = ReplicaWatcher(
          kazoo_client,
          agent_config,
          zk_watch_address_family=dct_config.zk_watch_address_family)
      sync_backend = replica_watcher.to_sync_wrapper()
    sync_backend.start()
    sync_backend.subscribe_model(dct_config.model_name or
                                 params.metrics.deep_insight_name)

  if dct_config.enable_sync_training:
    logging.info("Entering synchronous training.")
    # Import and init horovod/byteps on demand.
    enable_bps = int(os.getenv("MONOLITH_WITH_BYTEPS", "0"))
    if enable_bps:
      from monolith.native_training import distribution_utils
      distribution_utils.bps_init(dct_config.uuid)
      import byteps.tensorflow as hvd

      enable_bps_bcast = int(os.getenv("MONOLITH_WITH_BYTEPS_BCAST", "1"))
      enable_bps_allreduce = int(
          os.getenv("MONOLITH_WITH_BYTEPS_ALLREDUCE", "1"))
      if enable_bps_bcast == 0 or enable_bps_allreduce == 0:
        import horovod.tensorflow as real_hvd
        real_hvd.init()
    else:
      import horovod.tensorflow as hvd
      hvd.init()

    # At developement environment (dev machines, arnold clusters, containers etc.)
    # when mpirun launches multi-processes via cpu_runner.py with hvd,
    # every process needs the hdfs envs to be setup.
    # TODO: Enter sync training via cpu_runner_wrapper, and set it there if necessary.
    if dct_config.enable_gpu_training:
      env_utils.setup_hdfs_env()

    index = hvd.rank()
    num_workers = hvd.size()
    if dct_config.merge_sync_training_ckpt:
      model_dir = dct_config.model_dir
    else:
      #TODO(zouxuan): Async push has some bug in unit test, keep it off for now.
      model_dir_suffix = 'index-{:04}'.format(index)
      model_dir = os.path.join(dct_config.model_dir, dct_config.uuid,
                               model_dir_suffix)

    dct_config.model_dir = model_dir
    dct_config.num_ps = 0
    dct_config.reorder_fids_in_data_pipeline = True
    dct_config.index = index
    dct_config.num_workers = num_workers
    dct_config.enable_variable_partition = False

    benchmark_bps = os.environ.get("MONOLITH_BENCHMARK_BPS", "none")
    if benchmark_bps != "none":
      from monolith.native_training.distribution_utils import bps_comm_benchmark
      bps_comm_benchmark()
      return
    cpu_training.distributed_sync_train(dct_config, params, sync_backend)

  else:
    discovery_type = dct_config.discovery_type
    if discovery_type == ServiceDiscoveryType.PRIMUS:
      assert dct_config.tf_config is not None
      tf_config = json.loads(dct_config.tf_config)
      discovery = TfConfigServiceDiscovery(tf_config)
      dct_config.server_type = discovery.server_type
      dct_config.index = discovery.index
    elif discovery_type == ServiceDiscoveryType.CONSUL:
      # For async training, PS discovery is inside the process.
      discovery = ConsulServiceDiscovery(psm)
    else:
      discovery = ZKServiceDiscovery(dct_config.deep_insight_name, zk_servers)
    try:
      cpu_training.distributed_train(dct_config, discovery, params,
                                     sync_backend)
    finally:
      discovery.close()
      if kazoo_client:
        kazoo_client.stop()
      if sync_backend:
        sync_backend.stop()


if __name__ == "__main__":
  app.run(main)
