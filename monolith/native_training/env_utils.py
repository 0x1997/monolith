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

import contextlib
import hashlib
import os
import subprocess
import socket

from absl import logging


def check_gdpr_env():
  """Conditionally generates gdpr token."""
  if "SEC_TOKEN_STRING" in os.environ or "SEC_TOKEN_PATH" in os.environ:
    return
  raise EnvironmentError(
      "Fail to get SEC_TOKEN_STRING or SEC_TOKEN_PATH, please run the code with `doas` prefix."
  )


def setup_fake_gdpr_env():
  os.environ["SEC_TOKEN_STRING"] = "FAKE_GDPR_TOKEN"


@contextlib.contextmanager
def reset_to_default_py_env():
  """Resets some env variables into default python env.
  Useful when calling some bytedance system code that requires python 2.
  """
  old_value = None
  var = "PYTHONPATH"
  if var in os.environ:
    old_value = os.environ[var]
  os.environ[var] = (
      "/opt/tiger/ss_lib/python_package/lib/python2.7/site-packages:"
      "/opt/tiger/pyutil")
  try:
    yield
  finally:
    if old_value is not None:
      os.environ[var] = old_value
    else:
      del os.environ[var]


def setup_hdfs_env():
  """Sets up hdfs env."""
  check_gdpr_env()
  if not (os.path.exists("/opt/tiger/yarn_deploy") or os.path.exists("/opt/tiger/cfs_client_deploy")):
    logging.warning(
        ("Unable to set hdfs env. /opt/tiger/yarn_deploy doesn't exist."
         "Will continue without hdfs setup."))
    return

  if os.environ.get('PAAS_CLOUD_ENV') !=  "VOLCANO":
    os.environ["JAVA_HOME"] = "/opt/tiger/jdk/jdk1.8"
    os.environ["HDFS_JDK"] = "/opt/tiger/jdk/jdk8u265-b01/"
    os.environ["HADOOP_HDFS_HOME"] = "/opt/tiger/yarn_deploy/hadoop"
    with reset_to_default_py_env():
      os.environ["CLASSPATH"] = subprocess.check_output(
          "$HADOOP_HDFS_HOME/bin/hadoop classpath --glob", shell=True).decode()
    os.environ["LD_LIBRARY_PATH"] = os.environ[
        "JAVA_HOME"] + "/jre/lib/amd64/server" + ":" + os.environ[
            "HADOOP_HDFS_HOME"] + "/lib/native"
    # Limit how much memory JVM will use to access HDFS
    os.environ["LIBHDFS_OPTS"] = "-Xmx4096m"

  setup_host_ip()

def setup_host_ip():
  #support huoshan tce
  if "MY_HOST_IP_PRE" not in os.environ:
    if "MY_POD_IP" in os.environ:
      logging.warning("host ip: {}, pod ip {}".format(
        os.environ["MY_HOST_IP"], os.environ["MY_POD_IP"]))
      os.environ["MY_HOST_IP_PRE"] = os.environ["MY_HOST_IP"]
      os.environ["MY_HOST_IP"] = os.environ["MY_POD_IP"]
      if "MY_POD_IPV6" in os.environ:
        os.environ["MY_HOST_IPV6"] = os.environ["MY_POD_IPV6"]
        os.environ["MY_HOST_IPV6_PRE"] = os.environ["MY_HOST_IPV6"]
      logging.warning("change to host ip:{}, pod ip {}".format(
        os.environ["MY_HOST_IP"], os.environ["MY_POD_IP"]))
    #default = None
    #return os.environ.get("MY_HOST_IP", socket.gethostbyname(socket.gethostname()) if not default else default)

  if ("PORT" not in os.environ) and ("PORT0" in os.environ):
    os.environ["PORT"] = os.environ["PORT0"]


def generate_psm_from_uuid(uuid: str):
  return "data.aml.monolith_native_training_" + uuid


# This is for hdfs env setup for the instance_dataset_op_test.
def setup_hdfs_env_for_unit_test():
  os.environ["PATH"] = "/opt/tiger/yarn_deploy/hadoop/bin:" + os.environ["PATH"]
  os.environ["HADOOP_HOME"] = "/opt/tiger/yarn_deploy/hadoop"
  os.environ["JAVA_HOME"] = "/opt/tiger/jdk/jdk1.8"
  if os.environ.get("LD_LIBRARY_PATH", None) is not None:
    os.environ[
        "LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":" + os.environ[
            "HADOOP_HOME"] + "/lib/native" + ":" + os.environ[
                "JAVA_HOME"] + "/jre/lib/amd64/server"
  else:
    os.environ["LD_LIBRARY_PATH"] = os.environ[
        "HADOOP_HOME"] + "/lib/native" + ":" + os.environ[
            "JAVA_HOME"] + "/jre/lib/amd64/server"
  os.environ["CLASSPATH"] = subprocess.check_output((
      "export HDFS_JDK=/opt/tiger/jdk/jdk8u265-b01/ && export HADOOP_HOME=/opt/tiger/yarn_deploy/hadoop && "
      "export PYTHONPATH=/opt/tiger/pyutil/:/opt/tiger/ss_lib/python_package/lib/python2.7/site-packages/ && "
      "export JAVA_HOME=/opt/tiger/jdk/jdk1.8 && "
      "$HADOOP_HOME/bin/hadoop classpath --glob"),
                                                    shell=True).decode()
  os.environ["PYTHONPATH"] = ""


def get_zk_auth_data():
  ZK_AUTH = os.getenv('ZK_AUTH', None)
  if ZK_AUTH:
    print("ZK_AUTH", ZK_AUTH)
    return [("digest", ZK_AUTH)]
  return None
