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

from monolith.native_training.alert import alert_pb2
from monolith.native_training.alert import alert_manager

from absl import flags
from absl import app

FLAGS = flags.FLAGS

FLAGS.monolith_alert_proto = """
alert_message {
  user: "leqi.zou"
}
training_alert {
  prefix: "fakeone"
}
kafka_alert {
  topic: "f100_recommend_online_joiner_postclk_instance_pb"
  group: "house_rank_cvr_1652335628_worker_suffix"
}
check_interval_sec:5

"""
alert_manager.get_default_alert_manager().join()
