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

from monolith.monitoring import lark_bot
from monolith.native_training.alert import alert_pb2


class Messager:

  def __init__(self, config: alert_pb2.AlertMessageProto):
    if not config.user:
      raise ValueError("User must be specified in AlertMessageProto")
    self._bot = lark_bot.get_omniknight_us_bot()
    self._user_id = self._bot.get_bytedance_user_id(config.user)
    self._chat_id = self._bot.get_chat_id(self._user_id)

  def send(self, message):
    self._bot.send_message_to_chat(self._chat_id, message)
