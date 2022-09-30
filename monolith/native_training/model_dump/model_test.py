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

import os 
import copy
import getpass
from absl import flags
import tensorflow as tf

from monolith.native_training.data.datasets import PBDataset
from monolith.native_training.model_dump.model import DummpedModel
from monolith.estimator import EstimatorSpec, Estimator, RunConfig
from monolith.native_training.model_dump.dump_utils import DumpUtils


model_name = os.path.basename(os.path.dirname(__file__))
file_name = "monolith/experimental/training_instance/examplebatch.data"
feature_list = "monolith/native_training/model_dump/feature_list.conf"
model_dir = "{}/tmp/{}/ckpt".format(os.getenv("HOME"), model_name)

FLAGS = flags.FLAGS

class DummpedModelTest(tf.test.TestCase):

  def test_model(self):
    FLAGS.feature_list = feature_list
    FLAGS.lagrangex_header = True
    FLAGS.data_type = 'examplebatch'
    model = DummpedModel()
    model.file_name = file_name

    config_json = DumpUtils().get_config()
    est_config = RunConfig.from_json(config_json)
    est_config.model_dir=model_dir
    est_config.is_local=True

    estimator = Estimator(model, est_config)
    if FLAGS.mode == tf.estimator.ModeKeys.EVAL:
      estimator.evaluate()
    elif FLAGS.mode == tf.estimator.ModeKeys.TRAIN:
      estimator.train()
  
  def input_fn(self):
    FLAGS.feature_list = feature_list
    FLAGS.lagrangex_header = True
    FLAGS.data_type = 'examplebatch'
    est_config = RunConfig(model_dir=model_dir, is_local=True,
                           num_ps=3, dense_only_save_checkpoints_steps=10,
                           checkpoints_max_to_keep=3)
    model = DummpedModel()
    model.file_name = file_name
    
    config = tf.compat.v1.ConfigProto()
    config.graph_options.rewrite_options.disable_meta_optimizer = True
    with tf.compat.v1.Session(config=config) as sess:
      features = model.input_fn(mode='train')
      initializer = tf.compat.v1.get_collection('mkiter')[0]
      sess.run(initializer)

      for _ in range(3):
        self.assertEqual(len(sess.run(fetches=features)), 79)


if __name__ == "__main__":
  tf.compat.v1.disable_eager_execution()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  tf.test.main()
