### Example
First, start the debugging server using your own model directory and make sure that the model is training.
```
bazel run -c opt :debugging_server -- -port=6666 -model_dir=hdfs://haruna/yoda/galaxy/offline/xujinghao/iptv_v12
```
If you want to debugging variables, use the following command:
```
bazel run -c opt :debugging_client -- -port=6666 -type=debugging_variables -variable_names=global_step:0,mlp/dense_3/dense_3/bias/Adagrad:0
```
If you want to debugging features, use the following command:
```
bazel run -c opt :debugging_client -- -port=6666 -type=debugging_features -feature_name=slot_300 -feature_ids=5409530699325912000,5409699480722490192
bazel run -c opt :debugging_client -- -port=6666 -type=debugging_features -feature_names=slot_300,slot_300 -feature_ids=5409530699325912000,5409699480722490192
```
