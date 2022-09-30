 Monolith(磐石)

## What is it?

Monolith is a framework for training/serving large scale sparse embedding models.
It is built on the top of TensorFlow 2 and estimator APIs.


## Quick start

### Build from source

Currently, we only support compilation on the Linux.

Frist, download bazel 3.1.0
```bash
wget https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-installer-linux-x86_64.sh && \
  chmod +x bazel-3.1.0-installer-linux-x86_64.sh && \
  ./bazel-3.1.0-installer-linux-x86_64.sh && \
  rm bazel-3.1.0-installer-linux-x86_64.sh
```

Then, prepare a python environment
```bash
pip install -U --user pip numpy wheel packaging requests opt_einsum
pip install -U --user keras_preprocessing --no-deps
```

Finally, you can build any target in the monolith.
For example,
```bash
bazel run //monolith/native_training:demo --output_filter=IGNORE_LOGS
```
