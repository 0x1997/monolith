export PATH=$PATH:/opt/tiger/bazel/bin/

./configure

# Ignore warnings since it is too big.
bazel build //monolith/native_training:cpu_runner_wrapper --output_filter=DONT_MATCH_ANYTHING

rm  -r -f output
mkdir output
DIR=bazel-bin/monolith/native_training
cp -r -L $DIR/cpu_runner_wrapper.runfiles output/
cp -L $DIR/cpu_runner_wrapper output
