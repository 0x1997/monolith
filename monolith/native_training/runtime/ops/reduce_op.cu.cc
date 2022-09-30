// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace monolith_tf {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct SetZeroFunctor {
  void operator()(const GPUDevice& d, typename TTypes<T>::Flat out) {
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
  }
};

// To run mnay segment_sum ops on various input lengths and emb dims,
// in one single GPU kernel. We define input group i:
//  * indices with n_i length and s_i segments,
//    s_i <= n_i as input_outer_dim_size;
//    s_i <= output_outer_dim_size also;
//    For example, [1,1,1,2,2,4] with n_i = 5, s_i = 3
//    where output_outer_dim_size >= 4 >= s_i
//  * values with n_i input_outer_dim_size and d_i dims
// The total computation workload is sum n_i * d_i on i.
// For all n_i, we stride with a fixed length k_n, so that
// the same stride can have chance to reduce in local thread.
// The total gpu workload is now the sum on i of
//  [(n_i // k_n) + 1] * d_i
template <typename T, typename Index, int OuterDimTileSize>
__global__ void FusedSortedSegmentSumCustomKernel(
    GpuDeviceArrayStruct<Index> input_outer_dim_sizes_data,
    GpuDeviceArrayStruct<Index> inner_dim_sizes_data,
    GpuDeviceArrayStruct<Index> output_outer_dim_sizes_data,
    GpuDeviceArrayStruct<const Index*> segment_idss_data,  // __restrict__
    GpuDeviceArrayStruct<const T*> inputs_data,            // __restrict__
    GpuDeviceArrayStruct<T*> outputs_data,                 // __restrict__
    GpuDeviceArrayStruct<Index> stripe_offsets_data,
    const Index total_stripe_count) {
  Index* input_outer_dim_sizes =
      GetGpuDeviceArrayOnDevice(&input_outer_dim_sizes_data);
  Index* inner_dim_sizes = GetGpuDeviceArrayOnDevice(&inner_dim_sizes_data);
  Index* output_outer_dim_sizes =
      GetGpuDeviceArrayOnDevice(&output_outer_dim_sizes_data);

  const Index*__restrict__ * segment_idss = GetGpuDeviceArrayOnDevice(&segment_idss_data);
  const T*__restrict__ * inputs = GetGpuDeviceArrayOnDevice(&inputs_data);
  T*__restrict__ * outputs = GetGpuDeviceArrayOnDevice(&outputs_data);
  Index* stripe_offsets = GetGpuDeviceArrayOnDevice(&stripe_offsets_data);

  // if using shared memory
  // Ref:
  // https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/core/kernels/split_lib_gpu.cu.cc#L124
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(Index), unsigned char, smem);
  Index N = input_outer_dim_sizes_data.size;
  Index* ptr = reinterpret_cast<Index*>(smem);
  Index* smem_input_outer_dim_sizes = ptr;
  ptr += N;
  Index* smem_inner_dim_sizes = ptr;
  ptr += N;
  Index* smem_output_outer_dim_sizes = ptr;
  ptr += N;
  Index* smem_stripe_offsets = ptr;
  for (int x = threadIdx.x; x < N; x += blockDim.x) {
    smem_input_outer_dim_sizes[x] = input_outer_dim_sizes[x];
    smem_inner_dim_sizes[x] = inner_dim_sizes[x];
    smem_output_outer_dim_sizes[x] = output_outer_dim_sizes[x];
  }
  for (int x = threadIdx.x; x < N + 1 /*stripe_offsets_data.size*/;
       x += blockDim.x) {
    smem_stripe_offsets[x] = stripe_offsets[x];
  }
  __syncthreads();
  stripe_offsets = smem_stripe_offsets;
  input_outer_dim_sizes = smem_input_outer_dim_sizes;
  inner_dim_sizes = smem_inner_dim_sizes;
  output_outer_dim_sizes = smem_output_outer_dim_sizes;

  Index i = 0;
  for (Index stripe_index : GpuGridRangeX(total_stripe_count)) {
    // Determine the abstract computation unit amd local_stripe_index
    while (stripe_offsets[i + 1] <= stripe_index) ++i;
    Index local_stripe_index = stripe_index - stripe_offsets[i];

    auto input_outer_dim_size = input_outer_dim_sizes[i];
    auto inner_dim_size = inner_dim_sizes[i];
    auto output_outer_dim_size = output_outer_dim_sizes[i];
    if (input_outer_dim_size == 0 || inner_dim_size == 0 ||
        output_outer_dim_size == 0)
      continue;
    auto segment_ids = segment_idss[i];
    auto input = inputs[i];
    auto output = outputs[i];

    // Start computation: segment sum
    const Index segment_offset = local_stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        local_stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    // #pragma unroll
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id) {
          GpuAtomicAdd(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
                 segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    GpuAtomicAdd(output + output_index, sum);
  }
}

// Returns true if the three tensors have valid number of elements
// If shape_input has 0 elements, then we need to have indices and updates with
// exactly 0 elements too, otherwise we should error. If indices has 0 elements
// then updates should also have 0 elements, otherwise we should error.
bool ValidEmptyOutputShape(int64 num_inputs, int64 num_indices,
                           int64 num_updates) {
  if (num_indices == 0 && num_updates == 0) {
    return true;  // regardless of num_inputs ?= 0, covers both cases
  }
  // now we want all 3 tensors to have values
  return (num_inputs != 0 && num_indices != 0 && num_updates != 0);
}

template <typename T, typename Index>
class FusedSegmentSumGPU : public OpKernel {
 public:
  explicit FusedSegmentSumGPU(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
  }

  void Compute(OpKernelContext* ctx) override {
    GPUDevice gpu_device = ctx->eigen_device<GPUDevice>();
    const int OuterDimTileSize = 8;
    Index stripe_offset = 0;  // max as total_stripe_count
    GpuDeviceArrayOnHost<Index> stripe_offsets(ctx, N_ + 1);
    OP_REQUIRES_OK(ctx, stripe_offsets.Init());

    OpInputList indices_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("indices", &indices_list));
    OpInputList updates_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("updates", &updates_list));
    OpInputList shape_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("shape", &shape_list));
    OpOutputList outputs;
    OP_REQUIRES_OK(ctx, ctx->output_list("outputs", &outputs));

    GpuDeviceArrayOnHost<const Index*> indices_ptrs(ctx, N_);
    // TODO(peng): concat then memcpy if necessary
    OP_REQUIRES_OK(ctx, indices_ptrs.Init());
    GpuDeviceArrayOnHost<const T*> updates_ptrs(ctx, N_);
    OP_REQUIRES_OK(ctx, updates_ptrs.Init());
    GpuDeviceArrayOnHost<T*> output_ptrs(ctx, N_);
    OP_REQUIRES_OK(ctx, output_ptrs.Init());

    GpuDeviceArrayOnHost<Index> input_outer_dim_sizes(ctx, N_);
    OP_REQUIRES_OK(ctx, input_outer_dim_sizes.Init());
    GpuDeviceArrayOnHost<Index> inner_dim_sizes(ctx, N_);
    OP_REQUIRES_OK(ctx, inner_dim_sizes.Init());
    GpuDeviceArrayOnHost<Index> output_outer_dim_sizes(ctx, N_);
    OP_REQUIRES_OK(ctx, output_outer_dim_sizes.Init());
    // Shared memory used by four <Index> typed Device array.
    int smem_usage = sizeof(Index) * (4 * N_ + 1);

    for (int i = 0; i < N_; ++i) {
      const Tensor& indices = indices_list[i];
      const Tensor& updates = updates_list[i];
      const Tensor& shape_input = shape_list[i];

      OP_REQUIRES(ctx, indices.shape().dims() >= 1,
                  errors::InvalidArgument(
                      "Indices shape must have rank at least one. Found:",
                      indices.shape().DebugString()));
      OP_REQUIRES(ctx, updates.shape().dims() >= 1,
                  errors::InvalidArgument(
                      "Updates shape must have rank at least one. Found:",
                      updates.shape().DebugString()));

      auto vec = shape_input.flat<Index>();
      TensorShape output_shape;
      OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(vec.data(), vec.size(),
                                                      &output_shape));

      OP_REQUIRES(ctx, ValidEmptyOutputShape(shape_input.NumElements(),
                                             indices.shape().num_elements(),
                                             updates.shape().num_elements()),
                  errors::InvalidArgument(
                      "Indices and updates specified for empty output shape"));

      OP_REQUIRES(ctx, shape_input.dims() == 1,
                  errors::InvalidArgument("Shape must be a vector"));

      //
      Index input_total_size = updates.NumElements();
      auto input_shape = updates.shape();
      Index input_outer_dim_size = input_shape.dim_size(0);
      Index inner_dim_size = 1;
      for (int j = 1; j < input_shape.dims(); ++j)
        inner_dim_size *= input_shape.dim_size(j);
      input_outer_dim_sizes.Set(i, input_outer_dim_size);
      inner_dim_sizes.Set(i, inner_dim_size);
      output_outer_dim_sizes.Set(i, output_shape.dim_size(0));

      stripe_offsets.Set(i, stripe_offset);
      Index input_outer_dim_num_stripe =
          Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));
      stripe_offset += input_outer_dim_num_stripe * inner_dim_size;

      //
      Tensor* out;
      OP_REQUIRES_OK(ctx, outputs.allocate(i, output_shape, &out));
      gpu_device.memset(out->flat<T>().data(), T(0),
                        sizeof(T) * out->NumElements());
      output_ptrs.Set(i, out->flat<T>().data());
      updates_ptrs.Set(i, updates.flat<T>().data());
      indices_ptrs.Set(i, indices.flat<Index>().data());
    }
    const Index total_stripe_count = stripe_offset;
    stripe_offsets.Set(N_, stripe_offset);
    OP_REQUIRES_OK(ctx, stripe_offsets.Finalize());

    OP_REQUIRES_OK(ctx, input_outer_dim_sizes.Finalize());
    OP_REQUIRES_OK(ctx, inner_dim_sizes.Finalize());
    OP_REQUIRES_OK(ctx, output_outer_dim_sizes.Finalize());

    OP_REQUIRES_OK(ctx, indices_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, updates_ptrs.Finalize());
    OP_REQUIRES_OK(ctx, output_ptrs.Finalize());

    auto config = GetGpuLaunchConfig(total_stripe_count, gpu_device);
    GpuLaunchKernel(
        FusedSortedSegmentSumCustomKernel<T, Index, OuterDimTileSize>,
        config.block_count, config.thread_per_block,
        /*shared_memory_size_bytes=*/smem_usage, gpu_device.stream(),
        input_outer_dim_sizes.data(), inner_dim_sizes.data(),
        output_outer_dim_sizes.data(), indices_ptrs.data(), updates_ptrs.data(),
        output_ptrs.data(), stripe_offsets.data(), total_stripe_count);
  }

 private:
  int N_;
};

#define REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX(type, index_type)      \
  REGISTER_KERNEL_BUILDER(Name("MonolithFusedSegmentSum")             \
                              .Device(DEVICE_GPU)                     \
                              .TypeConstraint<type>("T")              \
                              .TypeConstraint<index_type>("Tindices") \
                              .HostMemory("shape"),                   \
                          FusedSegmentSumGPU<type, index_type>)

#define REGISTER_FUSED_SCATTER_ND_KERNEL(type)         \
  REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX(type, int32); \
  REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX(type, int64);

TF_CALL_float(REGISTER_FUSED_SCATTER_ND_KERNEL);
// TF_CALL_GPU_NUMBER_TYPES(REGISTER_FUSED_SCATTER_ND_KERNEL);

#undef REGISTER_FUSED_SCATTER_ND_KERNEL
#undef REGISTER_FUSED_SCATTER_ND_KERNEL_INDEX

REGISTER_OP("MonolithFusedSegmentSum")
    .Input("indices: N * Tindices")
    .Input("updates: N * T")
    .Input("shape: N * Tindices")
    .Output("outputs: N * T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("N: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int N;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
      for (int i = N - 1; i >= 0; --i) {
        shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(i), 1, &indices_shape));
        shape_inference::ShapeHandle updates_shape;
        TF_RETURN_IF_ERROR(
            c->WithRankAtLeast(c->input(N + i), 1, &updates_shape));
        shape_inference::ShapeHandle output_shape;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromShapeTensor(2 * N + i, &output_shape));
        shape_inference::ShapeHandle
            expanded_indices_shape;  // mimic expand_dims(indices, -1)
        TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, c->Vector(1),
                                          &expanded_indices_shape));
        TF_RETURN_IF_ERROR(shape_inference::ScatterNdShapeHelper(
            c, expanded_indices_shape, updates_shape,
            output_shape));  // set shape to output 0
        if (c->input_handle_shapes_and_types(0) == nullptr &&
            c->num_outputs() > 0) {
          c->set_output(i, c->output(0));
        }
      }
      return Status::OK();
    });

}  // namespace monolith_tf
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
