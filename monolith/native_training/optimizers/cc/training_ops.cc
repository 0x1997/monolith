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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS
#include "monolith/native_training/optimizers/cc/training_op_helpers.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace monolith_tf {
struct ApplyAdamom {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::Flat c,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar ada_decay,
                  typename TTypes<float>::ConstScalar mom_decay,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;

    if (update_slots) {
      m.device(d) = mom_decay() * m + (1.0f - mom_decay()) * grad_after_decay;
      v.device(d) = ada_decay() * v + grad_after_decay * grad_after_decay;
      c.device(d) = ada_decay() * c + 1.0f;
    }
    var.device(d) -= m * lr() * (v / c + epsilon()).rsqrt();
  }
};

struct ApplyAdamomV2 {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::Flat c,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar ada_decay,
                  typename TTypes<float>::ConstScalar mom_decay,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;

    if (update_slots) {
      m.device(d) = mom_decay() * m + (1.0f - mom_decay()) * grad_after_decay;
      v.device(d) = ada_decay() * v + grad_after_decay * grad_after_decay;
      c.device(d) = ada_decay() * c + 1.0f;
    }
    var.device(d) -= m * lr() / ((v / c).sqrt() + epsilon());
  }
};

class ApplyAdamomOp : public OpKernel {
 public:
  explicit ApplyAdamomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_v2", &use_v2_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, float>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    Tensor c;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 3, use_exclusive_lock_, sparse, &c));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    OP_REQUIRES(
        ctx, c.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(3)));
    const Tensor& lr = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& ada_decay = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(ada_decay.shape()),
                errors::InvalidArgument("ada_decay is not a scalar: ",
                                        ada_decay.shape().DebugString()));
    const Tensor& mom_decay = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(mom_decay.shape()),
                errors::InvalidArgument("mom_decay is not a scalar: ",
                                        mom_decay.shape().DebugString()));
    const Tensor& epsilon = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& weight_decay = ctx->input(8);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(weight_decay.shape()),
                errors::InvalidArgument("weight_decay is not a scalar: ",
                                        weight_decay.shape().DebugString()));
    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var.shape().DebugString(), " ", m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var.shape().DebugString(), " ", v.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(c.shape()),
                errors::InvalidArgument(
                    "var and accum do not have the same shape",
                    var.shape().DebugString(), " ", c.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const CPUDevice& device = ctx->eigen_device<CPUDevice>();
    if (!use_v2_) {
      ApplyAdamom()(device, var.flat<float>(), m.flat<float>(), v.flat<float>(),
                    c.flat<float>(), lr.scalar<float>(),
                    ada_decay.scalar<float>(), mom_decay.scalar<float>(),
                    epsilon.scalar<float>(), weight_decay.scalar<float>(),
                    grad.flat<float>(), update_slots_);
    } else {
      ApplyAdamomV2()(device, var.flat<float>(), m.flat<float>(),
                      v.flat<float>(), c.flat<float>(), lr.scalar<float>(),
                      ada_decay.scalar<float>(), mom_decay.scalar<float>(),
                      epsilon.scalar<float>(), weight_decay.scalar<float>(),
                      grad.flat<float>(), update_slots_);
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
  bool use_v2_;
};

REGISTER_KERNEL_BUILDER(Name("ResourceApplyAdamom")
                            // .HostMemory("var")
                            // .HostMemory("m")
                            // .HostMemory("v")
                            // .HostMemory("c")
                            .Device(DEVICE_CPU),
                        ApplyAdamomOp);

struct ApplyRmsprop {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;
    if (update_slots) {
      v.device(d) += (grad_after_decay.square() - v) * (1.0f - beta2());
      m.device(d) =
          beta1() * m + (grad_after_decay * lr()) * (v + epsilon()).rsqrt();
      var.device(d) -= m;
    }
  }
};

struct ApplyRmspropV2 {
  void operator()(const CPUDevice& d, typename TTypes<float>::Flat var,
                  typename TTypes<float>::Flat m,
                  typename TTypes<float>::Flat v,
                  typename TTypes<float>::ConstScalar lr,
                  typename TTypes<float>::ConstScalar beta1,
                  typename TTypes<float>::ConstScalar beta2,
                  typename TTypes<float>::ConstScalar epsilon,
                  typename TTypes<float>::ConstScalar weight_decay,
                  typename TTypes<float>::ConstFlat grad, bool update_slots) {
    auto grad_after_decay = weight_decay() * var + grad;
    if (update_slots) {
      v.device(d) = beta2() * v + grad_after_decay.square();
      //      m.device(d) = beta1() * m + (grad_after_decay * lr()) * (v +
      //      epsilon()).rsqrt();
      m.device(d) =
          beta1() * m + (grad_after_decay * lr()) / (v.sqrt() + epsilon());
      var.device(d) -= m;
    }
  }
};

class ApplyRmspropOp : public OpKernel {
 public:
  explicit ApplyRmspropOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("update_slots", &update_slots_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_v2", &use_v2_));
  }

  void Compute(OpKernelContext* ctx) override {
    const bool sparse = false;
    auto locks = MaybeLockVariableInputMutexesInOrder<CPUDevice, float>(
        ctx, use_exclusive_lock_, sparse, {0, 1});
    Tensor var;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 0, use_exclusive_lock_, sparse, &var));
    Tensor m;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 1, use_exclusive_lock_, sparse, &m));
    Tensor v;
    OP_REQUIRES_OK(ctx, GetInputTensorFromVariable<CPUDevice, float>(
                            ctx, 2, use_exclusive_lock_, sparse, &v));
    OP_REQUIRES(
        ctx, var.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(0)));
    OP_REQUIRES(
        ctx, m.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(1)));
    OP_REQUIRES(
        ctx, v.IsInitialized(),
        errors::FailedPrecondition(
            "Attempting to use uninitialized variables: ", requested_input(2)));
    const Tensor& lr = ctx->input(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr.shape()),
                errors::InvalidArgument("lr is not a scalar: ",
                                        lr.shape().DebugString()));
    const Tensor& beta1 = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    const Tensor& beta2 = ctx->input(5);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    const Tensor& epsilon = ctx->input(6);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));
    const Tensor& weight_decay = ctx->input(7);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(weight_decay.shape()),
                errors::InvalidArgument("weight_decay is not a scalar: ",
                                        weight_decay.shape().DebugString()));
    const Tensor& grad = ctx->input(8);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const CPUDevice& device = ctx->eigen_device<CPUDevice>();
    if (!use_v2_) {
      ApplyRmsprop()(device, var.flat<float>(), m.flat<float>(),
                     v.flat<float>(), lr.scalar<float>(), beta1.scalar<float>(),
                     beta2.scalar<float>(), epsilon.scalar<float>(),
                     weight_decay.scalar<float>(), grad.flat<float>(),
                     update_slots_);
    } else {
      ApplyRmspropV2()(device, var.flat<float>(), m.flat<float>(),
                       v.flat<float>(), lr.scalar<float>(),
                       beta1.scalar<float>(), beta2.scalar<float>(),
                       epsilon.scalar<float>(), weight_decay.scalar<float>(),
                       grad.flat<float>(), update_slots_);
    }

    MaybeForwardRefInputToRefOutput(ctx, 0, 0);
  }

 private:
  bool use_exclusive_lock_;
  bool update_slots_;
  bool use_v2_;
};

REGISTER_KERNEL_BUILDER(Name("ResourceApplyRmsprop").Device(DEVICE_CPU),
                        ApplyRmspropOp);

template <bool is_resource>
ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

template <>
ShapeHandle ShapeOrHandleShape<true>(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  // If a resource input is missing shape information, we should return
  // UnknownShape rather than the shape of the input, which is a scalar
  // resource handle.
  return c->UnknownShape();
}

// Handle the gradient and, if <is_sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
template <bool is_sparse, bool is_resource>
static Status HandleGradAndIndicesInputs(InferenceContext* c, int grad_idx,
                                         ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape<is_resource>(c, grad_idx);
  if (!is_sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));
  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status ApplyAdamomShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape</*is_resource=*/true>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 3), &s));  // c
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));              // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));  // ada_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));  // mom_decay
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));  // weight_decay
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, /*is_resource=*/true>(
          c, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceApplyAdamom")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("c: resource")
    .Input("learning_rate: float")
    .Input("ada_decay: float")
    .Input("mom_decay: float")
    .Input("epsilon: float")
    .Input("weight_decay: float")
    .Input("grad: float")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .Attr("use_v2: bool = false")
    .SetShapeFn(ApplyAdamomShapeFn);

static Status ApplyRmspropShapeFn(InferenceContext* c) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape</*is_resource=*/true>(c, 0);  // var
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(
      c->Merge(s, ShapeOrHandleShape</*is_resource=*/true>(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));              // lr
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));  // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));  // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));  // epsilon
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));  // weight_decay
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs</*is_sparse=*/false, /*is_resource=*/true>(
          c, 8 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("ResourceApplyRmsprop")
    .Input("var: resource")
    .Input("m: resource")
    .Input("v: resource")
    .Input("learning_rate: float")
    .Input("beta1: float")
    .Input("beta2: float")
    .Input("epsilon: float")
    .Input("weight_decay: float")
    .Input("grad: float")
    .Attr("use_locking: bool = false")
    .Attr("update_slots: bool = true")
    .Attr("use_v2: bool = false")
    .SetShapeFn(ApplyRmspropShapeFn);
}  // namespace monolith_tf
}  // namespace tensorflow
