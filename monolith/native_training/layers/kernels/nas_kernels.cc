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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace monolith_tf {

const float MAXEXP = 80;
const float MINEXP = -80;

float sigmoid(float x) {
  if (x > MAXEXP) {
    return 1;
  } else if (x < MINEXP) {
    return 0;
  } else {
    return 1.0 / (1 + std::exp(-x));
  }
}

float softplus(float x) {
  if (x > MAXEXP) {
    return x;
  } else if (x < MINEXP) {
    return 0;
  } else {
    return log(1 + exp(x));
  }
}

class BernoulliGateOp : public OpKernel {
 public:
  explicit BernoulliGateOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    // {softplus, clip, none}
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ste_type", &ste_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_logistic", &use_logistic_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *alpha_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("alpha", &alpha_tensor));

    auto alpha_flat = alpha_tensor->flat<float>();
    Tensor *sampled_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, alpha_tensor->shape(), &sampled_tensor));
    auto sampled_flat = sampled_tensor->flat<float>();

    Tensor *proba_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, alpha_tensor->shape(), &proba_tensor));
    auto proba_flat = proba_tensor->flat<float>();

    for (int i = 0; i < alpha_flat.size(); ++i) {
      float proba = calc_proba(alpha_flat(i));
      proba_flat(i) = proba;
      if (use_logistic_) {
        sampled_flat(i) = proba >= 0.5 ? 1 : 0;
      } else {
        float u = uniform_(generator_);
        sampled_flat(i) = u <= proba ? 1 : 0;
      }
    }
  }

 private:
  bool use_logistic_;
  float temperature_ = 1.0;
  std::string op_type_;
  std::string ste_type_;
  std::default_random_engine generator_;
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};

  float calc_proba(float alpha) {
    float logit;
    if (use_logistic_) {
      float u = 0;
      while (u == 0) {
        u = uniform_(generator_);
      }
      logit = (std::log(u) - std::log(1 - u) + alpha) / temperature_;
    } else {
      logit = alpha;
    }

    return sigmoid(logit);
  }
};

class BernoulliGateGradOp : public OpKernel {
 public:
  explicit BernoulliGateGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    // {softplus, clip, none}
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ste_type", &ste_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_logistic", &use_logistic_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    auto grad_flat = grad_tensor->flat<float>();

    const Tensor *alpha_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("alpha", &alpha_tensor));
    auto alpha_flat = alpha_tensor->flat<float>();

    const Tensor *proba_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("proba", &proba_tensor));
    auto proba_flat = proba_tensor->flat<float>();

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, grad_tensor->shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    if (ste_type_ == "softplus") {
      for (int i = 0; i < grad_flat.size(); ++i) {
        output_flat(i) = grad_flat(i) * softplus(alpha_flat(i));
      }
    } else if (ste_type_ == "clip") {
      for (int i = 0; i < grad_flat.size(); ++i) {
        output_flat(i) = grad_flat(i) * clip(softplus(alpha_flat(i)), 0, 1);
      }
    } else {
      for (int i = 0; i < grad_flat.size(); ++i) {
        output_flat(i) = grad_flat(i) * calc_grad(proba_flat(i));
      }
    }
  }

 private:
  bool use_logistic_;
  float temperature_ = 1.0;
  std::string ste_type_;

  inline float calc_grad(float proba) {
    if (use_logistic_) {
      return proba * (1 - proba) / temperature_;
    } else {
      return proba * (1 - proba);
    }
  }

  float clip(float x, float min, float max) {
    if (x >= max) {
      return max;
    } else if (x < min) {
      return min;
    } else {
      return x;
    }
  }
};

class DiscreteGateOp : public OpKernel {
 public:
  explicit DiscreteGateOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_one_hot", &is_one_hot_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_gumbel", &use_gumbel_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));

    unsigned long seed = std::time(0);
    generator_.seed(seed);
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *alpha_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("alpha", &alpha_tensor));
    auto alpha_flat = alpha_tensor->flat<float>();

    Tensor *sampled_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, alpha_tensor->shape(), &sampled_tensor));
    auto sampled_flat = sampled_tensor->flat<float>();
    sampled_flat.setZero();

    Tensor *proba_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, alpha_tensor->shape(), &proba_tensor));
    auto proba_flat = proba_tensor->flat<float>();

    // 1. calc weights_
    std::vector<float> weights_;
    if (use_gumbel_) {
      for (int i = 0; i < alpha_flat.size(); ++i) {
        float value = (alpha_flat(i) + ev_dist_(generator_)) / temperature_;
        weights_.push_back(std::exp(value > MAXEXP ? MAXEXP : value));
      }
    } else {
      for (int i = 0; i < alpha_flat.size(); ++i) {
        float value = alpha_flat(i) / temperature_;
        weights_.push_back(std::exp(value > MAXEXP ? MAXEXP : value));
      }
    }

    // 2. construct discrete_dist (softmax)
    std::discrete_distribution<int> discrete_dist(weights_.begin(),
                                                  weights_.end());

    // 3. sampling
    if (use_gumbel_) {
      int index = 0;
      float max_value;
      for (int i = 0; i < weights_.size(); ++i) {
        if (i == 0) {
          max_value = weights_[i];
        } else if (weights_[i] > max_value) {
          max_value = weights_[i];
          index = i;
        }
      }
      sampled_flat(index) = 1.0;
    } else {
      int index = discrete_dist(generator_);
      sampled_flat(index) = 1.0;
    }

    // 4. fill probas
    int i = 0;
    for (double proba : discrete_dist.probabilities()) {
      proba_flat(i++) = proba;
    }
  }

 private:
  bool is_one_hot_ = false, use_gumbel_ = false;
  float temperature_ = 1;
  std::default_random_engine generator_;
  std::extreme_value_distribution<double> ev_dist_;
};

class DiscreteGateGradOp : public OpKernel {
 public:
  explicit DiscreteGateGradOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_one_hot", &is_one_hot_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    auto grad_flat = grad_tensor->flat<float>();

    const Tensor *sampled_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("sampled", &sampled_tensor));
    auto sampled_flat = sampled_tensor->flat<float>();

    const Tensor *proba_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("proba", &proba_tensor));
    auto proba_flat = proba_tensor->flat<float>();

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, grad_tensor->shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();
    output_flat.setZero();

    if (is_one_hot_) {
      int i = 0;
      for (int k = 0; k < sampled_flat.size(); ++k) {
        if (sampled_flat(k) > 0.5) {
          i = k;
          break;
        }
      }

      for (int j = 0; j < grad_flat.size(); ++j) {
        int factor = i == j ? 1 : 0;
        output_flat(j) = grad_flat(i) * proba_flat(i) *
                         (factor - proba_flat(j)) / temperature_;
      }
    } else {
      for (int i = 0; i < grad_flat.size(); ++i) {
        float part = grad_flat(i) * proba_flat(i);
        for (int j = 0; j < grad_flat.size(); ++j) {
          int factor = i == j ? 1 : 0;
          output_flat(j) += part * (factor - proba_flat(j));
        }
      }

      for (int j = 0; j < grad_flat.size(); ++j) {
        output_flat(j) /= temperature_;
      }
    }
  }

 private:
  bool is_one_hot_ = false;
  float temperature_ = 1;
};

class DiscreteTruncatedGateOp : public OpKernel {
 public:
  explicit DiscreteTruncatedGateOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_first_dim", &drop_first_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_gumbel", &use_gumbel_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));

    unsigned long seed = std::time(0);
    generator_.seed(seed);
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *alpha_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("alpha", &alpha_tensor));
    auto alpha_flat = alpha_tensor->flat<float>();

    Tensor *sampled_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {alpha_tensor->dim_size(0) -
                                                 (drop_first_dim_ ? 1 : 0)},
                                             &sampled_tensor));
    auto sampled_flat = sampled_tensor->flat<float>();
    sampled_flat.setZero();

    Tensor *proba_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(1, alpha_tensor->shape(), &proba_tensor));
    auto proba_flat = proba_tensor->flat<float>();

    // 1. calc weights_
    std::vector<float> weights_;
    if (use_gumbel_) {
      for (int i = 0; i < alpha_flat.size(); ++i) {
        float value = (alpha_flat(i) + ev_dist_(generator_)) / temperature_;
        weights_.push_back(std::exp(value > MAXEXP ? MAXEXP : value));
      }
    } else {
      for (int i = 0; i < alpha_flat.size(); ++i) {
        float value = alpha_flat(i) / temperature_;
        weights_.push_back(std::exp(value > MAXEXP ? MAXEXP : value));
      }
    }

    // 2. construct discrete_dist (softmax)
    std::discrete_distribution<int> discrete_dist(weights_.begin(),
                                                  weights_.end());

    // 3. fill probas
    int i = 0;
    std::vector<double> probas_;
    for (double proba : discrete_dist.probabilities()) {
      proba_flat(i++) = proba;
      probas_.push_back(proba);
    }
    std::sort(probas_.begin(), probas_.end(),
              [](const auto &x, const auto &y) { return x > y; });
    float accumulate = 0, pivot = 0;
    for (const auto &proba : probas_) {
      accumulate += proba;
      if (accumulate >= threshold_) {
        pivot = proba;
        break;
      }
    }

    // 4. sampling
    for (int i = 0; i < proba_flat.size(); ++i) {
      if (proba_flat(i) >= pivot) {
        if (drop_first_dim_) {
          if (i > 0) {
            sampled_flat(i - 1) = 1.0;
          }
        } else {
          sampled_flat(i) = 1.0;
        }
      }
    }
  }

 private:
  float threshold_ = 1.0;
  bool use_gumbel_ = false, drop_first_dim_ = false;
  float temperature_ = 1;
  std::default_random_engine generator_;
  std::extreme_value_distribution<double> ev_dist_;
  // std::uniform_real_distribution<double> uniform_(0.0, q.0);
};

class DiscreteTruncatedGateGradOp : public OpKernel {
 public:
  explicit DiscreteTruncatedGateGradOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("drop_first_dim", &drop_first_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor *grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("grad", &grad_tensor));
    auto grad_flat = grad_tensor->flat<float>();

    const Tensor *sampled_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("sampled", &sampled_tensor));
    auto sampled_flat = sampled_tensor->flat<float>();

    const Tensor *proba_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("proba", &proba_tensor));
    auto proba_flat = proba_tensor->flat<float>();

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, proba_tensor->shape(), &output_tensor));
    auto output_flat = output_tensor->flat<float>();
    output_flat.setZero();

    for (int i = 0; i < grad_flat.size(); ++i) {
      // sampled_flat(i) is binary, 0/1
      if (sampled_flat(i) < 0.5) {
        continue;
      }
      int k = drop_first_dim_ ? i + 1 : i;
      float part = grad_flat(i) * proba_flat(k);
      for (int j = 0; j < proba_flat.size(); ++j) {
        int factor = k == j ? 1 : 0;
        output_flat(j) += part * (factor - proba_flat(j));
      }
    }

    for (int j = 0; j < output_flat.size(); ++j) {
      output_flat(j) /= temperature_;
    }
  }

 private:
  float threshold_ = 1.0;
  float temperature_ = 1;
  bool drop_first_dim_ = false;
};

namespace {

REGISTER_KERNEL_BUILDER(Name("BernoulliGate").Device(DEVICE_CPU),
                        BernoulliGateOp)

REGISTER_KERNEL_BUILDER(Name("BernoulliGateGrad").Device(DEVICE_CPU),
                        BernoulliGateGradOp)

REGISTER_KERNEL_BUILDER(Name("DiscreteGate").Device(DEVICE_CPU), DiscreteGateOp)

REGISTER_KERNEL_BUILDER(Name("DiscreteGateGrad").Device(DEVICE_CPU),
                        DiscreteGateGradOp)

REGISTER_KERNEL_BUILDER(Name("DiscreteTruncatedGate").Device(DEVICE_CPU),
                        DiscreteTruncatedGateOp)

REGISTER_KERNEL_BUILDER(Name("DiscreteTruncatedGateGrad").Device(DEVICE_CPU),
                        DiscreteTruncatedGateGradOp)

}  // namespace
}  // namespace monolith_tf
}  // namespace tensorflow
