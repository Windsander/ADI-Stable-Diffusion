/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/08/01.
 *
 * FlowSchedulerBase - rectified-flow (flow-matching) scheduler base
 * base on: https://huggingface.co/papers/2209.03003 (rectified flow)
 *          diffusers FlowMatchEulerDiscreteScheduler
 *
 * Coordinate system (DIFFERENT from the discrete/EDM family):
 *   sample = (1 - σ)·x0 + σ·ε,   σ ∈ [0, 1]  (σ=1 pure noise, σ=0 clean)
 *   model output = velocity v;   update: x_prev = x + (σ_next − σ_cur)·v
 * Sigma table: linspace(1, 1/n, n), then shift transform
 *   σ' = s·σ / (1 + (s−1)·σ)     (SD3.5: s=3.0, configurable via --shift)
 * Transformer timestep input = round(σ × 1000) (float on the model side,
 * carried through the framework's int64 contract and re-expanded by the
 * model-side dtype adaptation; sub-integer deviation < 0.05%, acceptable
 * for the first version — see PLAN-v2.0-mmdit.md §3).
 *
 * Interface mapping to the shared denoise loop:
 *   mask()  : x = ε·σ_0 with σ_0 always 1 after shift (s·1/(1+(s−1)) = 1)
 *   scale() : identity (FlowMatch does no input scaling, unlike VP/EDM)
 *   step()  : x0 = sample − σ·v, then execute_method (velocity update)
 * create()'s alphas_cumprod is VP-only and simply goes unused here.
 */
#ifndef SCHEDULER_FLOW_BASE
#define SCHEDULER_FLOW_BASE

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class FlowSchedulerBase: public SchedulerBase {
public:
    explicit FlowSchedulerBase(const SchedulerConfig &scheduler_config_ = DEFAULT_SCHEDULER_CONFIG)
        : SchedulerBase(scheduler_config_) {
    }

    ~FlowSchedulerBase() override = default;

public:
    uint64_t init(uint64_t inference_steps_) override;
    Tensor scale(const Tensor& masker_, int step_index_) override;
    Tensor step(const Tensor& sample_, const Tensor& dnoise_, int step_index_, float random_intensity_ = 1.0f) override;
};

uint64_t FlowSchedulerBase::init(uint64_t inference_steps_) {
    if (inference_steps_ == 0) {
        amon_report(class_exception(EXC_LOG_ERR, "ERROR:: inference_steps_ setting with 0!"));
        return 0;
    }

    const float shift_ = (scheduler_config.scheduler_shift > 0.0f) ?
                         scheduler_config.scheduler_shift : 3.0f;

    // diffusers schedule: raw = linspace(1, σ_min, n) with
    // σ_min = shift(1/num_train_timesteps) (the default table is shifted BEFORE
    // endpoints are taken), then each raw point is shift-transformed again
    float train_f_ = float(scheduler_config.scheduler_training_steps);
    float sigma_end_ = shift_ * (1.0f / train_f_) / (1.0f + (shift_ - 1.0f) / train_f_);
    for (uint64_t i = 0; i < inference_steps_; ++i) {
        float w_ = (inference_steps_ > 1) ? float(i) / float(inference_steps_ - 1) : 0.0f;
        float sigma_raw_ = 1.0f + (sigma_end_ - 1.0f) * w_;
        float sigma_ = shift_ * sigma_raw_ / (1.0f + (shift_ - 1.0f) * sigma_raw_);
        scheduler_timesteps.insert(make_pair(long(i), int64_t(std::llround(double(sigma_) * 1000.0))));
        scheduler_sigmas.push_back(sigma_);
        scheduler_max_sigma = std::max(scheduler_max_sigma, sigma_);
    }
    scheduler_sigmas.push_back(0.0f);
    return correction_steps(inference_steps_);
}

Tensor FlowSchedulerBase::scale(const Tensor& masker_, int step_index_) {
    // FlowMatch applies NO input scaling (diffusers scale_model_input is identity)
    SD_UNUSED(step_index_);
    return TensorHelper::clone<float>(masker_);
}

Tensor FlowSchedulerBase::step(
    const Tensor& sample_,
    const Tensor& dnoise_,
    int step_index_,
    float random_intensity_
) {
    if (step_index_ >= scheduler_timesteps.size()) {
        throw std::runtime_error("from time not found target TimeSteps.");
    }

    TensorShape output_shape_ = sample_.GetTensorTypeAndShapeInfo().GetShape();
    long data_size_ = TensorHelper::get_data_size(sample_);
    auto* sample_data_ = sample_.GetTensorData<float>();
    auto* dnoise_data_ = dnoise_.GetTensorData<float>();
    std::vector<float> predict_data_(data_size_);

    // velocity parameterization: x0 = sample − σ·v  (predict_data_ = x0)
    float sigma_ = scheduler_sigmas[step_index_];
    for (int i = 0; i < data_size_; i++) {
        predict_data_[i] = sample_data_[i] - sigma_ * dnoise_data_[i];
    }

    std::vector<float> latent_value_ = execute_method(
        predict_data_.data(), sample_data_, data_size_, step_index_, random_intensity_
    );
    Tensor result_latent = TensorHelper::create(output_shape_, latent_value_);

    return result_latent;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_FLOW_BASE
