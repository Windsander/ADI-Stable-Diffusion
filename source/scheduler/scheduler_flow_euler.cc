/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2026/08/01.
 *
 * Flow Euler (rectified-flow deterministic euler, flow-matching)
 * base on: diffusers FlowMatchEulerDiscreteScheduler (stochastic_sampling=false)
 *
 * update: x_prev = x + (σ_next − σ_cur)·v,  velocity v recovered from the
 * base-converted x0 as v = (sample − x0)/σ_cur (== raw model output).
 */
#ifndef SCHEDULER_FLOW_EULER_H
#define SCHEDULER_FLOW_EULER_H

#include "scheduler_flow_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class FlowEulerScheduler: public FlowSchedulerBase {
protected:
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit FlowEulerScheduler(SchedulerConfig scheduler_config_ = {}) : FlowSchedulerBase(scheduler_config_) {
    }

    ~FlowEulerScheduler() override = default;
};

std::vector<float> FlowEulerScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    float sigma_curs_ = scheduler_sigmas[size_t(step_index_)];
    float sigma_next_ = scheduler_sigmas[size_t(step_index_) + 1];
    float sigma_dt_ = sigma_next_ - sigma_curs_;

    std::vector<float> prev_samples_(data_size_, 0.0f);
    for (long i = 0; i < data_size_; i++) {
        float velocity_ = (samples_data_[i] - predict_data_[i]) / sigma_curs_;
        prev_samples_[i] = samples_data_[i] + velocity_ * sigma_dt_;
    }
    return prev_samples_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_FLOW_EULER_H
