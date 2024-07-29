/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/07/03.
 */
#ifndef SCHEDULER_DISCRETE_HEUN
#define SCHEDULER_DISCRETE_HEUN

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class HeunDiscreteScheduler : public SchedulerBase {
private:
    std::vector<float> prev_derivative;
    std::vector<float> original_sample;

protected:
    uint64_t correction_steps(uint32_t inference_steps_) override;
    std::vector<float> execute_method(
        const float *predict_data_,
        const float *samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit HeunDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
    }

    ~HeunDiscreteScheduler() override = default;
};

// base on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_heun_discrete.py
uint64_t HeunDiscreteScheduler::correction_steps(uint32_t inference_steps_){
    int start_at = 0;
    int end_when = int(scheduler_sigmas.size() - 1);

    std::map<long, int64_t> temp_scheduler_timesteps;
    vector<float> temp_scheduler_sigmas;

    temp_scheduler_timesteps.insert(make_pair(start_at, scheduler_timesteps[start_at]));
    temp_scheduler_sigmas.push_back(scheduler_sigmas[start_at]);
    for (uint32_t i = 1; i < scheduler_sigmas.size() - 1; ++i) {
        uint32_t maj_i = 2 * i - 1;
        uint32_t dup_i = 2 * i;
        temp_scheduler_timesteps.insert(make_pair(maj_i, scheduler_timesteps[i]));
        temp_scheduler_timesteps.insert(make_pair(dup_i, scheduler_timesteps[i]));
        temp_scheduler_sigmas.push_back(scheduler_sigmas[i]);
        temp_scheduler_sigmas.push_back(scheduler_sigmas[i]);
    }
    temp_scheduler_sigmas.push_back(scheduler_sigmas[end_when]);

    scheduler_timesteps = temp_scheduler_timesteps;
    scheduler_sigmas = temp_scheduler_sigmas;

    return inference_steps_ * 2 - 1;
}

std::vector<float> HeunDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    std::vector<float> scaled_sample_(data_size_);
    bool is_first_order_ = (step_index_ % 2 == 0);

    // Heun method:: heun start with euler normal
    float sigma_prev = is_first_order_ ? -1 : scheduler_sigmas[step_index_ - 1];
    float sigma_curs = scheduler_sigmas[step_index_];
    float sigma_next = scheduler_sigmas[step_index_ + 1];
    float sigma_dt = 0;
    {
        sigma_dt = is_first_order_ ? sigma_next - sigma_curs : sigma_curs - sigma_prev;
    }

    // Heun method derivative logic
    if(sigma_next > 0) {
        prev_derivative.resize(data_size_);
        original_sample.resize(data_size_);
        for (int i = 0; i < data_size_; i++) {  // needs to be built in local step, order_ recalculate;
            float curs_derivative = (samples_data_[i] - predict_data_[i]) / sigma_curs;
            if (is_first_order_) {
                scaled_sample_[i] = (samples_data_[i] + curs_derivative * sigma_dt);    // output = sample + derivative_mid * dt
                prev_derivative[i] = curs_derivative;
                original_sample[i] = samples_data_[i];
            } else {
                scaled_sample_[i]  = 0.5f * (prev_derivative[i] + curs_derivative);     // curs_der = (prev_sample - predict_next) / sigma_next
                scaled_sample_[i] = (original_sample[i] + scaled_sample_[i] * sigma_dt);  // output = sample + derivative_mid * dt
            }
        }
    } else {
        // Final round use euler normal to calculate
        for (int i = 0; i < data_size_; i++) {
            scaled_sample_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;     // derivative_out = (sample - predict_sample) / sigma
            scaled_sample_[i] = (samples_data_[i] + scaled_sample_[i] * sigma_dt);      // previous_down = sample + derivative_out * dt
        }
        original_sample.clear();
        prev_derivative.clear();
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_HEUN
