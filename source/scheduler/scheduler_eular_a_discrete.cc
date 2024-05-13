/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_EULAR_A_DISCRETE
#define SCHEDULER_EULAR_A_DISCRETE

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class EularAncestralDiscreteScheduler: public SchedulerBase {
private:
    std::normal_distribution<float> random_style;

protected:
    float generate_random_at(float timestep) override;
    float generate_sigma_at(float timestep) override;

    std::vector<float> execute_method(
        const vector<float>& dnoised_data_,
        long step_index_
    ) override;

public:
    explicit EularAncestralDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
        random_style =std::normal_distribution<float>(0.0, 1.0);
    }

    ~EularAncestralDiscreteScheduler() override{
        random_style.reset();
    }
};

float EularAncestralDiscreteScheduler::generate_random_at(float timestep_) {
    SD_UNUSED(timestep_);
    return random_style(random_generator);
}

float EularAncestralDiscreteScheduler::generate_sigma_at(float timestep_) {
    int low_idx  = static_cast<int>(std::floor(timestep_));
    int high_idx = static_cast<int>(std::ceil(timestep_));
    float w      = timestep_ - static_cast<float>(low_idx);      // divide always 1
    float sigma  = (1.0f - w) * alphas_cumprod[low_idx] + w * alphas_cumprod[high_idx];
    return sigma;
}

std::vector<float> EularAncestralDiscreteScheduler::execute_method(
    const vector<float>& dnoised_data_,
    long step_index_
) {
    std::vector<float> scaled_sample(dnoised_data_.size());

    // Euler method:: sigma get
    float sigma_curs = scheduler_sigmas[step_index_];
    float sigma_next = scheduler_sigmas[step_index_ + 1];
    float sigma_up = 0;
    float sigma_dt = 0;
    {
        float sigma_curs_pow = std::powf(sigma_curs, 2);
        float sigma_next_pow = std::powf(sigma_next, 2);
        float sigma_up_pow = sigma_next_pow * (sigma_curs_pow - sigma_next_pow) / sigma_curs_pow;
        float sigma_down_pow = sigma_next_pow - sigma_up_pow;
        sigma_up = (sigma_up_pow < 0) ? -std::sqrtf(std::abs(sigma_up_pow)) : std::sqrtf(sigma_up_pow);
        sigma_dt = (sigma_down_pow < 0) ?
                   -std::sqrtf(std::abs(sigma_down_pow)) - sigma_curs :
                   std::sqrtf(sigma_down_pow) - sigma_curs;
    }

    // Euler method:: current noise decrees
    for (int i = 0; i < dnoised_data_.size(); i++) {
        scaled_sample[i] = (dnoised_data_[i] - scaled_sample[i]) / sigma_curs;        // derivative_out = (sample - predict_sample) / sigma
        scaled_sample[i] = (dnoised_data_[i] + scaled_sample[i] * sigma_dt);          // previous_down = sample + derivative_out * dt
        if (sigma_next > 0) {
            scaled_sample[i] = scaled_sample[i] + generate_random_at(0.f) * sigma_up;    // producted_out = previous_down + random_noise * sigma_up
        }
    }

    return scaled_sample;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_EULAR_A_DISCRETE
