/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_DISCRETE_EULAR_A
#define SCHEDULER_DISCRETE_EULAR_A

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class EularAncestralDiscreteScheduler: public SchedulerBase {
protected:
    std::vector<float> execute_method(
        const float* samples_data_,
        const float* predict_data_,
        int elements_in_batch,
        long step_index_,
        long order_
    ) override;

public:
    explicit EularAncestralDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
    }

    ~EularAncestralDiscreteScheduler() override = default;
};

std::vector<float> EularAncestralDiscreteScheduler::execute_method(
    const float* samples_data_,
    const float* predict_data_,
    int elements_in_batch,
    long step_index_,
    long order_
) {
    SD_UNUSED(order_);

    std::vector<float> scaled_sample(elements_in_batch);

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
    for (int i = 0; i < elements_in_batch; i++) {
        scaled_sample[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;        // derivative_out = (sample - predict_sample) / sigma
        scaled_sample[i] = (predict_data_[i] + scaled_sample[i] * sigma_dt);          // previous_down = sample + derivative_out * dt
        if (sigma_next > 0) {
            scaled_sample[i] = scaled_sample[i] + generate_random_at(0.f) * sigma_up;    // producted_out = previous_down + random_noise * sigma_up
        }
    }

    return scaled_sample;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_EULAR_A
