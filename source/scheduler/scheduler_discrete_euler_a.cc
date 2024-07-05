/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_DISCRETE_EULER_A
#define SCHEDULER_DISCRETE_EULER_A

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class EulerAncestralDiscreteScheduler : public SchedulerBase {
private:
    RandomGenerator euler_a_random;

protected:
    std::vector<float> execute_method(
        const float *predict_data_,
        const float *samples_data_,
        long data_size_,
        long step_index_,
        long order_
    ) override;

public:
    explicit EulerAncestralDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
        euler_a_random.seed(0);
    }

    ~EulerAncestralDiscreteScheduler() override = default;
};

std::vector<float> EulerAncestralDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    long order_
) {
    SD_UNUSED(order_);

    std::vector<float> scaled_sample_(data_size_);

    // Euler method:: sigma get
    float sigma_curs = scheduler_sigmas[step_index_];
    float sigma_next = scheduler_sigmas[step_index_ + 1];
    float sigma_up = 0;
    float sigma_dt = 0;
    {
        float sigma_curs_pow = sigma_curs * sigma_curs;
        float sigma_next_pow = sigma_next * sigma_next;
        float sigma_up_numerator = sigma_next_pow * (sigma_curs_pow - sigma_next_pow);
        sigma_up = std::min(sigma_next, std::sqrt(sigma_up_numerator / sigma_curs_pow));
        sigma_dt = std::sqrt(sigma_next_pow - sigma_up * sigma_up) - sigma_curs;
    }

    // Euler Ancestral method:: current noise decrees
    for (int i = 0; i < data_size_; i++) {
        scaled_sample_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;         // derivative_out = (sample - predict_sample) / sigma
        scaled_sample_[i] = (samples_data_[i] + scaled_sample_[i] * sigma_dt);          // previous_down = sample + derivative_out * dt
        if (sigma_next > 0) {
            scaled_sample_[i] = scaled_sample_[i] + euler_a_random.next() * sigma_up;    // producted_out = previous_down + random_noise * sigma_up
        }
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_EULER_A
