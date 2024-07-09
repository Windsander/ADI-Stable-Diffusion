/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2024/07/09.
 *
 * Denoising Diffusion Probabilistic Models
 */
#ifndef SCHEDULER_DISCRETE_DDPM
#define SCHEDULER_DISCRETE_DDPM

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class DDPMDiscreteScheduler: public SchedulerBase {
private:
    RandomGenerator ddpm_random;

private:
    float get_lms_coefficient(long order, long t, int current_order);

protected:
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        long order_
    ) override;

public:
    explicit DDPMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
        ddpm_random.seed(0);
    }

    ~DDPMDiscreteScheduler() override = default;
};

std::vector<float> DDPMDiscreteScheduler::execute_method(
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
    float factor_a = 0;
    float factor_b = 0;
    {
        float sigma_curs_pow = sigma_curs * sigma_curs;
        float sigma_next_pow = sigma_next * sigma_next;
        sigma_up = std::sqrt(sigma_next_pow / sigma_curs_pow * (sigma_curs_pow - sigma_next_pow) / (sigma_next_pow + 1));
        factor_a = (sigma_next_pow / sigma_curs_pow) * std::sqrt((sigma_curs_pow + 1) / (sigma_next_pow + 1));
        factor_b = (1 - sigma_next_pow / sigma_curs_pow) / std::sqrt(sigma_next_pow + 1);
    }

    // Euler Ancestral method:: current noise decrees
    for (int i = 0; i < data_size_; i++) {
        scaled_sample_[i] = samples_data_[i] * factor_a + predict_data_[i] * factor_b;         // derivative_out = (sample - predict_sample) / sigma
        if (sigma_next > 0) {
            scaled_sample_[i] = scaled_sample_[i] + ddpm_random.next() * sigma_up;    // producted_out = previous_down + random_noise * sigma_up
        }
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_DDPM
