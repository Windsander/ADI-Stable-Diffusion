/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/07/03.
 */
#ifndef SCHEDULER_DISCRETE_LCM
#define SCHEDULER_DISCRETE_LCM

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class LCMDiscreteScheduler : public SchedulerBase {
private:
    RandomGenerator lcm_random;

protected:
    std::vector<float> execute_method(
        const float *predict_data_,
        const float *samples_data_,
        long data_size_,
        long step_index_,
        long order_
    ) override;

public:
    explicit LCMDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
        lcm_random.seed(0);
    }

    ~LCMDiscreteScheduler() override = default;
};

// base on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lcm.py
std::vector<float> LCMDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    long order_
) {
    SD_UNUSED(order_);

    std::vector<float> scaled_sample_(data_size_);

    // LCM method:: sigma get, only next sigma be needed
    float sigma_next = scheduler_sigmas[step_index_ + 1]; // sigma_next prev_timestep(caused by inference is a reversed working flow)

    // LCM method:: current noise decrees
    for (int i = 0; i < data_size_; i++) {
        if (sigma_next > 0) {
            scaled_sample_[i] = (predict_data_[i] + lcm_random.next() * sigma_next);       // producted_out = predict_sample + random_noise * sigma_next
        } else {
            scaled_sample_[i] = (predict_data_[i]);
        }
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_LCM
