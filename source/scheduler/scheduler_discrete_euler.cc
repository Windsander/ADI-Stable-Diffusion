/*
 * Copyright (c) 2018-2050 SD_Scheduler - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef SCHEDULER_DISCRETE_EULER
#define SCHEDULER_DISCRETE_EULER

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class EulerDiscreteScheduler : public SchedulerBase {
protected:
    std::vector<float> execute_method(
        const float *predict_data_,
        const float *samples_data_,
        long data_size_,
        long step_index_,
        long order_
    ) override;

public:
    explicit EulerDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_){
    }

    ~EulerDiscreteScheduler() override = default;
};

std::vector<float> EulerDiscreteScheduler::execute_method(
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
    float sigma_dt = 0;
    {
        sigma_dt = sigma_next - sigma_curs;
    }

    // Euler method:: current noise decrees
    for (int i = 0; i < data_size_; i++) {
        scaled_sample_[i] = (samples_data_[i] - predict_data_[i]) / sigma_curs;         // derivative_out = (sample - predict_sample) / sigma
        scaled_sample_[i] = (samples_data_[i] + scaled_sample_[i] * sigma_dt);          // previous_down = sample + derivative_out * dt
    }

    return scaled_sample_;
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_EULER
